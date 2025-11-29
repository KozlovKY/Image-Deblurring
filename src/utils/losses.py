import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from torchvision.models.vgg import VGG19_Weights
from .loss_utils import weighted_loss

_reduction_modes = ["none", "mean", "sum"]


def _l1_loss_base(pred, target):
    return F.l1_loss(pred, target, reduction="none")


def _mse_loss_base(pred, target):
    return F.mse_loss(pred, target, reduction="none")


@weighted_loss
def l1_loss(pred, target):
    return _l1_loss_base(pred, target)


@weighted_loss
def mse_loss(pred, target):
    return _mse_loss_base(pred, target)


class PerceptualLoss(nn.Module):
    def __init__(self, feature_layers=[0, 5, 10, 19, 28], device="cuda"):
        super(PerceptualLoss, self).__init__()
        vgg19 = models.vgg19(weights=VGG19_Weights.DEFAULT).features.to(device)
        self.layers = nn.ModuleList([vgg19[i] for i in feature_layers]).eval()
        for param in self.layers.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        loss = 0.0
        for layer in self.layers:
            x = layer(x)
            y = layer(y)
            loss += F.l1_loss(x, y)
        return loss


class CombinedLoss(nn.Module):
    def __init__(self, perceptual_weight=0.1):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss()
        self.perceptual_weight = perceptual_weight

    def forward(self, output, target):
        mse = self.mse_loss(output, target)
        perceptual = self.perceptual_loss(output, target)
        loss = mse + self.perceptual_weight * perceptual
        return loss


class EdgeLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction="mean"):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[0.05, 0.25, 0.4, 0.25, 0.05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1).cuda()

        self.weight = loss_weight

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode="replicate")
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4
        filtered = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y, weight=None, **kwargs):
        loss = l1_loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss * self.weight


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction="mean"):
        super(L1Loss, self).__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. "
                f"Supported ones are: {_reduction_modes}"
            )

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        loss = _l1_loss_base(pred, target)
        if weight is not None:
            loss = loss * weight
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return self.loss_weight * loss


class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction="mean"):
        super(MSELoss, self).__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. "
                f"Supported ones are: {_reduction_modes}"
            )

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        loss = _mse_loss_base(pred, target)
        if weight is not None:
            loss = loss * weight
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return self.loss_weight * loss


class PSNRLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction="mean", toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == "mean"
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.0
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.0

            pred, target = pred / 255.0, target / 255.0
            pass
        assert len(pred.size()) == 4

        return (
            self.loss_weight
            * self.scale
            * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
        )


class FFTLoss(nn.Module):
    """L1 loss in frequency domain with FFT.

    Args:
        loss_weight (float): Loss weight for FFT loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction="mean"):
        super(FFTLoss, self).__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. "
                f"Supported ones are: {_reduction_modes}"
            )

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (..., C, H, W). Predicted tensor.
            target (Tensor): of shape (..., C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (..., C, H, W). Element-wise
                weights. Default: None.
        """
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)

        loss = _l1_loss_base(pred_fft, target_fft)
        if weight is not None:
            loss = loss * weight
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return self.loss_weight * loss
