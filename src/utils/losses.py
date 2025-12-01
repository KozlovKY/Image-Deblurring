import torch
import torch.nn as nn


class PSNRLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction="mean"):
        super().__init__()
        assert reduction == "mean"
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        mse = ((pred - target) ** 2).mean(dim=(1, 2, 3))

        psnr_loss = 10 * torch.log10(mse.clamp(min=1e-8)).mean()

        return self.loss_weight * psnr_loss
