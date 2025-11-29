import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange, repeat
import math

from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
import numpy as np


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        return (
            x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight
            + self.bias
        )


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()

        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class EDFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(EDFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )

        self.fft = nn.Parameter(
            torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1))
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)

        x_patch = rearrange(
            x,
            "b c (h patch1) (w patch2) -> b c h w patch1 patch2",
            patch1=self.patch_size,
            patch2=self.patch_size,
        )
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(
            x_patch,
            "b c h w patch1 patch2 -> b c (h patch1) (w patch2)",
            patch1=self.patch_size,
            patch2=self.patch_size,
        )

        return x


class SS2D(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.0, **kwargs):
        """
        Упрощённый 2D-блок внимания без mamba-ssm.
        Вместо селективного сканирования используем стандартный MultiheadAttention по пространству (H*W).
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(self, x: torch.Tensor, **kwargs):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # (B, C, H, W) -> (B, L, C), где L = H*W
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)
        y, _ = self.attn(x_flat, x_flat, x_flat)
        y = self.out_proj(y)
        if self.dropout is not None:
            y = self.dropout(y)
        # (B, L, C) -> (B, C, H, W)
        y = y.permute(0, 2, 1).contiguous().view(B, C, H, W)
        return y


##########################################################################
class EVS(nn.Module):
    def __init__(
        self,
        dim,
        ffn_expansion_factor=3,
        bias=False,
        LayerNorm_type="WithBias",
        att=False,
        idx=3,
        patch=128,
    ):
        super(EVS, self).__init__()

        self.att = att
        self.idx = idx
        if self.att:
            self.norm1 = LayerNorm(dim)
            self.attn = SS2D(d_model=dim, patch=patch)

        self.norm2 = LayerNorm(dim)
        self.ffn = EDFFN(dim, ffn_expansion_factor, bias)

        self.kernel_size = (patch, patch)

    def forward(self, x):
        if self.att:

            if self.idx % 2 == 1:
                x = torch.flip(x, dims=(-2, -1)).contiguous()
            if self.idx % 2 == 0:
                x = torch.transpose(x, dim0=-2, dim1=-1).contiguous()

            x = x + self.attn(self.norm1(x))

        x = x + self.ffn(self.norm2(x))

        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode="bilinear", align_corners=False),
            nn.Conv2d(n_feat, n_feat * 2, 3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(n_feat, n_feat // 2, 3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        return self.body(x)


##########################################################################
##---------- EVSSM -----------------------
class EVSSM(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[6, 6, 12],
        ffn_expansion_factor=3,
        bias=False,
    ):
        super(EVSSM, self).__init__()

        self.encoder = True

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential()
        for i in range(num_blocks[0]):
            block = EVS(
                dim=dim,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                att=True,
                idx=i,
                patch=384,
            )
            self.encoder_level1.add_module(f"block{i}", block)

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential()
        for i in range(num_blocks[1]):
            block = EVS(
                dim=dim * 2,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                att=True,
                idx=i,
                patch=192,
            )
            self.encoder_level2.add_module(f"block{i}", block)

        self.down2_3 = Downsample(int(dim * 2**1))
        self.encoder_level3 = nn.Sequential()
        for i in range(num_blocks[2]):
            block = EVS(
                dim=dim * 4,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                att=True,
                idx=i,
                patch=96,
            )
            self.encoder_level3.add_module(f"block{i}", block)

        self.decoder_level3 = nn.Sequential()
        for i in range(num_blocks[2]):
            block = EVS(
                dim=dim * 4,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                att=True,
                idx=i,
                patch=96,
            )
            self.decoder_level3.add_module(f"block{i}", block)

        self.up3_2 = Upsample(int(dim * 2**2))

        self.decoder_level2 = nn.Sequential()
        for i in range(num_blocks[1]):
            block = EVS(
                dim=dim * 2,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                att=True,
                idx=i,
                patch=192,
            )
            self.decoder_level2.add_module(f"block{i}", block)

        self.up2_1 = Upsample(int(dim * 2**1))

        self.decoder_level1 = nn.Sequential()
        for i in range(num_blocks[0]):
            block = EVS(
                dim=dim,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                att=True,
                idx=i,
                patch=384,
            )
            self.decoder_level1.add_module(f"block{i}", block)

        self.output = nn.Conv2d(
            int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        out_dec_level3 = self.decoder_level3(out_enc_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)

        inp_dec_level2 = inp_dec_level2 + out_enc_level2

        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)

        inp_dec_level1 = inp_dec_level1 + out_enc_level1
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1
