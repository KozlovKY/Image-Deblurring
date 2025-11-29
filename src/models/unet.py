import torch.nn as nn
import torch.nn.functional as F
import torch


def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
    )
    return conv


def up_conv(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv1 = double_conv(3, 64)
        self.down_conv2 = double_conv(64, 128)
        self.down_conv3 = double_conv(128, 256)
        self.down_conv4 = double_conv(256, 512)
        self.down_conv5 = double_conv(512, 1024)

        self.up_trans1 = up_conv(1024, 512)
        self.up_conv1 = double_conv(1024, 512)

        self.up_trans2 = up_conv(512, 256)
        self.up_conv2 = double_conv(512, 256)

        self.up_trans3 = up_conv(256, 128)
        self.up_conv3 = double_conv(256, 128)

        self.up_trans4 = up_conv(128, 64)
        self.up_conv4 = double_conv(128, 64)

        self.out = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        # encoder
        x1 = self.down_conv1(x)
        x2 = self.max_pool(x1)

        x3 = self.down_conv2(x2)
        x4 = self.max_pool(x3)

        x5 = self.down_conv3(x4)
        x6 = self.max_pool(x5)

        x7 = self.down_conv4(x6)
        x8 = self.max_pool(x7)

        x9 = self.down_conv5(x8)

        # decoder
        x = self.up_trans1(x9)
        x = torch.cat([x, x7], 1)
        x = self.up_conv1(x)

        x = self.up_trans2(x)
        x = torch.cat([x, x5], 1)
        x = self.up_conv2(x)

        x = self.up_trans3(x)
        x = torch.cat([x, x3], 1)
        x = self.up_conv3(x)

        x = self.up_trans4(x)
        x = torch.cat([x, x1], 1)
        x = self.up_conv4(x)

        x = self.out(x)
        return x
