import torch.nn as nn
import torch.nn.functional as F
import torch


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        # in_channels - количество каналов после конкатенации (умноженное на 2 из-за skip connections)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # Использование ConvTranspose2d для повышения разрешения
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Коррекция размеров, если необходимо
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # Объединение через конкатенацию по каналам
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DeblurringResUNet(nn.Module):
    def __init__(self, n_channels=3, bilinear=True):
        super(DeblurringResUNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.res1 = ResidualBlock(64)
        self.down1 = Down(64, 128)
        self.res2 = ResidualBlock(128)
        self.down2 = Down(128, 256)
        self.res3 = ResidualBlock(256)
        self.down3 = Down(256, 512)
        self.res4 = ResidualBlock(512)

        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.res5 = ResidualBlock(1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.se1 = SEBlock(512 // factor)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.se2 = SEBlock(256 // factor)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.se3 = SEBlock(128 // factor)
        self.up4 = Up(128, 64, bilinear)
        self.se4 = SEBlock(64)

        self.outc = OutConv(64, n_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.res1(x1)
        x2 = self.down1(x1)
        x2 = self.res2(x2)
        x3 = self.down2(x2)
        x3 = self.res3(x3)
        x4 = self.down3(x3)
        x4 = self.res4(x4)
        x5 = self.down4(x4)
        x5 = self.res5(x5)

        x = self.up1(x5, x4)
        x = self.se1(x)
        x = self.up2(x, x3)
        x = self.se2(x)
        x = self.up3(x, x2)
        x = self.se3(x)
        x = self.up4(x, x1)
        x = self.se4(x)
        x = self.outc(x)
        return x
