import torch
import torch.nn as nn


def double_conv2d(in_channels, out_channels, kernel=3):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=(kernel - 1) // 2),
        nn.Tanh(),
        nn.Conv2d(out_channels, out_channels, kernel, padding=(kernel - 1) // 2),
        nn.Tanh(),
    )


def double_conv1d(in_channels, out_channels, kernel=3, dropout=0):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel, padding=(kernel - 1) // 2),
        nn.BatchNorm1d(out_channels),
        nn.Tanh(),
        # nn.Dropout(dropout),
        nn.Conv1d(out_channels, out_channels, kernel, padding=(kernel - 1) // 2),
        nn.BatchNorm1d(out_channels),
        nn.Tanh(),
        nn.Dropout(dropout),
    )


class UNet1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, channels=64, kernel=3, dropout=0, input_drop=0):
        super().__init__()
        self.in_channels = in_channels

        self.dconv_down1 = double_conv1d(in_channels, channels, kernel, dropout)
        self.dconv_down2 = double_conv1d(channels, 2 * channels, kernel, dropout)
        self.dconv_down3 = double_conv1d(2 * channels, 4 * channels, kernel, dropout)
        self.dconv_down4 = double_conv1d(4 * channels, 8 * channels, kernel, dropout)

        self.maxpool = nn.MaxPool1d(2)
        self.upsample = nn.Upsample(scale_factor=2)

        self.dconv_up3 = double_conv1d((4 + 8) * channels, 4 * channels, kernel, dropout)
        self.dconv_up2 = double_conv1d((2 + 4) * channels, 2 * channels, kernel, dropout)
        self.dconv_up1 = double_conv1d((2 + 1) * channels, channels, kernel, dropout)

        self.conv_last = nn.Conv1d(channels, out_channels, kernel, padding=(kernel - 1) // 2)
        # self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)

    def forward(self, x):
        baseline = x[:, [1]]  # heuristic baseline
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        # x = self.dropout(x)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        # x = self.dropout(x)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        # x = self.dropout(x)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        # x = self.dropout(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        # x = self.dropout(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        # x = self.dropout(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x) + baseline

        return out


class UNetCustom(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, channels=64, kernel=3, dropout=0, input_drop=0):
        super().__init__()
        self.in_channels = in_channels

        self.dconv_down1 = double_conv2d(in_channels, channels, kernel)
        self.dconv_down2 = double_conv2d(channels, 2 * channels, kernel)
        self.dconv_down3 = double_conv2d(2 * channels, 4 * channels, kernel)
        self.dconv_down4 = double_conv2d(4 * channels, 8 * channels, kernel)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2)

        self.dconv_up3 = double_conv2d((4 + 8) * channels, 4 * channels, kernel)
        self.dconv_up2 = double_conv2d((2 + 4) * channels, 2 * channels, kernel)
        self.dconv_up1 = double_conv2d((2 + 1) * channels, channels, kernel)

        self.conv_last = nn.Conv2d(channels, out_channels, kernel, padding=(kernel - 1) // 2)
        self.dropout = nn.Dropout2d(dropout)
        self.input_drop = nn.Dropout2d(input_drop)

    def forward(self, x):
        x = x[:, : self.in_channels]
        x[:, :, 3:] = self.input_drop(x[:, :, 3:])
        pp = x[:, [3]]  # previous pressure
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        x = self.dropout(x)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        x = self.dropout(x)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        x = self.dropout(x)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = self.dropout(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = self.dropout(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = self.dropout(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x) + pp

        return out


class UNetCustomNoPP(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, channels=64, dropout=0):
        super().__init__()
        self.in_channels = in_channels

        self.dconv_down1 = double_conv1d(in_channels, channels)
        self.dconv_down2 = double_conv1d(channels, 2 * channels)
        self.dconv_down3 = double_conv1d(2 * channels, 4 * channels)
        self.dconv_down4 = double_conv1d(4 * channels, 8 * channels)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.dconv_up3 = double_conv1d((4 + 8) * channels, 4 * channels)
        self.dconv_up2 = double_conv1d((2 + 4) * channels, 2 * channels)
        self.dconv_up1 = double_conv1d((2 + 1) * channels, channels)

        self.conv_last = nn.Conv2d(channels, out_channels, 1)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = x[:, : self.in_channels]
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        x = self.dropout(x)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        x = self.dropout(x)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        x = self.dropout(x)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = self.dropout(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = self.dropout(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = self.dropout(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        channels_factor = 1
        self.dconv_down1 = double_conv1d(in_channels, 64 * channels_factor)
        self.dconv_down2 = double_conv1d(64 * channels_factor, 128 * channels_factor)
        self.dconv_down3 = double_conv1d(128 * channels_factor, 256 * channels_factor)
        self.dconv_down4 = double_conv1d(256 * channels_factor, 512 * channels_factor)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.dconv_up3 = double_conv1d((256 + 512) * channels_factor, 256 * channels_factor)
        self.dconv_up2 = double_conv1d((128 + 256) * channels_factor, 128 * channels_factor)
        self.dconv_up1 = double_conv1d((128 + 64) * channels_factor, 64 * channels_factor)

        self.conv_last = nn.Conv2d(64 * channels_factor, out_channels, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        x = self.dropout(x)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        x = self.dropout(x)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        x = self.dropout(x)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = self.dropout(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = self.dropout(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = self.dropout(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out


class UNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, channels=64, kernel=3, dropout=0):
        super().__init__()
        self.in_channels = in_channels

        self.dconv_down1 = double_conv2d(in_channels, channels, kernel)
        self.dconv_down2 = double_conv2d(channels, 2 * channels, kernel)
        self.dconv_down3 = double_conv2d(2 * channels, 4 * channels, kernel)
        self.dconv_down4 = double_conv2d(4 * channels, 8 * channels, kernel)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2)

        self.dconv_up3 = double_conv2d((4 + 8) * channels, 4 * channels, kernel)
        self.dconv_up2 = double_conv2d((2 + 4) * channels, 2 * channels, kernel)
        self.dconv_up1 = double_conv2d((2 + 1) * channels, channels, kernel)

        self.conv_last = nn.Conv2d(channels, out_channels, kernel, padding=(kernel - 1) // 2)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        baseline = x[:, [1]]  # heuristic baseline
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        x = self.dropout(x)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        x = self.dropout(x)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        x = self.dropout(x)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = self.dropout(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = self.dropout(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = self.dropout(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x) + baseline

        return out


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        baseline = x[:, [1]]  # heuristic baseline
        out = baseline

        return out
