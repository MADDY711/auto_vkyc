# mobilefacenet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))

class MobileFaceNet(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()
        self.conv1 = ConvBlock(3, 64, 3, 2, 1)
        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1)
        self.conv2 = ConvBlock(64, 128, 3, 2, 1)
        self.dw_conv2 = ConvBlock(128, 128, 3, 1, 1)
        self.conv3 = ConvBlock(128, 128, 3, 2, 1)
        self.dw_conv3 = ConvBlock(128, 128, 3, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.conv2(x)
        x = self.dw_conv2(x)
        x = self.conv3(x)
        x = self.dw_conv3(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.bn(x)
        return F.normalize(x)

