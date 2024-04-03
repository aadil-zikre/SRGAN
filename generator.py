import torch.nn as nn
import torch.nn.functional as F

class ganGenerator(nn.Module):
    def __init__(self):
        super(ganGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4)
        self.prelu1 = nn.PReLU()
        self.GRB1 = GeneratorResidualBlock()
        self.GRB2 = GeneratorResidualBlock()
        self.GRB3 = GeneratorResidualBlock()
        self.GRB4 = GeneratorResidualBlock()
        self.GRB5 = GeneratorResidualBlock()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pxlshuffle1 = nn.PixelShuffle(2)
        self.prelu2 = nn.PReLU()
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pxlshuffle2 = nn.PixelShuffle(2)
        self.prelu3 = nn.PReLU()
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.prelu1(x1)

        x2 = self.GRB1(x1)
        x2 = self.GRB2(x2)
        x2 = self.GRB3(x2)
        x2 = self.GRB4(x2)
        x2 = self.GRB5(x2)

        x2 = self.conv2(x2)
        x2 = self.bn1(x2)
        x3 = x1 + x2

        x3 = self.conv3(x3)
        x3 = self.pxlshuffle1(x3)
        x3 = self.prelu2(x3)
        x3 = self.conv4(x3)
        x3 = self.pxlshuffle2(x3)
        x4 = self.prelu3(x3)

        x5 = self.conv5(x4)

        return x5

class GeneratorResidualBlock(nn.Module):
    def __init__(self):
        super(GeneratorResidualBlock, self).__init__()
        # convolution
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        # batchnorm
        self.bn1 = nn.BatchNorm2d(64)
        # prelu
        self.prelu1 = nn.PReLU()
        #convolution
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        # batchnorm
        self.bn2 = nn.BatchNorm2d(64)
        # prelu
        self.prelu2 = nn.PReLU()
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.prelu2(out)
        return out + x