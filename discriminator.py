import torch.nn as nn

class ganDiscriminator(nn.Module):
    def __init__(self):
        super(ganDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1)
        self.lrelu1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.lrelu2 = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.lrelu3 = nn.LeakyReLU()
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2)
        self.bn4 = nn.BatchNorm2d(128)
        self.lrelu4 = nn.LeakyReLU()

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.lrelu5 = nn.LeakyReLU()
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2)
        self.bn6 = nn.BatchNorm2d(256)
        self.lrelu6 = nn.LeakyReLU()

        self.conv7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.lrelu7 = nn.LeakyReLU()
        self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2)
        self.bn8 = nn.BatchNorm2d(512)
        self.lrelu8 = nn.LeakyReLU()

        # self.flat = nn.Flatten()

        # self.dense9 = nn.Linear(in_features=60*124*512, out_features=1024, bias=True)
        # self.lrelu9 = nn.LeakyReLU()

        # self.dense10 = nn.Linear(in_features=1024, out_features=1, bias=True)
        # self.sigmoid10 = nn.Sigmoid()

        self.adaptive_pool_1 = nn.AdaptiveAvgPool2d(1)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1)
        self.lrelu9 = nn.LeakyReLU()
        self.conv10 = nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1) 
        self.sigmoid9 = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.lrelu1(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.lrelu2(x2)

        x3 = self.conv3(x2)
        x3 = self.bn3(x3)
        x3 = self.lrelu3(x3)

        x4 = self.conv4(x3)
        x4 = self.bn4(x4)
        x4 = self.lrelu4(x4)

        x5 = self.conv5(x4)
        x5 = self.bn5(x5)
        x5 = self.lrelu5(x5)

        x6 = self.conv6(x5)
        x6 = self.bn6(x6)
        x6 = self.lrelu6(x6)

        x7 = self.conv7(x6)
        x7 = self.bn7(x7)
        x7 = self.lrelu7(x7)

        x8 = self.conv8(x7)
        x8 = self.bn8(x8)
        x8 = self.lrelu8(x8)

        x9 = self.adaptive_pool_1(x8)
        x9 = self.conv9(x9)
        x9 = self.lrelu9(x9)

        x10 = self.conv10(x9)
        x10 = self.sigmoid9(x10.view(x10.size()[0]))

        # x9 = self.flat(x8)
        # x10 = self.dense9(x9)
        # x10 = self.lrelu9(x10)
        # x11 = self.dense10(x10)
        # x11 = self.sigmoid10(x11)

        x11 = x10

        return x11