import torch.nn as nn
from torchvision.models.resnet import resnet101
import torch.nn.functional as F
import torch

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 12, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 12, kernel_size=5, stride=1, padding=2),
            nn.ReLU())

        self.layer3 = nn.Sequential(
            nn.Conv2d(12, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(6, 1, kernel_size=5, stride=1, padding=2),
            nn.ReLU())

        self.drop_out = nn.Dropout()

        self.upsampling = F.interpolate

    def forward(self, x):
        out = self.conv1(x)

        _, _, h, w = out.shape
        out = self.upsampling(out, size=(h*2,w*2))

        out = self.layer4(out)

        return out


class NetTwo(nn.Module):
    def __init__(self):
        super(NetTwo, self).__init__()

        self.upsampling = F.interpolate

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU())

        self.convds2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.convds3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.ups1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU())

        self.ups2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU())

        self.prediction = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=5, stride=1, padding=2),
            nn.ReLU())

        self.drop_out = nn.Dropout()
        

    def forward(self, x):
        out = self.conv1(x)
        #print(out.shape); a = input("layer1")

        out = self.convds2(out)

        out = self.convds3(out)        

        # out = self.drop_out(out)

        out = self.ups1(out)
        _, _, h, w = out.shape
        out = self.upsampling(out, size=(h*2,w*2))

        out = self.ups2(out)
        _, _, h, w = out.shape
        out = self.upsampling(out, size=(h*2,w*2))

        out = self.prediction(out)

        return out


class NetOne(nn.Module):
    def __init__(self):
        super(NetOne, self).__init__()
        self.preProcess = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU())

        self.upsampling = F.interpolate

    def forward(self, x):
        out = self.conv1(x)

        _, _, h, w = out.shape
        out = self.upsampling(out, size=(h*2,w*2))

        out = self.layer4(out)

        return out