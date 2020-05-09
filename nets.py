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
        # print(out.shape)
        # a = input("hola")

        # out = self.conv2(out)
        # # print(out.shape)
        # # a = input("hola")

        # out = self.layer1(out)
        # # print(out.shape)
        # # a = input("hola")

        # out = self.layer2(out)
        # # print(out.shape)
        # # a = input("hola")

        # out = self.drop_out(out)
        # # print(out.shape)
        # # a = input("Upsampling:")

        # _, _, h, w = out.shape
        # out = self.upsampling(out, size=(h*5,w*5) )
        # # print(out.shape)
        # # a = input("hola")

        _, _, h, w = out.shape
        out = self.upsampling(out, size=(h*2,w*2))
        # print(out.shape)
        # a = input("hola")

        # out = self.layer3(out)
        # # print(out.shape)
        # # a = input("hola")

        out = self.layer4(out)
        # print(out.shape)
        # a = input("hola")

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
        # print(out.shape)
        # a = input("hola")

        # out = self.conv2(out)
        # # print(out.shape)
        # # a = input("hola")

        # out = self.layer1(out)
        # # print(out.shape)
        # # a = input("hola")

        # out = self.layer2(out)
        # # print(out.shape)
        # # a = input("hola")

        # out = self.drop_out(out)
        # # print(out.shape)
        # # a = input("Upsampling:")

        # _, _, h, w = out.shape
        # out = self.upsampling(out, size=(h*5,w*5) )
        # # print(out.shape)
        # # a = input("hola")

        _, _, h, w = out.shape
        out = self.upsampling(out, size=(h*2,w*2))
        # print(out.shape)
        # a = input("hola")

        # out = self.layer3(out)
        # # print(out.shape)
        # # a = input("hola")

        out = self.layer4(out)
        # print(out.shape)
        # a = input("hola")

        return out