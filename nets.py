import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np
import torch
from collections import OrderedDict
from layers import *
import utils as u
import cv2

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

class ResNetShpere(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetShpere, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = SphereConv2D(num_input_images * 3, 64, stride=2, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = SphereMaxPool2D(stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SphereConv2D):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                SphereConv2D(self.inplanes, planes * block.expansion, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
        
    # def _make_layer(self, block, planes, blocks, stride=1):
    #     downsample = None
    #     if stride != 1 or self.inplanes != planes * block.expansion:
    #         print("test")
    #         downsample = nn.Sequential(
    #             SphereConv2D(self.inplanes, planes * block.expansion,
    #                         stride=stride, bias=False),
    #             nn.BatchNorm2d(planes * block.expansion),
    #         )

    #     layers = []
    #     layers.append(block(self.inplanes, planes, stride, downsample))
    #     self.inplanes = planes * block.expansion
    #     for i in range(1, blocks):
    #         layers.append(block(self.inplanes, planes))

    #     return nn.Sequential(*layers)


# def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
#     """Constructs a ResNet model.
#     Args:
#         num_layers (int): Number of resnet layers. Must be 18 or 50
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         num_input_images (int): Number of frames stacked as input
#     """
#     assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
#     blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
#     block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
#     model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

#     if pretrained:
#         loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
#         loaded['conv1.weight'] = torch.cat(
#             [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
#         model.load_state_dict(loaded)
#     return model

def resnet_sphereconv(num_layers, pretrained=False, num_input_images=1):
    assert num_layers in [18], "Can only run with 18 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: BasicBlockSph}[num_layers]
    model = ResNetShpere(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        print("Loading petrained conv1 model")
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1, sphere=False):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        # if num_input_images > 1:
        #     self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        # else:
        if not sphere:
            self.encoder = resnets[num_layers](pretrained)
        else:
            self.encoder = resnet_sphereconv(num_layers, pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)

        self.features.append(self.encoder.relu(x))
        # for x in self.features:
        #     print(x.shape)
        # print("- END -")
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        # for x in self.features:
        #     print(x.shape)
        # print("- END -")
        self.features.append(self.encoder.layer2(self.features[-1]))
        # for x in self.features:
        #     print(x.shape)
        # print("- END -")
        
        self.features.append(self.encoder.layer3(self.features[-1]))
        # for x in self.features:
        #     print(x.shape)
        # print("- END -")
        self.features.append(self.encoder.layer4(self.features[-1]))
        # for x in self.features:
        #     print(x.shape)
        # print("- END -")

        return self.features


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, sphere=False):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out, sphere=sphere)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out, sphere=sphere)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels, sphere=sphere)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):

            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]

            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs