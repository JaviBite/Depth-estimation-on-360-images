# Hyperparameters
import torchvision
import torch as th
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch.nn as nn
from torch.nn import Parameter
#import eval_cnn as eval
import threeD60_dataset as dataset
import cv2
import utils
import time
from functools import *
from numpy import pi
import math

class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()
    # L2 norm
    def forward(self, grad_fake, grad_real):
        
        num_pixels = reduce(lambda x, y: x*y, grad_real.size())
        return th.sum( th.pow(grad_real-grad_fake, 2) ) / num_pixels

class SphereMSE(nn.Module):
    def __init__(self, h, w):
        super(SphereMSE, self).__init__()
        self.h, self.w = h, w
        weight = th.zeros(1, 1, h, w)
        theta_range = th.linspace(0, pi, steps=h + 1)
        dtheta = pi / h
        dphi = 2 * pi / w
        for theta_idx in range(h):
            weight[:, :, theta_idx, :] = dphi * (math.sin(theta_range[theta_idx]) + math.sin(theta_range[theta_idx+1]))/2 * dtheta
        self.weight = Parameter(weight, requires_grad=False)

    def forward(self, out, target):
        return th.sum((out - target) ** 2 * self.weight) / out.size(0)