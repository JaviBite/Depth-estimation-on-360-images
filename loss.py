# Hyperparameters
import torchvision
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch.nn as nn
#import eval_cnn as eval
import threeD60_dataset as dataset
import cv2
import utils
import time
from functools import *

class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()
    # L2 norm
    def forward(self, grad_fake, grad_real):
        
        num_pixels = reduce(lambda x, y: x*y, grad_real.size())
        return torch.sum( torch.pow(grad_real-grad_fake, 2) ) / num_pixels

class Cost(nn.Module):
    def __init__(self):
        super(Cost, self).__init__()
    # L1 norm
    def forward(self, grad_fake, grad_real):
        return torch.mean( torch.abs(grad_real-grad_fake) )