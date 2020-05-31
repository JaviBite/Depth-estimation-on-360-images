import torch
import sys
import os
import io
import utils
import cv2
import numpy
from torch.utils.data import Dataset

def loadImage(file):
    image = numpy.array(cv2.imread(file))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def loadDepth(file):
    depthImg = numpy.array(cv2.imread(file, cv2.IMREAD_ANYDEPTH))
    return depthImg

class ThreeD60(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, txt_file, root_dir, transform=None):
        """
        Args:
            txt_file (string): Path to the txt file with images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.data = []
        self.height = 256
        self.width = 512

        with open (txt_file, 'r') as f:
            for line in f.readlines():
                images = line.split(' ')
                color_image = images[0]
                depth_image = images[3]
                self.data.append({'image': color_image, 'depth': depth_image})
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.data[idx]['image'])
        depth_name = os.path.join(self.root_dir,
                                self.data[idx]['depth'])

        img = loadImage(img_name)
        dep = loadDepth(depth_name)

        imgTensor = utils.imageToTensor(img)
        depthTensor = utils.depthToTensor(dep)

        sample = {'image': imgTensor, 'depth': depthTensor, 'name': self.data[idx]['image']}

        return sample