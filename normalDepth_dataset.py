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
    depthImg = numpy.array(cv2.imread(file, cv2.IMREAD_GRAYSCALE))
    return depthImg

class NormalDepth(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir):
        """
        Args:
            txt_file (string): Path to the txt file with images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir # normalDepthDataset/test/LR/
            

    def __len__(self):
        return 1505 if 'train' in self.root_dir else 500

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                        'outleft/' + str(idx+1).zfill(4) + '.png')
        #print(img_name)
        depth_name = os.path.join(self.root_dir,
                        'depthmap/' + str(idx+1).zfill(4) + '.png')

        img = loadImage(img_name)
        dep = loadDepth(depth_name)

        imgTensor = utils.imageToTensor(img)
        depthTensor = utils.depthToTensor(dep, 'normal')

        sample = {'image': imgTensor, 'depth': depthTensor}

        return sample