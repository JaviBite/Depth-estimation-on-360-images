import torch
import sys
import os
import io
import utils
from torch.utils.data import Dataset

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
        self.transform = transform
        self.data = []

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
        img = utils.getColorImage(img_name)
        w, h = img.shape  

        dep = utils.getDepthImage(depth_name)
        w, h = dep.shape 

        image = torch.from_numpy(img).type('torch.FloatTensor').reshape(1, h, w) / 255.0
        depth = torch.from_numpy(dep).type('torch.FloatTensor').reshape(1, h, w)
        sample = {'image': image, 'depth': depth}

        if self.transform:
            sample = self.transform(sample)

        return sample