import matplotlib.pyplot as plt
import cv2
import numpy
import torch
import sys
import torchvision

class CaptureOnSetAttribute:
    def __setattr__(self, attr, value):
        # our hook to do something
        #print(f'set value of {attr} to {value}')
        # actually set the attribute the normal way after
        super().__setattr__(attr, value)

def imageToTensor(image):
    image = image.transpose(2, 0, 1)
    tensor = torch.from_numpy(image).type(torch.float32) / 255.0
    return tensor

def tensorToImage(tensor):
    # Convert tensor
    out = tensor.cpu().numpy()

    # Transform tensor to image
    out = out.transpose(1, 2, 0)
    image = numpy.array(out)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def depthToTensor(depth, dataset='3d60'):
    h, w = depth.shape
    if dataset == '3d60':
        depth[numpy.isinf(depth)] = 0.0   # 5000 is inf
        depth[depth > 7.5] = 0.0   # 5000 is inf
        
    depthT = depth.reshape(1, h, w)
    tensor = torch.from_numpy(depthT).type(torch.float32)
    if dataset == 'normal':
        tensor = tensor / 255.0
    elif dataset == '3d60':
        tensor = tensor / 7.5
    return tensor

def tensorToDepth(tensor):
    # Convert tensor
    out = tensor.cpu().detach().numpy()
    c, h, w = out.shape

    # Transform tensor to image
    out = out.reshape(h, w)
    depth = numpy.array(out)
    return depth

def getDepthImage(image, dataset='3d60'):
    image = image
    if dataset == 'normal':
        cmap = plt.cm.magma
        norm = plt.Normalize(vmin=0.0, vmax=1.0)
        image = cmap(norm(image))
    elif dataset == '3d60':
        image[numpy.isinf(image)] = 0.0   # 5000 is inf
        image[image > 1.0] = 0.0   # 5000 is inf
        cmap = plt.cm.magma
        norm = plt.Normalize(vmin=0.0, vmax=1.0)
        image = cmap(norm(image))

    image = image[:,:,0:3]
    image = numpy.float32(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image[:,:,0:3]

def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)