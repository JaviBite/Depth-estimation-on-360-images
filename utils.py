import matplotlib.pyplot as plt
import cv2
import numpy
import torch
import sys
import torchvision


def imageToTensor(image):
    image = image.transpose(2, 0, 1)
    tensor = torch.from_numpy(image).type(torch.float32) / 255.0
    return tensor

def tensorToImage(tensor):
    # Convert tensor
    out = tensor.numpy()

    # Transform tensor to image
    out = out.transpose(1, 2, 0)
    image = numpy.array(out)
    return image

def depthToTensor(depth):
    h, w = depth.shape
    depthT = depth.reshape(1, h, w)
    tensor = torch.from_numpy(depthT).type(torch.float32)
    return tensor

def tensorToDepth(tensor):
    # Convert tensor
    out = tensor.numpy()
    c, h, w = out.shape

    # Transform tensor to image
    out = out.reshape(h, w)
    depth = numpy.array(out)
    return depth

def getDepthImage(image):
    image[numpy.isinf(image)] = 0.0   # 5000 is inf
    image[image > 5000] = 0.0   # 5000 is inf
    cmap = plt.cm.magma
    norm = plt.Normalize(vmin=0.0, vmax=7.5)
    image = cmap(norm(image))
    return image[:,:,0:3]

def show_depths_grid(depths, title):
        b, c, h, w = depths.size()
        depths[torch.isinf(depths)] = 0.0
        canvas = torch.zeros(b, 3, h, w)        
        for i in range(b):
            image = normDepthImage(depths[i, :, :, :], h, w)
            canvas[i, :, :, :] = torch.from_numpy(image)[:3, :, :].float()
        grid = torchvision.utils.make_grid(canvas)
        c, h, w = grid.size()
        scale_factor = 4

        out = grid.numpy()
        image = numpy.array(out)
        c, h, w = image.shape
        image = numpy.array(image.reshape(h, w, c))

        #print("depth shape ", image.shape)
        cv2.imshow(title, image)

def show_images_grid(images, title):     
        print("color tensor shape 1: ", images.shape)   
        # grid = torchvision.utils.make_grid(images)

        out = images.detach()
        c, h, w = out.shape
        image = numpy.array(out.reshape(h, w, c))
        print("color tensor shape 2: ", images.shape) 
        print("e", image[:,:,0])
        print("e", image[:,:,1])
        print("e", image[:,:,2])
        cv2.imshow(title, image[:,:,0])
        cv2.waitKey()
        cv2.imshow(title, image[:,:,1])
        cv2.waitKey()
        cv2.imshow(title, image[:,:,2])
        cv2.waitKey()