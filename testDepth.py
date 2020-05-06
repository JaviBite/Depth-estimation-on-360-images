import utils
import cv2
import numpy
import torch

FUNCTIONS = True

file = "3d60/Stanford2D3D/area1/0_area_11_depth_0_Left_Down_0.0.exr"

if FUNCTIONS:
    import threeD60_dataset as d

    # Load image from file
    depth = d.loadDepth(file)
    print("Img Shape: ", depth.shape)
    cv2.imshow("Imagen: ", utils.getDepthImage(depth))
    cv2.waitKey()

    # Transform image for tensor
    tensor = utils.depthToTensor(depth)
    print("Tensor shape: ", tensor.shape)
    cv2.waitKey()

    # Convert tensor to image
    imageDep = utils.tensorToDepth(tensor)
    print("Image shape (from tensor): ",  imageDep.shape)
    cv2.imshow("Tensor: ", utils.getDepthImage(imageDep))
    cv2.waitKey()


else:

    # Load image from file
    depthImg = numpy.array(cv2.imread(file, cv2.IMREAD_ANYDEPTH))
    print("Depth Img Shape: ", depthImg.shape)
    cv2.imshow("Depth: ", utils.getDepthImage(depthImg))
    cv2.waitKey()

    # Transform image for tensor
    h, w = depthImg.shape
    depthImgT = depthImg.reshape(1, h, w)
    print("Depth Img Shape (transformed): ", depthImgT.shape)
    cv2.waitKey()

    ## ----------------------------------
    ## IMAGE NO SHOWABLE, TORCH FORMAT
    ## ----------------------------------

    # Get tensor
    tensorDep = torch.from_numpy(depthImgT).type(torch.float32)
    print("Tensor Shape: ", tensorDep.shape)
    cv2.waitKey()

    # Convert tensor
    out = tensorDep.numpy()
    print("Detached Shape: ", out.shape)
    c, h, w = out.shape
    cv2.waitKey()

    # Transform tensor to image
    out = out.reshape(h, w)
    depth2 = numpy.array(out)
    print("Detached Shape (transformed): ", out.shape)
    cv2.waitKey()

    ## ----------------------------------
    ## IMAGE SHOWABLE, RGB FORMAT
    ## ----------------------------------

    print("TensorImage shape: ", depth2.shape) 
    cv2.imshow("Tensor: ", utils.getDepthImage(depth2))
    cv2.waitKey()