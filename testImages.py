import utils
import cv2
import numpy
import torch


# color_image = utils.getColorImage("3d60/Stanford2D3D/area1/0_area_11_color_0_Left_Down_0.0.png")
# print ("Color image shape: ", color_image.shape)
# cv2.imshow("Color", color_image)

# depth_map = utils.getDepthImage("3d60/Stanford2D3D/area1/0_area_11_depth_0_Left_Down_0.0.exr")
# print ("Depth map shape: ", depth_map.shape)
# cv2.imshow("Depth", depth_map)

# print(depth_map)

# cv2.waitKey(10)


FUNCTIONS = True

file = "3d60/Stanford2D3D/area1/0_area_11_color_0_Left_Down_0.0.png";

if FUNCTIONS:
    import threeD60_dataset as d

    # Load image from file
    image = d.loadImage(file)
    print("Img Shape: ", image.shape)
    cv2.imshow("Imagen: ", image)
    cv2.waitKey()

    # Transform image for tensor
    tensor = utils.imageToTensor(image)
    print("Tensor shape: ", tensor.shape)
    cv2.waitKey()

    # Convert tensor to image
    image = utils.tensorToImage(tensor)
    print("Image shape (from tensor): ",  image.shape)
    cv2.imshow("Tensor: ", image)
    cv2.waitKey()

else:

    # Load image from file
    image = numpy.array(cv2.imread(file))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("Img Shape: ", image.shape)
    cv2.imshow("Imagen: ", image)
    cv2.waitKey()

    # Transform image for tensor
    image = image.transpose(2, 0, 1)
    print("Img Shape (transformed): ", image.shape)
    cv2.waitKey()

    ## ----------------------------------
    ## IMAGE NO SHOWABLE, TORCH FORMAT
    ## ----------------------------------

    # Get tensor
    tensorImg = torch.from_numpy(image).type(torch.float32) / 255.0
    print("Tensor Shape: ", tensorImg.shape)
    cv2.waitKey()

    # Convert tensor
    out = tensorImg.numpy()
    print("Detached Shape: ", out.shape)
    c, h, w = out.shape
    cv2.waitKey()

    # Transform tensor to image
    out = out.transpose(1, 2, 0)
    image = numpy.array(out)
    print("Detached Shape (transformed): ", out.shape)
    cv2.waitKey()

    ## ----------------------------------
    ## IMAGE SHOWABLE, RGB FORMAT
    ## ----------------------------------

    print("TensorImage shape: ", image.shape) 
    cv2.imshow("Tensor: ", image)
    cv2.waitKey()