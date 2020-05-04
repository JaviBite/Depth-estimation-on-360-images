import utils
import cv2

color_image = utils.getColorImage("3d60/Stanford2D3D/area1/0_area_11_color_0_Left_Down_0.0.png")
print ("Color image shape: ", color_image.shape)
cv2.imshow("Color", color_image)

depth_map = utils.getDepthImage("3d60/Stanford2D3D/area1/0_area_11_depth_0_Left_Down_0.0.exr")
print ("Depth map shape: ", depth_map.shape)
cv2.imshow("Depth", depth_map)

cv2.waitKey(0)
