import utils
import cv2

# cv2.imshow("Color", utils.getColorImage("3d60/Stanford2D3D/area1/0_area_11_color_0_Left_Down_0.0.png"))
# cv2.imshow("Depth", utils.getDepthImage("3d60/Stanford2D3D/area1/0_area_11_depth_0_Left_Down_0.0.exr"))
#cv2.imshow("Depth", utils.getDepthImage("3d60/0000000000.bin"))
utils.showDepthImage(utils.getDepthImage("3d60/Stanford2D3D/area1/0_area_11_depth_0_Left_Down_0.0.exr"))

cv2.waitKey(0)
