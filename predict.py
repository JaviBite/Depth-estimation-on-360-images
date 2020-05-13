import utils
import cv2
import nets as m
import torch
import numpy
import threeD60_dataset as dataset
from torch.utils.data import DataLoader
import sys

DATA_PATH = '3d60'
TEST_FILE = '3d60/v1/test_files.txt'
SAVE_PATH = './models/model2_ep0.pt'
bs = 4

test_dataset = dataset.ThreeD60(root_dir=DATA_PATH, txt_file=TEST_FILE)

test_loader = DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

def main():

    # Load model
    net = m.NetTwo().float()

    if len(sys.argv) > 1:
        model_file = sys.argv[1]
        net.load_state_dict(torch.load(model_file))
        print("Model " + model_file + "loaded")
    else:
        net.load_state_dict(torch.load(SAVE_PATH))

    # color_image = utils.getColorImage("3d60/Stanford2D3D/area1/0_area_11_color_0_Left_Down_0.0.png")
    # h, w, c = color_image.shape   
    # image_tensor = torch.from_numpy(color_image).type('torch.DoubleTensor').reshape(c, h, w) / 255.0

    with torch.no_grad():
        for data in test_loader:
            outputs:torch.Tensor = net(data['image'])
            for i, out in enumerate(outputs):

                color = utils.tensorToImage(data['image'][i,:,:,:])
                ground = utils.getDepthImage(utils.tensorToDepth(data['depth'][i,:,:,:]))
                predicted = utils.getDepthImage(utils.tensorToDepth(outputs[i,:,:,:]))
            
                cv2.imshow("Color, Ground Truth, prediction", numpy.concatenate([color,ground,predicted]))

                key = cv2.waitKey()
                if chr(key) == 'q':
                    print("Exit")
                    sys.exit()


if __name__ == '__main__':
    main()
    # eval.eval(model,test_loader)

cv2.waitKey(0)
