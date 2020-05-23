import utils
import cv2
import nets as m
import torch
import numpy
from normalDepth_dataset import *
from threeD60_dataset import *
from torch.utils.data import DataLoader
import sys

DATA_PATH = '3d60'
#DATA_PATH = 'normalDepthDataset/test/LR'
TEST_FILE = '3d60/v1/test_files.txt'
SAVE_PATH = './models/model2_ep0.pt'
bs = 4

test_dataset = ThreeD60(root_dir=DATA_PATH, txt_file=TEST_FILE)
#test_dataset = NormalDepth(root_dir=DATA_PATH)

test_loader = DataLoader(dataset=test_dataset, batch_size=bs, shuffle=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    # LOADING PRETRAINED MODEL
    device = DEVICE
    print("   Loading pretrained encoder")
    encoder = m.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load('models/model_shpere_1_ep22/encoder.pth', map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = m.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load('models/model_shpere_1_ep22/depth.pth', map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)

    with torch.no_grad():
        for data in test_loader:
            features = encoder(data['image'].to(DEVICE))
            outputs:torch.Tensor = depth_decoder(features)

            for i, out in enumerate(outputs):
                disp = outputs[("disp", 0)]
                color = utils.tensorToImage(data['image'][i,:,:,:].to(DEVICE))
                ground = utils.getDepthImage(utils.tensorToDepth(data['depth'][i,:,:,:].to(DEVICE)), 'normal')
                predicted = utils.getDepthImage(utils.tensorToDepth(disp[i,:,:,:].to(DEVICE)), 'normal')
            
                cv2.imshow("Color, Ground Truth, prediction", numpy.concatenate([color,ground,predicted]))

                key = cv2.waitKey()
                if chr(key) == 'q':
                    print("Exit")
                    sys.exit()


def main2():

    # Load model
    net = m.NetTwo().float().to(DEVICE)

    # color_image = utils.getColorImage("3d60/Stanford2D3D/area1/0_area_11_color_0_Left_Down_0.0.png")
    # h, w, c = color_image.shape   
    # image_tensor = torch.from_numpy(color_image).type('torch.DoubleTensor').reshape(c, h, w) / 255.0

    if len(sys.argv) > 1:
        model_file = sys.argv[1]
        net.load_state_dict(torch.load(model_file, map_location=DEVICE))
        print("Model " + model_file + "loaded")
    else:
        net.load_state_dict(torch.load(SAVE_PATH))

    with torch.no_grad():
        for data in test_loader:
            outputs:torch.Tensor = net(data['image'].to(DEVICE))
            for i, out in enumerate(outputs):

                color = utils.tensorToImage(data['image'][i,:,:,:].to(DEVICE))
                ground = utils.getDepthImage(utils.tensorToDepth(data['depth'][i,:,:,:].to(DEVICE)))
                predicted = utils.getDepthImage(utils.tensorToDepth(outputs[i,:,:,:].to(DEVICE)))
            
                cv2.imshow("Color, Ground Truth, prediction", numpy.concatenate([color,ground,predicted]))

                key = cv2.waitKey()
                if chr(key) == 'q':
                    print("Exit")
                    sys.exit()


if __name__ == '__main__':
    main()
    # eval.eval(model,test_loader)

cv2.waitKey(0)
