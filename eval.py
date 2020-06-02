import utils
import cv2
import nets as m
import torch
import numpy as np
from normalDepth_dataset import *
from threeD60_dataset import *
from torch.utils.data import DataLoader
import loss as lss
import sys

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

DATA_PATH = '3d60'
#DATA_PATH = 'normalDepthDataset/test/LR'
TEST_FILE = '3d60/v1/test_files.txt'
SAVE_PATH = './models/model2_ep0.pt'
bs = 1
PRINT_FREC = 10
MIN_DEPTH = 1e-3

DEPTH_SCALE = 7.5

test_dataset = ThreeD60(root_dir=DATA_PATH, txt_file=TEST_FILE)
#test_dataset = NormalDepth(root_dir=DATA_PATH)

total = len(test_dataset)

test_loader = DataLoader(dataset=test_dataset, batch_size=bs, shuffle=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

spherical_loss = lss.SphereMSE(test_dataset.height, test_dataset.width).to(DEVICE)
l2_loss = lss.L2Loss()

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    # gt = gt * DEPTH_SCALE 
    # pred = pred * DEPTH_SCALE 

    gt[gt <= 0] = MIN_DEPTH
    pred[pred <= 0] = MIN_DEPTH

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse_ = (gt - pred) ** 2
    rmse = np.sqrt(rmse_.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(rmse_ / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def main():

    SPHERE_CONVS = True
    no_image = True

    if len(sys.argv) > 1:
        model_file = sys.argv[1]
        print("Model " + model_file + " loaded")
    else:
        print("Usage: " + sys.argv[0] + " <model folder> [--no_sphere]")
        exit()

    if '--no_sphere' in sys.argv:
        SPHERE_CONVS = False

    print("Test images: " + str(total))


    # LOADING PRETRAINED MODEL
    device = DEVICE
    print("   Loading pretrained encoder")
    encoder = m.ResnetEncoder(18, True, sphere=SPHERE_CONVS)
    loaded_dict_enc = torch.load(model_file + '/encoder.pth', map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = m.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4), sphere=SPHERE_CONVS)

    loaded_dict = torch.load(model_file + '/depth.pth', map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)

    total_sph_losses = 0
    total_l2_losses = 0

    errors = []

    printIndex = 0
    count = 0
    worst_error = 0
    worst_file = 'none'

    with torch.no_grad():
        for data in test_loader:
            input_data = data['image'].to(DEVICE)
            features = encoder(input_data)
            outputs:torch.Tensor = depth_decoder(features)
            disp = outputs[("disp", 0)]

            # this_sphloss = spherical_loss(outputs[("disp", 0)], input_data) 
            # this_l2loss = l2_loss(outputs[("disp", 0)], input_data) 
            
            for i in range(disp.size(0)):
                
                i_gt = data['depth'][i,:,:,:].to(DEVICE)
                i_pred = disp[i,:,:,:].to(DEVICE)
                ground = utils.tensorToDepth(i_gt)
                predicted = utils.tensorToDepth(i_pred)

                metrics = compute_errors(ground, predicted) + (spherical_loss(i_pred, i_gt),)
                #metrics = compute_errors(ground, predicted)
                if metrics[2] > worst_error:
                    worst_error = metrics[2]
                    worst_file = data['name']

                errors.append(metrics)
            
            count += bs
            printIndex += bs
            if printIndex > PRINT_FREC:
                printIndex = 0
                sys.stdout.write('\r')
                # the exact output you're looking for:
                ratio = count/total
                sys.stdout.write("[%-80s] %d%%" % ('='*int(ratio*80), ratio * 100))
                sys.stdout.flush()


    mean_errors = np.array(errors).mean(0)
    print("\n  " + ("{:>8} | " * 8).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "srmse"))
    print(("&{: 8.3f}  " * 8).format(*mean_errors.tolist()) + "\\\\")
    print("Worst RSME: " + str(worst_error) + " file: " + str(worst_file))
    print("\n-> Done!")


if __name__ == '__main__':
    main()

cv2.waitKey(0)
