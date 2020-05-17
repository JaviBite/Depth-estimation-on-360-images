import utils
import cv2
import nets as m
import torch
import numpy
import threeD60_dataset as dataset
from torch.utils.data import DataLoader
import sys
import loss as c

DATA_PATH = '3d60'
TEST_FILE = '3d60/v1/test_files.txt'
SAVE_PATH = './models/fyn_model_ep4.pt'
bs = 4
PRINT_FREC = 10

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def eval(model, test_loader, total):

    costFunc = c.Cost()
    printIndex = 0
    count = 0

    with torch.no_grad():
        cost = 0.0
        for data in test_loader:
            outputs:torch.Tensor = model(data['image'].to(DEVICE))
            for i, out in enumerate(outputs):
                cost += costFunc(out, data['depth'][i].to(DEVICE))
                count += 1

            printIndex = printIndex + 1
            if printIndex > PRINT_FREC:
                printIndex = 0
                sys.stdout.write('\r')
                # the exact output you're looking for:
                ratio = count/total
                sys.stdout.write("[%-80s] %d%%" % ('='*int(ratio*80), ratio * 100))
                sys.stdout.flush()

        sys.stdout.write('\r')
        sys.stdout.write("[%-80s] %d%%" % ('='*int(80), 100))
        print('Test Accuracy (mean of the all costs) of the model on the ' + str(total) + ' test images: {}'.format((cost / total)))


def main():

    test_dataset = dataset.ThreeD60(root_dir=DATA_PATH, txt_file=TEST_FILE)
    test_loader = DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

    total = len(test_dataset)

    # Load model
    model = m.ConvNet().float().to(DEVICE)
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))

    eval(model ,test_loader, total)              


if __name__ == '__main__':
    main()

