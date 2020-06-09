import signal
import sys
from functools import partial

import torchvision
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

import nets as net
import torch.nn as nn
#import eval_cnn as eval
import threeD60_dataset as dataset
import cv2
import utils
import time
import loss as lss
from trainer import Trainer

num_epochs = 20
bs = 8
lr = 0.0001

DATA_PATH = '3d60'
TRAIN_FILE = '3d60/v1/train_files.txt'
TEST_FILE = '3d60/v1/test_files.txt'
LOAD_DIR = 'models'
SAVE_FILE = 'model2'

wrapper = utils.CaptureOnSetAttribute()
wrapper.END = False

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def terminate_handler(signum, frame):
    response = input("End? (yes/no)")
    if response == 'yes':
        print('The model will be saved in next item')
        wrapper.END = True

#signal.signal(signal.SIGINT, terminate_handler)

def main():
    sphere_mode = True
    sphere_loss = True

    if '--no_sphere' in sys.argv:
        sphere_mode = False
    if '--no_sphere_loss' in sys.argv:
        sphere_loss = False

    model_folder = None
    epoch = 0
    if len(sys.argv) > 3 and '--load' in sys.argv:
        model_folder = sys.argv[1]
        epoch = int(sys.argv[2])

    
    print("Convs sphere: ", sphere_mode)
    print("Loss sphere: ", sphere_loss)

    trainer = Trainer(sphere_mode, sphere_loss, model_folder, epoch)
    trainer.train()

if __name__ == '__main__':
    main()
    # eval.eval(model,test_loader)