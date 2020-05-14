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

num_epochs = 20
bs = 4
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
    print('The model will be saved in next item')
    wrapper.END = True

signal.signal(signal.SIGINT, terminate_handler)

def main():
    # transforms to apply to the data
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # dataset
    train_dataset = dataset.ThreeD60(root_dir=DATA_PATH, txt_file=TRAIN_FILE)
    test_dataset = dataset.ThreeD60(root_dir=DATA_PATH, txt_file=TEST_FILE)

    train_loader = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

    # Show images from dataset

    '''
    for i in range(len(test_dataset)):
        sample = test_dataset[i]

        # print(i, sample['image'].shape, sample['landmarks'].shape)

        cv2.imshow("color", sample['image'])
        print(sample['image'].shape)
        utils.showDepthImage(sample['depth'])
        cv2.waitKey()

        if i == 3:
            break
    '''


    model = net.NetTwo().to(DEVICE).float()

    if len(sys.argv) > 1:
        model_file = sys.argv[1]
        model.load_state_dict(torch.load(model_file))
        print("Model " + model_file + " loaded")

    # Loss and optimizer
    criterion = lss.GradLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) #, betas=(0.9, 0.999), eps=1e-08, weight_decay=4e-5)

    # Train the model
    total_step = len(train_loader)
    loss_list = []
    acc_list = []

    # criterion = GradLoss()

    start = time.time()

    print("Starting...")

    for epoch in range(num_epochs):
        for i, sample in enumerate(train_loader):
            
            # Run the forward pass
            # print(sample['image'].shape)
            # print(sample['depth'].shape)
            outputs = model(sample['image'])
            
            # print("outs")
            # print(outputs.shape)
            # print(sample['depth'].shape)
            loss = criterion(outputs, sample['depth'])
            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            # total = sample['depth'].size(0)
            # _, predicted = torch.max(outputs.data, 1)
            # correct = (utils.accurate(predicted,sample['depth'])).sum().item()
            # acc_list.append(correct / total)

            # Verbose
            if (i + 1) % 100 == 0:
                # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                #     .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                #             (correct / total) * 100))
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

            if wrapper.END:
                break

        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
        torch.save(model.state_dict(), LOAD_DIR + '/' + SAVE_FILE +'_ep' + str(epoch) + '.pt')
        print('model saved')
        end = time.time()
        print('time elapsed: %fs' % (end - start))

        if wrapper.END:
                break

if __name__ == '__main__':
    main()
    # eval.eval(model,test_loader)