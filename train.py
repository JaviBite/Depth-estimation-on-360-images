# Hyperparameters
import torchvision
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

import cnn_net as net
import torch.nn as nn
#import eval_cnn as eval
import threeD60_dataset as dataset
import cv2
import utils
import time

num_epochs = 5
num_classes = 10
bs = 4
lr = 0.001

DATA_PATH = '3d60'
TRAIN_FILE = '3d60/v1/train_files.txt'
TEST_FILE = '3d60/v1/test_files.txt'
LOAD_DIR = 'models'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()
    # L1 norm
    def forward(self, grad_fake, grad_real):
        return torch.mean( torch.abs(grad_real-grad_fake) )

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


    model = net.I2D().to(DEVICE).double()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=4e-5)

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
            print(sample['image'].shape)
            outputs = model(sample['image'])
            loss = criterion(outputs, sample['depth'])
            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = sample['depth'].size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (utils.accurate(predicted,sample['depth'])).sum().item()
            acc_list.append(correct / total)

            # Verbose
            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                    .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                            (correct / total) * 100))

        torch.save(model.state_dict(),'{}/fyn_model_ep' + str(epoch) + '.pt'.format(LOAD_DIR))
        end = time.time()
        print('model saved')
        print('time elapsed: %fs' % (end - start))

if __name__ == '__main__':
    main()
    # eval.eval(model,test_loader)