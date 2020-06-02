import signal
import sys
from functools import partial

import torchvision
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch.optim as optim

import nets as networks
import torch.nn as nn
from threeD60_dataset import *
from normalDepth_dataset import *
import cv2
import utils
import time
import loss as lss
from layers import *
import os
import utils as u


class Trainer:
    def __init__(self, sphere=False, model_folder=None, epoch=0):

        self.savedir = 'models'
        self.datapath = '3d60'
        self.trainfile = '3d60/v1/train_files.txt'
        self.testfile = '3d60/v1/test_files.txt'
        self.valfile = '3d60/v1/val_files.txt'

        self.logfile = 'logfile.txt'
        self.log_frequency = 100

        self.bs = 1
        self.num_epochs = 40
        self.save_frequency = 5
        self.num_layers = 18
        self.weights_init = "pretrained"
        self.scales = range(4)
        self.learning_rate = 1e-4
        self.scheduler_step_size = 15

        self.height = 256   #384
        self.width = 512    #640

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.models["encoder"] = networks.ResnetEncoder(
            self.num_layers, self.weights_init == "pretrained", sphere=sphere)
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.scales, sphere=sphere)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())



        self.model_optimizer = optim.Adam(self.parameters_to_train, self.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.scheduler_step_size, 0.1)

        self.criterion = lss.L2Loss()
        # self.criterion = lss.SphereMSE(self.height, self.width).to(self.device)

        # dataset
        train_dataset = ThreeD60(root_dir=self.datapath, txt_file=self.trainfile)
        val_dataset = ThreeD60(root_dir=self.datapath, txt_file=self.valfile)
        #train_dataset = NormalDepth(root_dir='normalDepthDataset/train/LR')
        

        # train_size = int(0.8 * len(train_dataset))
        # val_size = len(train_dataset) - train_size
        # train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.bs, shuffle=True)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size=self.bs, shuffle=False)

        self.val_iter = iter(self.val_loader)

        self.num_steps = len(self.train_loader)

        self.epoch = None
        if model_folder is not None:
            print("Loading model: ", model_folder)
            print("Model at epoch", epoch)
            self.load_model(model_folder)
            self.epoch = epoch + 1


        # print("Training model named:\n  ", self.opt.model_name)
        print("Training images: ", str(len(train_dataset)))
        print("Validation images: ", str(len(val_dataset)))
        print("Models and tensorboard events files are saved to:\n  ", self.savedir)
        print("Training is using:\n  ", self.device)


    def train(self):
        """Run the entire training pipeline
        """
        if self.epoch is None:
            self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.epoch, self.num_epochs, 1):
            self.run_epoch()
            if (self.epoch + 1) % self.save_frequency == 0 or \
            		self.epoch == self.num_epochs or \
            		self.epoch < 1:
                print("Saving model...")
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                
            self.step += 1
            
        self.log("train", inputs, outputs, losses)
        self.val()

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.bs / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  u.sec_to_hm_str(time_sofar), u.sec_to_hm_str(training_time_left)))

    def log(self, logType, inputs, outputs, loss):
        f = open(self.logfile, "a")
        f.write(logType + " " + str(loss["loss"].item()) + " ")
        if logType == "val":
        	f.write("\n")
        f.close()
    
    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)


            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()
    
    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        losses = {}
        for i in range(self.bs):
            inputs['image'] = inputs['image'].to(self.device)

        features = self.models["encoder"](inputs['image'])
        outputs = self.models["depth"](features)

        #for scale in self.scales:
        # disp = outputs[("disp", 0)]
        # print("outputs shape = ", disp[0,:,:,:].shape)
        # img = u.tensorToDepth(disp[0,:,:,:])
        # print("outputs shape = ", img.shape)
        # cv2.imshow("Test", u.getDepthImage(img))
        # cv2.waitKey()
        losses["loss"] = self.criterion(outputs[("disp", 0)], inputs['depth'].to(self.device)) 
        #losses = self.compute_losses(inputs, outputs)
        # losses["loss"] = lss.GradLoss([("disp", scale)], inputs['depth']) 

        return outputs, losses
    
    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join("models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.height
                to_save['width'] = self.width
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self, folder):
        """Load model(s) from disk
        """

        assert os.path.isdir(folder), \
            "Cannot find folder {}".format(folder)
        print("loading model from folder {}".format(folder))

        for n in ['encoder','depth']:
            print("Loading {} weights...".format(n))
            path = os.path.join(folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path, map_location=self.device)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path, map_location=self.device)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")