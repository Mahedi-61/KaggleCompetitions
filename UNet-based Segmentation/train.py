import torch
from torch import nn 
from tqdm import tqdm
from torch import optim 
from matplotlib import pyplot as plt 
from network import UNet 
import config
from dataset import CarvanaDataset

import numpy as np
import pandas as pd 
 
class TrainNetwork():
    def __init__(self, model, train_loader, val_loader):

        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optim = optim.Adam(self.model.parameters(), 
                                lr = config.LEARNING_RATE)
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                            self.optim, 
                            "min" if config.NUM_CLASSES > 1 else "max", 
                            patience = 2)
        
        self.criterion = nn.BCEWithLogitsLoss()

        if config.MULTI_GPU:
            self.model = nn.DataParallel(self.model, device_ids=config.DEVICE_IDs)

        if config.LOAD_MODEL:
            print("loading previous save model: ", config.SAVE_FILE)
            self.load_model(config.SAVE_FILE, self.model, self.optim)


    def dice_calc(self, pred, gt) :
        pred = torch.sigmoid(pred)
        pred = ((pred) >= .5).float()
        dice_score = (2 * (pred * gt).sum()) / ((pred + gt).sum() + 1e-8)
        return dice_score
    

    def train(self, num_epochs=1):
        train_loop = tqdm(self.train_loader, leave=False, total = self.train_loader.__len__())
        val_loop = tqdm(self.val_loader, leave=False)
        best_loss = 100

        
        for epoch in range(num_epochs):
            # training one epoch
            total_loss = 0
            dice_score = 0
            
            for img, mask in train_loop:
                img, mask = img.to(config.DEVICE), mask.to(config.DEVICE)
            
                self.optim.zero_grad()
                mask_pred = self.model(img)
            
                loss = self.criterion(mask_pred, mask)
                total_loss += loss.item()
                loss.backward()
                self.optim.step()

                dice_score += self.dice_calc(mask_pred, mask)
                train_loop.set_postfix(loss=loss.item())

            print("Epoch %d| Training loss: %f | Training DS %f" % (
                    epoch+1, total_loss, dice_score))

            # validation epoch
            total_loss = 0
            dice_score = 0
            self.model.eval()

            for val_img, val_mask in val_loop:
                val_img, val_mask = val_img.to(config.DEVICE), val_mask.to(config.DEVICE)
                

                with torch.no_grad():
                    val_mask_pred = self.model(val_img)
                    dice_score += self.dice_calc(val_mask_pred, val_mask)

                    loss = self.criterion(val_mask_pred, val_mask)
                    total_loss += loss.item()
                
                if config.SAVE_MODEL:
                    if (best_loss > total_loss):

                        print("saving current best model")
                        best_loss = total_loss
                        self.save_model(self.model, self.optim, epoch + 1)
        
            print("Epoch %d| Validation loss: %f | Validation DS %f" % (
                    epoch+1, total_loss, dice_score))

            self.model.train()
            


    def save_model(self, model, optimizer, epoch):
        checkpoint = {
            "model_state_dict" : model.state_dict(), 
            "optimzer_state_dict" : optimizer.state_dict()
        }

        torch.save(checkpoint, config.MODEL_DIR + "unet_seg" + "_" + str(epoch) + ".pth.tar")


    def load_model(self, filename, model, optimizer):
        checkpoint = torch.load(filename, map_location=config.DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimzer_state_dict"])



def do_train():
    unet = UNet(img_channels=config.IMG_CHANNELS, 
                num_classes=config.NUM_CLASSES).to(config.DEVICE)

    dataset = CarvanaDataset("./data")
    train_loader, val_loader = dataset.get_loader()

    tn = TrainNetwork(unet, train_loader, val_loader)
    #tn.train(num_epochs=15)

    # return the pretrained model for test
    return tn.model



if __name__ == "__main__":
    do_train()