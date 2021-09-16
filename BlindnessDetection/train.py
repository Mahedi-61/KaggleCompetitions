import pandas as pd 
import os
import numpy as np
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader 
from torch import nn 
import torch 
from tqdm import tqdm 

# application packages
import config 
import utils 
from dataset import BlindDetectionDataset
from efficientnet_pytorch import EfficientNet

class DoTrain:
    def __init__(self):
        train_csv = pd.read_csv(os.path.join(config.data_dir, "train.csv"))
        train_images, train_labels = train_csv["id_code"].tolist(), train_csv["diagnosis"].tolist()

        X_train, X_valid, y_train, y_valid = train_test_split(
                            train_images, train_labels, test_size=0.50)


        self.train_loader = DataLoader(BlindDetectionDataset(X_train, y_train, train=True ),
                            batch_size=config.batch_size, 
                            num_workers=config.num_workers, 
                            shuffle=True)

        self.val_loader = DataLoader(BlindDetectionDataset(X_train, y_train, train=False),
                    batch_size=config.batch_size, 
                    num_workers=config.num_workers, 
                    shuffle=True)

        self.model = EfficientNet.from_pretrained("efficientnet-b0")
        self.model._fc = nn.Linear(in_features=1280, out_features=5)
        self.criterion = nn.CrossEntropyLoss()
        self.adam = torch.optim.Adam(self.model.parameters(), lr= config.learning_rate)


        self.model = self.model.to(config.device)
        if config.multi_gpu:
            self.model = nn.DataParallel(self.model , device_ids=[0, 1])

        if config.is_load:
            filename = os.path.join(config.model_dir, config.model_file)
            utils.load_model(filename, self.model, self.adam)


    def train(self):

        self.train_loop = tqdm(self.train_loader, leave=True)
        self.val_loop = tqdm(self.val_loader, leave=True)

        for e in range(config.num_epochs):
            total_loss = 0
            correct = 0

            for img, label in self.train_loop:

                #forward
                img = img.to(config.device)
                label = label.to(config.device)
                
                out = self.model(img)

                # backward
                loss = self.criterion(out, label)
                loss.backward()
                self.adam.step()
                self.model.zero_grad()
            
                # metric
                total_loss += loss.item()
                correct += sum(torch.argmax(out, dim=1)[i] == label[i] 
                                    for i in range(len(label))).item()
                 
            total_len = config.batch_size * len(self.train_loader)
            print("Epochs %d | training loss %.3f | accuracy %.3f" % 
                        (e+1, total_loss/total_len, correct/total_len))
                

            # validation
            total_loss = 0
            correct = 0
            best_loss = -1
            self.model.eval()
            for img, label in self.val_loop:

                #forward
                img = img.to(config.device)
                label = label.to(config.device)
                
                with torch.no_grad():
                    out = self.model(img)

                    # backward
                    loss = self.criterion(out, label)
                
                    # metric
                    total_loss += loss.item()
                    correct += sum(torch.argmax(out, dim=1)[i] == label[i] 
                                        for i in range(len(label))).item()
                    
            total_len = config.batch_size * len(self.val_loader)
            print("Epochs %d | val loss %.3f | accuracy %.3f" % 
                        (e+1, total_loss/total_len, correct/total_len))
                
            if(e+1 > 4 and best_loss < total_loss):
                total_loss = best_loss
                filename = os.path.join(config.model_dir, "bd_efficient_e_" + str(e+1))
                utils.save_model(filename, self.model, self.adam)

            self.model.train()


dt = DoTrain()
dt.train()