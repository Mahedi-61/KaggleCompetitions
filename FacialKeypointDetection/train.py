import torch
from torch import device, nn
import numpy as np

from torch import optim 
import config
import utils
from torch.utils.data import DataLoader
from tqdm import tqdm 
from efficientnet_pytorch import EfficientNet
from dataset import KeypointDataset

class main():
    def __init__(self, train_15_dir, train_4_dir, 
                val_15_dir, val_4_dir, test_csv_dir):
        super().__init__()

        self.train_15_loader = DataLoader(
            KeypointDataset(train_15_dir, is_train=True),
            batch_size = config.BATCH_SIZE,
            pin_memory = config.PIN_MEMORY, 
            shuffle = True)

        self.val_15_loader = DataLoader(
            KeypointDataset(val_15_dir, is_train=True),
            batch_size = config.BATCH_SIZE,
            pin_memory = config.PIN_MEMORY, 
            shuffle = False)

        self.train_4_loader = DataLoader(
            KeypointDataset(train_4_dir, is_train=True),
            batch_size = config.BATCH_SIZE,
            pin_memory = config.PIN_MEMORY, 
            shuffle = True)

        self.val_4_loader = DataLoader(
            KeypointDataset(val_4_dir, is_train=True),
            batch_size = config.BATCH_SIZE,
            pin_memory = config.PIN_MEMORY, 
            shuffle = False)

        self.test_loader = DataLoader(
            KeypointDataset(test_csv_dir, is_train=False),
            batch_size = 1,
            pin_memory = config.PIN_MEMORY, 
            shuffle = False)

        # model
        self.model_4 = EfficientNet.from_pretrained("efficientnet-b0")
        self.model_15 = EfficientNet.from_pretrained("efficientnet-b0")
        
        self.model_4._fc = nn.Linear(1280, 30)
        self.model_15._fc = nn.Linear(1280, 30)
        
        self.model_4 = self.model_4.to(config.DEVICE)
        self.model_15 = self.model_15.to(config.DEVICE)

        if config.MULTI_GPU:
            self.model_4 = torch.nn.DataParallel(self.model_4, device_ids=[0, 1])
            self.model_15 = torch.nn.DataParallel(self.model_15, device_ids=[0, 1])

        self.optimizer_4 = torch.optim.Adam(self.model_4.parameters(), 
        lr = config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

        self.loss = nn.MSELoss(reduction="sum")
        self.optimizer_15 = torch.optim.Adam(self.model_15.parameters(), 
            lr = config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)


        if config.LOAD_MODEL:
            print("loading models ...")
            self.load_model(config.FILE_4, self.model_4, self.optimizer_4)
            self.load_model(config.FILE_15, self.model_15, self.optimizer_15)

        # submitting the results 
        utils.get_submission(self.test_loader, self.train_4_loader, self.model_4, self.model_15)



    def train(self):
        loop_4_train = tqdm(self.train_4_loader, leave=False)
        loop_15_train = tqdm(self.train_15_loader, leave=False)

        loop_4_val = tqdm(self.val_4_loader, leave=False)
        loop_15_val = tqdm(self.val_15_loader, leave=False)

        best_error_15 = 100.0
        best_error_4 = 100.0

        # train for 4 keypoints
        for epoch in range(config.NUM_EPOCHS):
            l = 0
            example = 0

            for img, label in loop_4_train:
                img, label = img.to(config.DEVICE), label.to(config.DEVICE)

                # forward pass
                output = self.model_4(img)
                output[label == -1] = -1
                error = self.loss(output, label)

                example += torch.numel(output[label != -1])
                l += error.item()

                # backward pass
                self.optimizer_4.zero_grad()
                error.backward()
                self.optimizer_4.step()

            epoch_loss = (l / example) ** 0.5
            print("Epoch %d | Model_4 Train Loss %f" % (epoch, epoch_loss))


            l = 0
            example = 0
            for img, label in loop_15_train:
                img, label = img.to(config.DEVICE), label.to(config.DEVICE)

                # forward pass
                output = self.model_15(img)
                output[label == -1] = -1
                error = self.loss(output, label)

                example += torch.numel(output[label != -1])
                l += error.item()

                # backward pass
                self.optimizer_15.zero_grad()
                error.backward()
                self.optimizer_15.step()

            epoch_loss = (l / example) ** 0.5
            print("Epoch %d | Model_15 Train Loss %f" % (epoch, epoch_loss))

            # validation for 4 keypoints
            l = 0
            example = 0
            self.model_4.eval()
            for img, label in loop_4_val:
                img, label = img.to(config.DEVICE), label.to(config.DEVICE)

                # forward pass
                output = self.model_4(img)
                output[label == -1] = -1
                error = self.loss(output, label)

                example += torch.numel(output[label != -1])
                l += error.item()

            epoch_loss_4 = (l / example) ** 0.5
            print("Epoch %d | Model_4 Validation Loss %f" % (epoch, epoch_loss_4))

            
            # validation for 15 keypoints
            l = 0
            example = 0
            self.model_15.eval()
            for img, label in loop_15_val:
                img, label = img.to(config.DEVICE), label.to(config.DEVICE)

                # forward pass
                output = self.model_15(img)
                output[label == -1] = -1
                error = self.loss(output, label)

                example += torch.numel(output[label != -1])
                l += error.item()

            epoch_loss_15 = (l / example) ** 0.5
            print("Epoch %d | Model_15 Validation Loss %f" % (epoch, epoch_loss))


            # saving models
            if config.SAVE_MODEL and (epoch > 30):
                if (best_error_4 > epoch_loss_4):
                    print("\nsaving model_4 ....")
                    best_error_4 = epoch_loss_4
                    self.save_model("efficeint_b0_4", self.model_4, self.optimizer_4, epoch)

                if (best_error_15 > epoch_loss_15):
                    print("\nsaving model_15 ....")
                    best_error_15 = epoch_loss_15
                    self.save_model("efficeint_b0_15", self.model_15, self.optimizer_15, epoch)

            self.model_4.train()
            self.model_15.train()


    def save_model(self, model_name, model, optimizer, epoch):
        checkpoint = {
            "model_state_dict" : model.state_dict(), 
            "optimzer_state_dict" : optimizer.state_dict()
        }

        torch.save(checkpoint, config.MODEL_DIR + model_name + "_" + str(epoch) + ".pth.tar")


    def load_model(self, filename, model, optimizer):
        checkpoint = torch.load(filename, map_location=config.DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimzer_state_dict"])



if __name__ == "__main__":
    #utils.data_split("data/training.csv")
    m = main("./data/train_15.csv", 
             "./data/train_4.csv", 
             "./data/val_15.csv", 
             "./data/val_4.csv", 
             "./data/test.csv")

    
    #m.train()