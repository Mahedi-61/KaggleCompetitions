import config
from torch.utils.data import DataLoader 
import torch 
from torch import nn 
from tqdm import tqdm 
from model_k import SiameseNetwork_Kaggle, initialize_weights
import numpy as np 
from dataset import SOCOFing, SOCOFingTest
from torch.optim.lr_scheduler import ExponentialLR
import utils


class Train():
    def __init__(self):
        self.train_loader = DataLoader(
            SOCOFing(),
            batch_size=config.batch_size, 
            shuffle=False,
            num_workers= 4  
        )

        self.val_loader = DataLoader(
            SOCOFingTest(val=True),
            batch_size=config.batch_size, 
            shuffle=False,
            num_workers= 4  
        )

        self.test_loader = DataLoader(
            SOCOFingTest(),
            batch_size=config.way, 
            shuffle=False,
            num_workers= 4  
        )

        self.model = SiameseNetwork_Kaggle().to(config.device)
        #self.model.apply(initialize_weights)

        if config.multi_gpus:
            self.model = torch.nn.DataParallel(self.model)
        
        if config.is_load_model:
            pass 

        #self.criterion =  torch.nn.BCEWithLogitsLoss()
        self.criterion_k = torch.nn.CosineEmbeddingLoss(margin=0.1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                    lr = config.learning_rate)

        self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)

        i1, i2, l = next(iter(self.val_loader))


    def test(self, loader):
        correct = 0
        error = 0

        self.model.eval()
        with torch.no_grad():
            for img1, img2, label in loader:
                img1, img2, label = (img1.to(config.device), 
                                    img2.to(config.device), label.to(config.device))

                out1, out2 = self.model(img1, img2)
                cos = nn.CosineSimilarity(dim=1)
                out = cos(out1, out2)
                out = out.cpu().detach().numpy()
                pred = np.argmax(out)
                
                if pred == 0: correct += 1
                else: error += 1

        self.model.train()
        print("Total correct {} | wrong: {} precision: {}".
                format(correct, error, correct*1.0/(correct + error)))

        return correct, error



    def train(self):
        train_loop = tqdm(self.train_loader)
        avg_loss = 0
        best_correction = 0

        for epoch in range(config.num_epochs):
            for batch_id, (img1, img2, label) in enumerate(self.train_loader):
                img1 = img1.to(config.device)
                img2 = img2.to(config.device)
                label = label.to(config.device)

                self.optimizer.zero_grad()
                #out = self.model(img1, img2)
                #loss = self.criterion(out, label)

                out1, out2 = self.model(img1, img2)
                loss = self.criterion_k(out1, out2, label)

                avg_loss += loss.item()
                loss.backward()
                self.optimizer.step()

                if (batch_id % 100 == 0):
                    print("Batch ID: {} | avg loss: {}".format(batch_id, avg_loss))
                    avg_loss = 0

                if (batch_id % 300 == 0):
                    correct, error = self.test(self.val_loader)

                if (correct > best_correction):
                    best_correction = correct
                    if config.is_save_model:
                        utils.save_model(self.model, self.optimizer, batch_id) 
            
            self.scheduler.step()
            print("Epoch  {} | learning rate {}".format(epoch + 1, 
                 self.scheduler.get_lr()))


if __name__ == "__main__":
    t = Train()
    t.train()