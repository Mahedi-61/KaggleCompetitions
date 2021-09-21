from numpy.core.fromnumeric import repeat
import config
from torch.utils.data import DataLoader 
import torch 
from torch import nn 
from tqdm import tqdm 
from models.resnet_model import SiameseResentNetwork, initialize_weights
import numpy as np 
from dataset import SOCOFing, SOCOFingTest
from torch.optim.lr_scheduler import ExponentialLR
from sklearn import metrics
import utils
from contrastive_loss import ContrastiveLoss


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

        self.model = SiameseResentNetwork().to(config.device)
        #self.model.apply(initialize_weights)

        if config.multi_gpus:
            self.model = torch.nn.DataParallel(self.model)
        

        self.criterion =  torch.nn.BCEWithLogitsLoss()
        #self.criterion_k = torch.nn.CosineEmbeddingLoss(margin=1)
        #self.criterion = ContrastiveLoss()


        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                    lr = config.learning_rate,
                                    weight_decay = 5e-3)

        self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)

        if config.is_load_model:
            utils.load_model(self.model, self.optimizer) 


    def test(self, loader):
        correct = 0
        error = 0
        y_true = []
        y_pred = [] 

        self.model.eval()
        with torch.no_grad():

            for img1, img2, label in loader:
                img1, img2, label = (img1.to(config.device), 
                                    img2.to(config.device), label.to(config.device))

                # making three channel 
                img1 = torch.repeat_interleave(img1, repeats=3, dim=1)
                img2 = torch.repeat_interleave(img2, repeats=3, dim=1)

                out = self.model(img1, img2)

                label = label.cpu().detach().tolist()
                #out = torch.norm(out1 - out2, dim=1)
                out = out.cpu().detach().tolist()

                y_true += label 
                y_pred += out
    
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

            fprs, tprs, thresholds = metrics.roc_curve(y_true, y_pred)
            eer = fprs[np.nanargmin(np.absolute((1 - tprs) - fprs))]
            auc = metrics.auc(fprs, tprs)

        print("AUC {} | EER {}".format(auc, eer))
        self.model.train()
        return  auc, eer 


    def train(self):
        train_loop = tqdm(self.train_loader)
        avg_loss = 0
        best_auc = 0

        for epoch in range(config.num_epochs):
            for batch_id, (img1, img2, label) in enumerate(self.train_loader):
                img1 = img1.to(config.device)
                img2 = img2.to(config.device)
                label = label.to(config.device)

                 # making three channel 
                img1 = torch.repeat_interleave(img1, repeats=3, dim=1)
                img2 = torch.repeat_interleave(img2, repeats=3, dim=1)

                self.optimizer.zero_grad()
                out = self.model(img1, img2)
                loss = self.criterion(out, label)

                #out1, out2 = self.model(img1, img2)
                #loss = self.criterion(out1, out2, label)

                avg_loss += loss.item()
                loss.backward()
                self.optimizer.step()

                if (batch_id % 100 == 0):
                    print("Batch ID: {} | avg loss: {}".format(batch_id, avg_loss))
                    avg_loss = 0
                    
                if (batch_id % 300 == 0):
                    auc, eer  = self.test(self.val_loader)

                    if (auc >= best_auc):
                        best_auc = auc

                        if config.is_save_model:
                            utils.save_model(self.model, self.optimizer, epoch+1, batch_id) 
            
            self.scheduler.step()
            print("Epoch  {} | learning rate {}".format(epoch + 1, 
                 self.scheduler.get_lr()))


if __name__ == "__main__":
    t = Train()
    if config.is_train == True:
        t.train()

    else: 
        print("testing ...")
        t.test(t.test_loader)