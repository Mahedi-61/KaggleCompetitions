import torch
from torch import nn 
from tqdm import tqdm
from torch import optim 
from matplotlib.pyplot import plt 
from network import UNet 

class ImageSegment(object):
    def __init__(self, train_loader, val_loader, device):
        super().__init__()
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = 1
        self.img_channels = 3
        self.unet = UNet(self.img_channels, self.num_classes).to(device)
        self.optim = optim.RMSprop(
            self.unet.parameters(), 
            lr=1e-4, momentum=0.9, weight_decay=1e-8)
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optim, 
            "min" if self.num_classes > 1 else "max", 
            patience=2)
        
        self.criterion = nn.BCEWithLogitsLoss()

    def dice_calc(self, gt, pred) :
        pred = torch.sigmoid(pred)
        pred = ((pred) >= .5).float()
        dice_score = (2 * (pred * gt).sum()) / ((pred + gt).sum() + 1e-8)
        return dice_score
    
    def train(self, num_epochs=1):
        loop = tqdm(self.train_loader, leave=False, total=self.train_loader.__len__())
        total_loss = 0
        dice_score = 0
        
        for epoch in range(num_epochs):
            for img, mask in loop:
                img, mask = img.to(self.device), mask.to(self.device)

                self.optim.zero_grad()
                mask_pred = self.unet(img)
                loss = self.criterion(mask_pred, mask.float())
                total_loss += loss.item()
                loss.backward()
                self.optim.step()

                #buji na
                run_DS = self.dice_calc(mask, mask_pred)
                dice_score += run_DS

                loop.set_postfix(loss=loss.item())

            print("Epoch %d| loss: %f | dice score %f" % (epoch+1, total_loss, dice_score))
            
    def test(self):
        with torch.no_grad():
            images ,masks =next(iter(self.val_loader))
            images = images.to(self.device)
            masks  = masks.to(self.device)

            mask_pred = self.unet(images)

            img = mask_pred.cpu().numpy()
            masks = masks.cpu().numpy()
            masks_2 = (masks > 0.5).astype(int)

            fig, axes = plt.subplots(1, 3, figsize=(15, 15))

            axes[0].imshow(masks[0][0])
            axes[0].set_title('Ground Truth Mask')

            axes[1].imshow(img[0][0])
            axes[1].set_title('Prababilistic Mask')

            axes[2].imshow(masks_2[0][0])
            axes[2].set_title('Probabilistic Mask threshold')