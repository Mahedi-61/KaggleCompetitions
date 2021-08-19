import numpy as np
import os
import pandas as pd 
from PIL import Image
import torch 
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt 
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CarvanaDataset(Dataset):
    def __init__(self, root_dir, train_img_list):
        super().__init__()
        self.img_dir = os.path.join(root_dir, "train")
        self.mask_dir = os.path.join(root_dir, "train_masks")
        self.img_list = train_img_list
        self.img_transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        self.mask_transform = A.Compose([
            A.Resize(256, 256),
            ToTensorV2()
        ])
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_abs_path = os.path.join(self.img_dir, self.img_list[idx])
        mask_abs_path = os.path.join(
                self.mask_dir, 
                self.img_list[idx].split(".")[0] + "_mask.gif")
    
        img = np.array(Image.open(img_abs_path))
        mask = np.array(Image.open(mask_abs_path))
        
        img = self.img_transform(image=img)["image"]
        mask = self.mask_transform(image=mask)["image"]
        
        return img, mask


if __name__ == "__main__":
    fig = plt.figure(figsize=(10,10))
    img = np.array(Image.open("/kaggle/working/train/11acc40dc0ea_03.jpg"))
    img_mask = np.array(Image.open("/kaggle/working/train_masks/11acc40dc0ea_03_mask.gif"))

    plt.subplot(1, 2, 1)
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    plt.imshow(img_mask)

    train_img_list = pd.read_csv("/kaggle/working/train_masks.csv")['img']
    dataset = CarvanaDataset("/kaggle/working", train_img_list)

    train_size = int(len(train_img_list) * 0.8)
    val_size = len(train_img_list) - train_size

    train_set, val_set = torch.utils.data.random_split(
                        dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=True)