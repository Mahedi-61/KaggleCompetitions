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
    def __init__(self, root_dir):
        super().__init__()
        self.train_img_dir = os.path.join(root_dir, "train")
        self.train_mask_dir = os.path.join(root_dir, "train_masks")
        self.train_mask_csv_dir = os.path.join(root_dir, "train_masks.csv")

        self.img_list =  pd.read_csv(self.train_mask_csv_dir)['img']
        self.root_dir = root_dir 

        self.img_transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=[0.0, 0.0, 0.0],
                        std=[1.0, 1.0, 1.0],  
                        max_pixel_value=255.0),
            ToTensorV2()
        ])
        self.mask_transform = A.Compose([
            A.Resize(256, 256),
            ToTensorV2()
        ])
    
    def __len__(self):
        return len(self.img_list)

    def get_loader(self):
        train_size = int(len(self.img_list) * 0.8)
        val_size = len(self.img_list) - train_size

        dataset = CarvanaDataset(self.root_dir)
        train_set, val_set = torch.utils.data.random_split(
                            dataset, [train_size, val_size])

        train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=8, shuffle=True)

        return train_loader, val_loader

    
    def __getitem__(self, idx):
        img_abs_path = os.path.join(self.train_img_dir, self.img_list[idx])
        mask_abs_path = os.path.join(
                self.train_mask_dir, 
                self.img_list[idx].split(".")[0] + "_mask.gif")
    
        img = np.array(Image.open(img_abs_path).convert("RGB"))
        mask = np.array(Image.open(mask_abs_path).convert("L"),  dtype=np.float32)
        mask[mask == 255.0] = 1.0
        mask = np.expand_dims(mask, 2)
        
        img = self.img_transform(image=img)["image"]
        mask = self.mask_transform(image=mask)["image"]
        
        return img, mask


if __name__ == "__main__":
    
    #d = CarvanaDataset("./data")
    #img, mask = d.__getitem__(0)
    #t, v = d.get_loader()
    #img, mask = next(iter(t))

    """
    plt.subplot(1, 2, 1)
    plt.imshow(img[2].permute(1, 2, 0).numpy())

    plt.subplot(1, 2, 2)
    plt.imshow(mask[2].permute(1, 2, 0).numpy())
    plt.show()
    """