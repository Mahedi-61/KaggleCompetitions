import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt

# torch packages
import torch
from torchvision import transforms
from torch.utils.data import Dataset

# applications packages
import config 

class BlindDetectionDataset(Dataset):
    def __init__(self, X_data, y_data, train):
        self.images = [os.path.join(config.data_dir, 
                        "train_images", img_name + ".png") for img_name in X_data]

        self.labels = torch.as_tensor(y_data) 

        if(train == True):
            self.trans = transforms.Compose(
                [transforms.Resize((256, 256)),
                 transforms.RandomHorizontalFlip(p=0.4),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                      std=[0.229, 0.224, 0.225])
                ])
        else:
            self.trans = transforms.Compose(
                [transforms.Resize((256, 256)),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                      std=[0.229, 0.224, 0.225])
                ])

    def __len__(self):
        return len(self.images) 

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        img = self.trans(img)

        return img, self.labels[index]


if __name__ == "__main__":
    pass
    