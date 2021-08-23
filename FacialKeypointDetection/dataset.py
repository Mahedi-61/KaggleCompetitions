from albumentations.augmentations.transforms import Normalize
from albumentations.core.composition import KeypointParams
import numpy as np
import pandas as pd
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import config


class KeypointDataset(Dataset):
    def __init__(self, csv_dir, is_train=False):
        super().__init__()
        
        self.face_data = pd.read_csv(csv_dir)
        self.is_train = is_train

        self.train_transforms = A.Compose([
            A.Resize(96, 96),
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.8),
            #A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.2),
            A.IAAAffine(shear=15, mode="constant", p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, rotate_limit=0, 
                    border_mode=cv2.BORDER_CONSTANT, p=0.2), 

            A.Normalize(mean=[0.4897, 0.4897, 0.4897], 
                        std=[0.2330, 0.2330, 0.2330]), 

            ToTensorV2()
        ], keypoint_params = A.KeypointParams(format="xy", remove_invisible=False))

        self.val_transforms = A.Compose([
            A.Resize(96, 96),
            A.Normalize(mean=[0.4897, 0.4897, 0.4897], 
                std=[0.2330, 0.2330, 0.2330]), 

            ToTensorV2()
        ], keypoint_params = A.KeypointParams(format="xy", remove_invisible=False))


    def __len__(self):
        return self.face_data.shape[0]

    def __getitem__(self, index):
        if self.is_train:
            img = np.array(self.face_data.iloc[index, 30].split()).astype(np.float32)
            labels = np.array(list(self.face_data.iloc[index, :30]))
            labels[np.isnan(labels)] = -1

            img = np.repeat(img.reshape(96, 96, 1), repeats=3, axis=2)
            ignore_indices = labels == -1
            labels = labels.reshape(15, 2)

            aug = self.train_transforms(image=img, keypoints=labels)
        
        else:
            img = np.array(self.face_data.iloc[index, 1].split()).astype(np.float32)
            img = np.repeat(img.reshape(96, 96, 1), repeats=3, axis=2)
            labels = np.zeros(30)
            ignore_indices = labels == -1
            labels = labels.reshape(15, 2)

            aug = self.val_transforms(image=img, keypoints=labels)

        image = aug["image"]
        labels = aug["keypoints"]

        labels = np.array(labels).reshape(-1)
        labels[ignore_indices] = -1
        return image, labels.astype(np.float32)



if __name__ == "__main__":
    """
    data = KeypointDataset("./data/train_4.csv", is_train=True)
    loader = DataLoader(data, batch_size=32, shuffle=True)
    next(iter(loader))

    img, labels = data.__getitem__(9)

    img = np.transpose(img, [1, 2, 0])
    plt.imshow(img[:, :, 0].detach().cpu().numpy(), cmap="gray")
    plt.plot(labels[0::2], labels[1::2], "go")
    plt.show()

    for index in range(data.__len__()):
        img, labels = data.__getitem__(index)
        plt.imshow(img[:, :, 0], cmap="gray")
        plt.savefig("save_images/" + str(index) + ".png")
    """