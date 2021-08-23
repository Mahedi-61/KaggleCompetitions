import os
import pandas as pd
import numpy as np
from PIL import Image 
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch import nn 
from network import UNet 
import config 
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
import cv2
from train import do_train


def run_length_encode(mask):
    pixels = mask.flatten()
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle


def biggest_contour(im):
    contours, hierarchy = cv2.findContours(np.copy(im), 
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest, maxarea = None, 0

    for cnt in contours:
        a = cv2.contourArea(cnt)
        if a > maxarea:
            biggest, maxarea = cnt, a
    return biggest


class CarvanaTestDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.sample_submission_csv = os.path.join(root_dir, "sample_submission.csv")
        self.root_dir = root_dir

        test_df = pd.read_csv(self.sample_submission_csv)['img']
        self.test_img_list = test_df.values
        self.rels = []

        self.img_transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=[0.0, 0.0, 0.0],
                        std=[1.0, 1.0, 1.0],  
                        max_pixel_value=255.0),
            ToTensorV2()
        ])

    
    def __len__(self):
        return len(self.test_img_list)


    def test(self):
        self.model = do_train()
        test_loader = DataLoader(CarvanaTestDataset(self.root_dir), 
                                batch_size=256, num_workers=4, shuffle=False)

        self.model.eval()

        for batch_id, test_img in enumerate(test_loader): 
            print("batch_id : ", batch_id)
            test_img = test_img.to(config.DEVICE)
            
            with torch.no_grad():
                preds = self.model(test_img)
                preds = torch.sigmoid(preds).cpu().numpy()
                preds = np.squeeze(preds, axis = 1)

                for idx, pred in enumerate(preds):
                    pred = (pred > 0.5).astype(np.float)
                    pred = (pred * 255).astype(np.uint8)
                    pred = cv2.resize(pred, (config.ORIG_WIDTH, config.ORIG_HEIGHT))
 
                    cv2.imwrite(os.path.join(config.RESULT_DIR, self.test_img_list[batch_id*256 + idx]), pred)


    def load_results(self):
        all_seg_imgs = [os.path.join(config.RESULT_DIR, f) for f in self.test_img_list]
        for se_img in all_seg_imgs:
            img = cv2.imread(se_img, cv2.IMREAD_GRAYSCALE)
            img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1] #making black-white
            self.rels.append(run_length_encode(img))

    def create_submission(self):
        print("create submission ...")
        df = pd.DataFrame({'img': self.test_img_list, 'rle_mask': self.rels})
        df.to_csv(config.SUBMISSION_FILE, compression="gzip", index=False)

       
    def __getitem__(self, idx):
        img_abs_path = os.path.join(self.root_dir, "test", self.test_img_list[idx])
        img = np.array(Image.open(img_abs_path).convert("RGB"))
        img = self.img_transform(image=img)["image"]
        
        return img



if __name__ == "__main__":
    d = CarvanaTestDataset("./data")
    d.load_results()
    #d.test()
    d.create_submission()