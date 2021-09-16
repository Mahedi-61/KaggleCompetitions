import pandas as pd 
import os
import numpy as np

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms 
from torch import nn 
import torch 
from tqdm import tqdm 
from PIL import Image 
from efficientnet_pytorch import EfficientNet

# application packages
import config 
import utils 


class TestDataset(Dataset):
    def __init__(self, X_data):

        self.images = [os.path.join(config.data_dir, 
                        "test_images", img_name + ".png") for img_name in X_data]

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

        return img


class DoTest:
    def __init__(self):
        self.test_csv = pd.read_csv(os.path.join(config.data_dir, "sample_submission.csv"))
        test_images = self.test_csv["id_code"].tolist()

        self.test_loader = DataLoader(TestDataset(test_images),
                    batch_size=config.batch_size, 
                    num_workers=config.num_workers, 
                    shuffle=False)

        self.model = EfficientNet.from_pretrained("efficientnet-b0")
        self.model._fc = nn.Linear(in_features=1280, out_features=5)

        self.model = self.model.to(config.device)
        self.adam = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.result = []

        if config.multi_gpu:
            self.model = nn.DataParallel(self.model , device_ids=[0, 1])

        filename = os.path.join(config.model_dir, config.model_file)
        utils.load_model(filename, self.model, self.adam)



    def test(self):
        self.test_loop = tqdm(self.test_loader, leave=True)
        self.model.eval()
        
        with torch.no_grad():
            for img in self.test_loop:
                #forward
                img = img.to(config.device)
                out = self.model(img)
                self.result += torch.argmax(out, dim=1).cpu().detach().tolist()



    def create_submission(self):
        self.test_csv["diagnosis"] = self.result
        sub_file = os.path.join(config.data_dir, "submission.csv")

        self.test_csv.to_csv(sub_file, index=False)

dt = DoTest()
dt.test()
dt.create_submission()