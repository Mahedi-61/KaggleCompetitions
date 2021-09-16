import torch
import os
import pandas as pd
from PIL import Image
import config 

class VOCDataset(torch.utils.data.Dataset):
	def __init__(self, csv_file, S=7, B=2, C=20):

		super().__init__()
		self.annotations = pd.read_csv(os.path.join(config.data_dir, csv_file))
		self.img_dir = os.path.join(config.data_dir, "images")
		self.lable_dir = os.path.join(config.data_dir, "labels")
		
		self.S = S
		self.B = B
		self.C = C

	def __len__(self):
		return len(self.annotations)

	def __getitem(self, index):
		pass
	
	

voc = VOCDataset("train.csv")
print(voc.img_dir)
print(voc.lable_dir)