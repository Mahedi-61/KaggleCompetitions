import os, cv2
import pickle, random  
import numpy as np 
import torch
from matplotlib import pyplot as plt 
from torch.utils.data import Dataset 
from torchvision import transforms 
from PIL import Image 
import config 

class SOCOFing(Dataset):
    def __init__(self):
        super().__init__()
        self.img_dict_real = self.get_img_files(config.load_data_pkl(config.real))
        
        self.img_dict_hard_cr = self.get_img_files(config.load_data_pkl(config.hard_cr))
        self.img_dict_hard_obl = self.get_img_files(config.load_data_pkl(config.hard_obl))
        self.img_dict_hard_zcut = self.get_img_files(config.load_data_pkl(config.hard_zcut))

        self.train_trans = transforms.Compose([
            transforms.Resize((config.size, config.size)), 
            transforms.RandomAffine(15),
            transforms.ColorJitter(brightness=0.2),
            transforms.RandomRotation(degrees=(0, 45), fill=(255, )),
            transforms.ToTensor()
        ])


    def get_img_files(self, file_name_pkl):
        with open(file_name_pkl, "rb") as f:
            img_dict = pickle.load(f)
        return img_dict 

    
    def __len__(self):
        return config.total_len


    def __getitem__(self, index):

        min_id, max_id = (0, 4000)
        real_dict = {k : self.img_dict_real[k] for k in range(min_id, max_id + 1)}

        alter_cr_dict = {k : self.img_dict_hard_cr[k] 
                        for k in range(min_id, max_id + 1) 
                        if self.img_dict_hard_cr.get(k) != None}
        
        alter_obl_dict = {k : self.img_dict_hard_obl[k] 
                        for k in range(min_id, max_id + 1) 
                        if self.img_dict_hard_obl.get(k) != None}

        alter_zcut_dict = {k : self.img_dict_hard_zcut[k] 
                        for k in range(min_id, max_id + 1) 
                        if self.img_dict_hard_zcut.get(k) != None}

        # genuine pair
        if index % 2 == 0:
            label = 1.0
            class_indx_real = random.randint(min_id, max_id)
            alter_type = random.choice([0, 1, 2])

            if alter_type == 0: 
                while alter_cr_dict.get(class_indx_real) == None:
                    class_indx_real = random.randint(min_id, max_id)

                img2 = self.train_trans(alter_cr_dict[class_indx_real])

            elif alter_type == 1:
                while alter_obl_dict.get(class_indx_real) == None:
                    class_indx_real = random.randint(min_id, max_id)

                img2 = self.train_trans(alter_obl_dict[class_indx_real])

            elif alter_type == 2:
                while alter_zcut_dict.get(class_indx_real) == None:
                    class_indx_real = random.randint(min_id, max_id)

                img2 = self.train_trans(alter_zcut_dict[class_indx_real])
                

        # imposter pair
        else: 
            label = 0.0
            class_indx_real = random.randint(min_id, max_id)
            class_indx_alter = random.randint(min_id, max_id)

            alter_type = random.choice([0, 1, 2])

            if alter_type == 0: 
                while (alter_cr_dict.get(class_indx_alter) == None or 
                            class_indx_real == class_indx_alter):
                    class_indx_alter = random.randint(min_id, max_id)

                img2 = self.train_trans(alter_cr_dict[class_indx_alter])

            elif alter_type == 1:
                while (alter_obl_dict.get(class_indx_alter) == None or 
                            class_indx_real == class_indx_alter):
                    class_indx_alter = random.randint(min_id, max_id)

                img2 = self.train_trans(alter_obl_dict[class_indx_alter])

            elif alter_type == 2:
                while (alter_zcut_dict.get(class_indx_alter) == None or 
                            class_indx_real == class_indx_alter):
                    class_indx_alter = random.randint(min_id, max_id)

                img2 = self.train_trans(alter_zcut_dict[class_indx_alter])


        img1 = self.train_trans(real_dict[class_indx_real])
        return img1, img2, torch.from_numpy(np.array(label, dtype=np.float32))



class SOCOFingTest(Dataset):
    def __init__(self, val=False):
        super().__init__()
        self.img_dict_real = self.get_img_files(config.load_data_pkl(config.real))
        self.img_dict_hard_cr = self.get_img_files(config.load_data_pkl(config.hard_cr))
        self.img_dict_hard_obl = self.get_img_files(config.load_data_pkl(config.hard_obl))
        self.img_dict_hard_zcut = self.get_img_files(config.load_data_pkl(config.hard_zcut))

        self.test_trans = transforms.Compose([
            transforms.Resize((config.size, config.size)), 
            transforms.ToTensor()
        ])

        self.times = config.times
        self.way = config.way
        self.val = val 

    def __len__(self):
        return self.way * self.times 

    def get_img_files(self, file_name_pkl):
        with open(file_name_pkl, "rb") as f:
            img_dict = pickle.load(f)
        return img_dict 


    def __getitem__(self, index):
        if self.val:
            min_id, max_id = (4000, 4999)
        else:
            min_id, max_id = (5000, 5999)

        real_dict = {k : self.img_dict_real[k] for k in range(min_id, max_id + 1)}

        alter_cr_dict = {k : self.img_dict_hard_cr[k] 
                        for k in range(min_id, max_id + 1) 
                        if self.img_dict_hard_cr.get(k) != None}
        
        alter_obl_dict = {k : self.img_dict_hard_obl[k] 
                        for k in range(min_id, max_id + 1) 
                        if self.img_dict_hard_obl.get(k) != None}

        alter_zcut_dict = {k : self.img_dict_hard_zcut[k] 
                        for k in range(min_id, max_id + 1) 
                        if self.img_dict_hard_zcut.get(k) != None}


        # genuine pair
        if index % 3 == 0:
            label = 1.0
            class_indx_real = random.randint(min_id, max_id)
            alter_type = random.choice([0, 1, 2])

            if alter_type == 0: 
                while alter_cr_dict.get(class_indx_real) == None:
                    class_indx_real = random.randint(min_id, max_id)

                img2 = self.test_trans(alter_cr_dict[class_indx_real])

            elif alter_type == 1:
                while alter_obl_dict.get(class_indx_real) == None:
                    class_indx_real = random.randint(min_id, max_id)

                img2 = self.test_trans(alter_obl_dict[class_indx_real])

            elif alter_type == 2:
                while alter_zcut_dict.get(class_indx_real) == None:
                    class_indx_real = random.randint(min_id, max_id)

                img2 = self.test_trans(alter_zcut_dict[class_indx_real])


        # imposter pair
        else: 
            label = 0.0
            class_indx_real = random.randint(min_id, max_id)
            class_indx_alter = random.randint(min_id, max_id)

            alter_type = random.choice([0, 1, 2])

            if alter_type == 0: 
                while (alter_cr_dict.get(class_indx_alter) == None or 
                            class_indx_real == class_indx_alter):
                    class_indx_alter = random.randint(min_id, max_id)

                img2 = self.test_trans(alter_cr_dict[class_indx_alter])

            elif alter_type == 1:
                while (alter_obl_dict.get(class_indx_alter) == None or 
                            class_indx_real == class_indx_alter):
                    class_indx_alter = random.randint(min_id, max_id)

                img2 = self.test_trans(alter_obl_dict[class_indx_alter])

            elif alter_type == 2:
                while (alter_zcut_dict.get(class_indx_alter) == None or 
                            class_indx_real == class_indx_alter):
                    class_indx_alter = random.randint(min_id, max_id)

                img2 = self.test_trans(alter_zcut_dict[class_indx_alter])

        img1 = self.test_trans(real_dict[class_indx_real])
        return img1, img2, torch.from_numpy(np.array(label, dtype=np.float32))


if __name__ == "__main__":
    d = SOCOFing(train=False)
    i1, i2, l = d.__getitem__(0)
    print(i1.shape)
    print(i2.shape)
    print(l.shape)