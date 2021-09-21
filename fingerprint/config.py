import os
import torch 

# directories
root_dir = os.path.dirname(os.path.abspath(__file__))
wight_dir = os.path.join(root_dir, "weights")
data_dir = os.path.join(root_dir, "data")

real_dir = os.path.join(data_dir, "SOCOFing", "Real")
alter_easy_dir = os.path.join(data_dir, "SOCOFing", "Altered", "Altered-Easy")
alter_medium_dir = os.path.join(data_dir, "SOCOFing", "Altered", "Altered-Medium")
alter_hard_dir = os.path.join(data_dir, "SOCOFing", "Altered", "Altered-Hard")

#conditions
is_train = False 
multi_gpus = True 
is_save_model = is_train 
is_load_model = True             

# model hyperparameters
size = 96 
batch_size = 128
learning_rate = 0.0004
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 600
num_epochs = 20
total_len = 1050000

# training parameters
way = 400
times = 5 

def load_data_pkl(data_type):
    return os.path.join(data_dir, data_type + ".pkl")
    
# dataset type
real = "real"
easy_cr = "easy_cr"
easy_obl = "easy_obl"
easy_zcut = "easy_zcut"
medium_cr = "medium_cr"
medium_obl = "medium_obl"
medium_zcut = "medium_zcut"
hard_cr = "hard_cr"
hard_obl = "hard_obl"
hard_zcut = "hard_zcut"

model_file = os.path.join(wight_dir, "saimese_con_h_")
loaded_model_file = os.path.join(wight_dir, "saimese_hcr_15_4200.pth")