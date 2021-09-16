import torch 

data_dir = "./data"
model_dir = "./model"
num_workers = 4
batch_size = 32
num_epochs = 5
learning_rate = 2e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
multi_gpu = True
is_load = True
is_save = True 

model_file = "bd_efficient_e_10.pth"