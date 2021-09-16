import os
import torch 

# directories
root_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(root_dir, "models")

real_dir = os.path.join(root_dir, "data", "SOCOFing", "Real")
alter_easy_dir = os.path.join(root_dir, "data", "SOCOFing", "Altered", "Altered-Easy")
#test_dir = os.path.join(root_dir, "data", "omniglot", "images_evaluation")

#conditions
multi_gpus = True 
is_save_model = True
is_load_model = False 

# model hyperparameters
size = 96 
batch_size = 128
learning_rate = 0.0005
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 600
num_epochs = 20
total_len = 1050000

# training parameters
way = 100
times = 200 


# files
data_real_pkl = "./data_real.pkl"
data_alter_easy_pkl = "./data_alter_easy.pkl"
model_file = os.path.join(model_dir, "saimese_")
loaded_model_file = os.path.join(model_dir, "saimese_")