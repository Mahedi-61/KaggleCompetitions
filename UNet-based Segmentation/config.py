import torch
import os

# training 
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-8
BATCH_SIZE = 64
NUM_EPOCHS = 100
MOMENTUM = 0.9
PIN_MEMORY = True

ORIG_WIDTH = 1918
ORIG_HEIGHT = 1280

# network 
IMG_CHANNELS = 3
NUM_CLASSES = 1

# saving & Loading modesl
SAVE_MODEL = True 
LOAD_MODEL = True
MODEL_DIR = "./models/"
SUBMISSION_FILE = "./data/result_submission.csv.gz"
SAVE_FILE = os.path.join(MODEL_DIR + "unet_seg_14.pth.tar")
RESULT_DIR = "./results"

# multi gpus
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MULTI_GPU = True 
DEVICE_IDs = [0, 1]