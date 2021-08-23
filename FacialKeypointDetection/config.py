from sre_constants import CATEGORY
import torch

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 64
NUM_EPOCHS = 100
PIN_MEMORY = True
SAVE_MODEL = False
LOAD_MODEL = True
MULTI_GPU = True 
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_DIR = "./models/"
FILE_4 = MODEL_DIR + "efficeint_b0_4_91.pth.tar"
FILE_15 = MODEL_DIR + "efficeint_b0_15_59.pth.tar"
SUBMISSION_FILE = "./data/result_submission.csv"