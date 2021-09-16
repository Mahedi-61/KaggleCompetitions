import numpy as np
import os 
import config 
from PIL import Image 
import pickle, random  
import torch 

hand_en_dict = {"Left" : 0, "Right" : 1}
finger_en_dict = {"thumb" : 0, "index" : 1, "middle" : 2, 
                  "ring" : 3, "little" : 4}



def save_img_files(file_name_pkl, data_path):
    print("saving " + file_name_pkl)
    indx_dict = {}

    for img_file in os.listdir(data_path):
        lst = img_file.split("_") 
        index = ((int(lst[0]) - 1) * 10 + 
            hand_en_dict[lst[3]] * 5 + finger_en_dict[lst[4]]) 

        img_path = os.path.join(data_path, img_file)
        indx_dict[index] = Image.open(img_path).convert("L") 
    
    saved_file = open(file_name_pkl, "wb")
    pickle.dump(indx_dict, saved_file)
    saved_file.close()
    print("Complete !!")



def save_model(model, optimizer, batch_id):
    print("saving best model")
    checkpoint = {}
    checkpoint["model_state_dict"] = model.state_dict()
    checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, config.model_file + str(batch_id) + ".pth")


def load_model(model, optimizer):
    print("loaded model file")
    checkpoint = torch.load(config.loaded_model_file)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


if __name__ == "__main__":
    save_img_files(config.data_real_pkl, config.real_dir)
    save_img_files(config.data_alter_easy_pkl, config.alter_easy_dir)