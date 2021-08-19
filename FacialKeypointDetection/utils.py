import pandas as pd
import numpy as np
import torch
import random 
import config 
from tqdm import tqdm 

def data_split(training_data_dir):
    # code for spliting data into train and test direcotry
    
    face_data = pd.read_csv(training_data_dir)
    division = np.arange(len(face_data))
    random.shuffle(division)
    train = face_data.iloc[division[:5990]]
    val = face_data.iloc[division[5990:]]

    del face_data
    list_15 =  []
    list_4 = []
    for i in range(0, len(train)):
        if np.sum(pd.isnull(train.iloc[i, :30])) < 22:
            list_15.append(i)
        else:
            list_4.append(i)

    train_15 = train.iloc[list_15]
    train_4 = train.iloc[list_4]

    list_15 =  []
    list_4 = []
    for i in range(0, len(val)):
        if np.sum(pd.isnull(val.iloc[i, :30])) < 22:
            list_15.append(i)
        else:
            list_4.append(i)

    val_15 = val.iloc[list_15]
    val_4 = val.iloc[list_4]

    train_15.to_csv("./data/train_15.csv", index=False)
    train_4.to_csv("./data/train_4.csv", index=False)
    val_15.to_csv("./data/val_15.csv", index=False)
    val_4.to_csv("./data/val_4.csv", index=False)



def get_submission(test_loader, train_loader, model_4, model_15):
    model_4.eval()
    model_15.eval()
    preds = []
    ids = pd.read_csv("./data/IdLookupTable.csv")
    image_id = 1

  
    for image, label in tqdm(test_loader):
        image = image.to(config.DEVICE)

        rows = ids.loc[ids["ImageId"] == image_id]
        f_names = list(rows["FeatureName"])
        categories = train_loader.dataset.face_data.columns.values[:-1].tolist()

        if(len(f_names) > 10):
            pred_15 = model_15(image).squeeze(0)
            pred_15 = np.clip(pred_15.detach().cpu().numpy(), 0.0, 96.0)

            for f in f_names:
                f_index = categories.index(f)
                preds.append(pred_15[f_index])

        else:
            pred_4 = model_4(image).squeeze(0)
            pred_4 = np.clip(pred_4.detach().cpu().numpy(), 0.0, 96.0)

            for f in f_names:
                f_index = categories.index(f)
                preds.append(pred_4[f_index])

        image_id += 1

    print("submitting result ...")
    result_df = pd.DataFrame({"RowId" : np.arange(1, len(preds) + 1), 
                              "Location": preds})

    result_df.to_csv(config.SUBMISSION_FILE, index=False)
    model_4.train()
    model_15.train()