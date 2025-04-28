import shutup  # ignore warning
shutup.please()

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
import torch.optim
import warnings
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import os
os.chdir(os.path.dirname(__file__))

warnings.simplefilter(action='ignore', category=RuntimeWarning)
torch.manual_seed(2022)


def load_data(state="train"):  # load train/val/test dataset 

    if state == "train":
        data = pd.read_csv("../../data/train_example/20240808232043_OtJF37SH_data.txt", sep="\t")  # the path of training dataset
    elif state == "test":
        data = pd.read_csv("../../data/train_example/20240808232043_OtJF37SH_test.txt", sep="\t")  # the path of testing dataset
    else:  
        data = pd.read_csv("../../data/train_example/20240808232043_OtJF37SH_val.txt", sep="\t")  # the path of validation dataset


    # Delete the first column of data
    data = data.iloc[:, 1:]
    # confuse data
    data = shuffle(data, random_state=2024)
    # get feature and label
    feature = data.iloc[:,:-1].values.astype(float) 
    label = data.iloc[:,-1].values.astype(int)

    encoder = LabelEncoder()
    label = encoder.fit_transform(label.ravel())

    feature, label = torch.FloatTensor(feature), torch.LongTensor(label)
    return feature, label


def process(ratio="8:1:1"):

    data = pd.read_csv("../../data/train_example/20240808232043_OtJF37SH_data.txt", sep="\t")

    ratio_str = ratio.split(":")
    ratio_num = list(map(int, ratio_str))  # [8, 1, 1]
    train_ratio = ratio_num[0] / sum(ratio_num)
    test_ratio = ratio_num[2] / sum(ratio_num[1:])

    train_data, res_data = train_test_split(data, test_size=1-train_ratio, random_state=42, stratify=data[["Label"]])
    val_data, test_data = train_test_split(res_data, test_size=test_ratio, random_state=42, stratify=res_data[["Label"]])

    # save
    train_data.to_csv("../../data/train_example/20240808232043_OtJF37SH_data.txt", sep="\t", index=False)
    test_data.to_csv("../../data/train_example/20240808232043_OtJF37SH_test.txt", sep="\t", index=False)
    val_data.to_csv("../../data/train_example/20240808232043_OtJF37SH_val.txt", sep="\t", index=False)

