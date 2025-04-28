import shutup  # 控制台输出 忽略 warning
shutup.please()

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
import torch.optim
import warnings
import pandas as pd
import torch




warnings.simplefilter(action='ignore', category=RuntimeWarning)
torch.manual_seed(2022)


def load_data(state="train"):  # train, val, test
    if state == "train":
        data = pd.read_csv("../../data/train_example/20240808232043_OtJF37SH_train.txt", sep="\t")
    elif state == "test":
        data = pd.read_csv("../../data/train_example/20240808232043_OtJF37SH_test.txt", sep="\t")
    else:  
        data = pd.read_csv("../../data/train_example/20240808232043_OtJF37SH_val.txt", sep="\t")

    # Get the name of each row of data
    name = data.iloc[:, 0].values.astype(str) 
    # get data
    feature = data.iloc[:, 1:-1].values.astype(float) 
    label = data.iloc[:,-1].values


    encoder = LabelEncoder()
    label = encoder.fit_transform(label.ravel())

    feature, label, name = torch.FloatTensor(feature), torch.LongTensor(label), name
    return feature, label, name
