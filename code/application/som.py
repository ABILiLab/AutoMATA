import shutup 
shutup.please()

import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.chdir(os.path.dirname(__file__))
import numpy as np
import pylab as pl

import numpy as np
from utils.minisom import MiniSom
from sklearn import datasets
from numpy import sum as npsum
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec

import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,matthews_corrcoef,confusion_matrix,roc_auc_score,classification_report,multilabel_confusion_matrix,hamming_loss
from sklearn.metrics import precision_recall_fscore_support

import argparse
import pickle

from DataProcess import load_data


def classify(som,data,winmap):
    default_class = npsum(list(winmap.values())).most_common()[0][0]
    result = []
    for d in data:
        win_position = som.winner(d)
        if win_position in winmap:
            result.append(winmap[win_position].most_common()[0][0])
        else:
            result.append(default_class)
    return result


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="SOM", type=str)
    parser.add_argument("--model_path", default="./model.pth", type=str)
    parser.add_argument("--winmap_path", default="./winmap.pkl", type=str)

    args = parser.parse_args()
    savename = args.model_path
    savename_2 = args.winmap_path


    print('model_type = SOM')
    # savename = './model.pth'
    # savename_2 = './winmap.pkl'
    
    # load model
    with open(savename, 'rb') as infile:
        som = pickle.load(infile)
    with open(savename_2, 'rb') as infile:
        winmap = pickle.load(infile)
        


    # load data
    X_test, Y_test, name = load_data("test")
    
    # predict
    y_pred = classify(som, X_test, winmap)

    acc = accuracy_score(Y_test, y_pred)
    precision, recall, f1 = precision_recall_fscore_support(Y_test, y_pred, average='macro')[:-1]
    print("test acc = {}, precision = {}, recall = {}, f1 = {}".format(acc, precision, recall, f1))

    '''save the testing metrics results of using this model'''
    with open("./result/test_metrics_result.txt", mode="w") as f:
        f.write("test result: \n")
        f.write("acc = " + str(acc) + "\n")
        f.write("precision = " + str(precision) + "\n")
        f.write("recall = " + str(recall) + "\n")
        f.write("f1 = " + str(f1) + "\n")


    '''save the probability results of using this model'''
    with open("./result/test_result.txt", mode="w") as f:
        f.write("name" + "\t" + "probability" + "\n")
        for i in range(len(name)):
            f.write(name[i] + "\t" + str(y_pred[i]) + "\n")


    

