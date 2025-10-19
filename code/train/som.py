import shutup  # 控制台输出 忽略 warning
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
# os.chdir(os.path.dirname(__file__))
import numpy as np
import pylab as pl
import math
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
from sklearn.model_selection import  StratifiedKFold
import itertools


import argparse
import pickle


from DataProcess import load_data,process


def draw(C):
    colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
    for i in range(len(C)):
        coo_X = []  # X-coordinate list
        coo_Y = []  # Y-coordinate list
        for j in range(len(C[i])):
            coo_X.append(C[i][j][0])
            coo_Y.append(C[i][j][1])
        pl.scatter(coo_X, coo_Y, marker='x', color=colValue[i % len(colValue)], label=i)

    pl.legend(loc='upper right')
    pl.show()




# classification function
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



def show(som, output_size, result_path):

    plt.figure(figsize=(16, 16))
    # Define pattern markings, colours for different labels
    all_markers = ['o', 's', 'D', 'v', 'P', '*', 'X']
    all_colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    all_category_color = {'0': 'C0', '1': 'C1', '2': 'C2', '3': 'C3', '4': 'C4', '5': 'C5', '6': 'C6'}
    all_class_names = ['0', '1', '2', '3', '4', '5', '6']
    markers = all_markers[0:output_size] # 'V', 'P', '*', 'X'
    colors = all_colors[0:output_size]
    category_color = dict(itertools.islice(all_category_color.items(), output_size))
    class_names = all_class_names[0:output_size]

    # U-Matrix on background
    heatmap = som.distance_map()
    # draw a background picture
    plt.pcolor(heatmap, cmap='bone_r')

    for cnt, xx in enumerate(X_train):
        w = som.winner(xx)
        # Draw a mark where the sample Heat is located
        plt.plot(w[0] + .5, w[1] + .5, markers[Y_train[cnt]], markerfacecolor='None',
                 markeredgecolor=colors[Y_train[cnt]], markersize=12, markeredgewidth=2)

    plt.axis([0, size, 0, size])
    ax = plt.gca()
    # Reverse y-axis direction
    ax.invert_yaxis()
    legend_elements = [Patch(facecolor=clr, edgecolor='w', label=l) for l, clr in category_color.items()]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, .95))
    plt.savefig(result_path + "figure_1.png", format='png', dpi=300)
    plt.close()


    plt.figure(figsize=(16, 16))
    the_grid = GridSpec(size, size)
    plt.title('classify figure')
    for position in winmap.keys():
        label_fracs = [winmap[position][label] for label in range(output_size)]  # [0, 1, 2]
        plt.subplot(the_grid[position[1], position[0]], aspect=1)
        patches, texts = plt.pie(label_fracs)
        plt.text(position[0] / 100, position[1] / 100, str(len(list(winmap[position].elements()))),
                 color='black', fontdict={'weight': 'bold', 'size': 15}, va='center', ha='center')
        
    plt.legend(patches, class_names, loc='center right', bbox_to_anchor=(-1, 9), ncol=2) 
    plt.savefig(result_path + "figure_2.png", format='png', dpi=300)

    plt.close()
    plt.show()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--kfold", default=0, type=int, help='a number represents kfold, 0 means not use kfold') 
    parser.add_argument("--ratio", default="0", type=str, help='a dataset is split into train, validation and test datasets by ratio, and the seperator is :, e.g. 8:1:1, 0 means not use split')  # "0" 表示不使用split，值只能以:分割，e.g. 8:1:1 新加
    parser.add_argument("--epochs", default=50, type=int, help='number of epochs')
    parser.add_argument("--es", default=10, type=int, help='early stopping patience')
    parser.add_argument("--lr", default=0.01, type=float, help='learning rate')
    parser.add_argument("--bs", default=32, type=int, help="batch size")
    parser.add_argument("--loss_function", default="crossentropy", type=str, help='Options: crossentropy, nllloss, focalloss') 
    parser.add_argument("--optimizer_function", default="adam", type=str, help='Options: adam, rmsprop, sgd') 
    parser.add_argument("--output_size", default=2, type=int, help='label number. e.g. 2 for binary classification, 4 for 4-class classification. The range is [2, 7]') 

    args = parser.parse_args()
    kfold = args.kfold
    ratio = args.ratio
    epochs = args.epochs
    es = args.es
    lr = args.lr
    batch_size = args.bs
    loss_function = args.loss_function
    optimizer_function = args.optimizer_function
    output_size = args.output_size

    print('model = SOM')
    print('kfold =', kfold)
    print('ratio =', ratio)
    print('epochs =', epochs)
    print('earlystopping =', es)
    print('learning rate =', lr)
    print('batch size =', batch_size)
    print('loss function =', loss_function)
    print('optimizer function =', optimizer_function)
    print('label number =', output_size)

    # need split a dataset into train, validation and test datasets by ratio
    if (ratio != "0"):
        process(ratio=ratio)
    
    '''begin training'''
    iterations = epochs
    if (kfold):
        kfscore = []
        skf = StratifiedKFold(n_splits=kfold) 
        X_train_total, Y_train_total = load_data("train")

        for i, (train_idx, val_idx) in enumerate(skf.split(X_train_total, Y_train_total)):
            print("--------The {} fold is training---------".format(i+1))
            X_train, X_val = np.array(X_train_total)[train_idx], np.array(X_train_total)[val_idx]
            Y_train, Y_val = np.array(Y_train_total)[train_idx], np.array(Y_train_total)[val_idx]
            # sample size
            N = X_train.shape[0]
            # Number of dimensions/features
            M = X_train.shape[1]
            # Empirical formula: determines output layer size
            size = math.ceil(np.sqrt(5 * np.sqrt(N)))
            print("The best side length of the grid is :", size)

            # define the model
            som = MiniSom(size, size, M, sigma=3, learning_rate=lr, neighborhood_function='bubble')
            # Initialising weights
            som.pca_weights_init(X_train)
            # model training
            som.train_batch(X_train, iterations, verbose=False)
            # Labelling trained som networks using label information
            winmap = som.labels_map(X_train, Y_train)
            
            print("Finish training!") 

            print("--------The {} fold validation result---------".format(i+1))
            y_pred = classify(som, X_val, winmap)
            acc = accuracy_score(Y_val, y_pred)
            precision, recall, f1 = precision_recall_fscore_support(Y_val, y_pred, average='weighted')[:-1]
            print("validation acc = {}, precision = {}, recall = {}, f1 = {}".format(round(acc,4), round(precision,4), round(recall,4), round(f1,4)))
            kfscore.append([acc, precision, recall, f1])
        
        # average score
        kfscore = np.array(kfscore).sum(axis= 0)/float(kfold)  # acc, precision, recall, f1
        print("--------KFold Final Average Validation Results---------")
        print("Stratified KFold mean validation acc = {}, precision = {}, recall = {}, f1 = {}".format(round(kfscore[0], 4), round(kfscore[1], 4), round(kfscore[2], 4), round(kfscore[3], 4)))

    else:
        X_train, Y_train = load_data("train")
        X_train, Y_train = np.array(X_train), np.array(Y_train)
        X_val, Y_val = load_data("validate")
        X_val, Y_val = np.array(X_val), np.array(Y_val)
        # sample size
        N = X_train.shape[0]
        # Number of dimensions/features
        M = X_train.shape[1]
        size = math.ceil(np.sqrt(5 * np.sqrt(N)))
        print("The best side length of the grid is :", size)


        # Initialising the model
        som = MiniSom(size, size, M, sigma=3, learning_rate=lr, neighborhood_function='bubble')
        som.pca_weights_init(X_train)
        som.train_batch(X_train, iterations, verbose=False)
        winmap = som.labels_map(X_train, Y_train)
        
        print("Finish training!") 

        y_pred = classify(som, X_val, winmap)

        acc = accuracy_score(Y_val, y_pred)
        precision, recall, f1 = precision_recall_fscore_support(Y_val, y_pred, average='weighted')[:-1]
        print("validation acc = {}, precision = {}, recall = {}, f1 = {}".format(round(acc,4), round(precision,4), round(recall,4), round(f1,4)))


    print("Done!") 

    

    result_path = './result/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        
    savename = result_path+'model.pth'
    savename_2 = result_path+'model_winmap.pkl'

    # save model and winmap
    with open(savename_2, 'wb') as outfile:
        pickle.dump(winmap, outfile)
    with open(savename, 'wb') as outfile:
        pickle.dump(som, outfile)

    # visualize
    show(som, output_size, result_path)
    
    # load model and winmap
    # with open(savename, 'rb') as infile:
    #     som = pickle.load(infile)
    # with open(savename_2, 'rb') as infile:
    #     winmap = pickle.load(infile)

    # test model
    X_test, Y_test = load_data("test")
    y_pred = classify(som, X_test, winmap)

    acc = accuracy_score(Y_test,y_pred)
    precision, recall, f1 = precision_recall_fscore_support(Y_test, y_pred, average='weighted')[:-1]
    print("test acc = {}, precision = {}, recall = {}, f1 = {}".format(round(acc,4), round(precision,4), round(recall,4), round(f1,4)))


    # save test result
    with open(result_path + "test_result.txt", mode="w") as f:
        f.write("test result: \n")
        f.write("acc = " + str(round(acc, 4)) + "\n")
        f.write("precision = " + str(round(precision, 4)) + "\n")
        f.write("recall = " + str(round(recall, 4)) + "\n")
        f.write("f1 = " + str(round(f1, 4)) + "\n")

    print('test set result')
    print(classification_report(Y_test, np.array(y_pred)))

    
    

