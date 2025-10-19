import shutup 
shutup.please()
import warnings
import numpy as np
import torch.optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from torch import nn, optim
from utils.FocalLoss import FocalLoss
from sklearn.model_selection import  StratifiedKFold
import os
# os.chdir(os.path.dirname(__file__))
import torch
from sklearn.metrics import accuracy_score
from utils.plot_utils import plotfig
from utils.earlystopping import EarlyStopping
import argparse
import pickle


warnings.simplefilter(action='ignore', category=RuntimeWarning)




from DataProcess import load_data,process

# 一审
def set_random_seed(seed):
    """设置随机种子确保可重现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.Generator().manual_seed(seed)  # dataloader的随机种子
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(dataloader, model):
    model.train()
    num_batches = len(dataloader)
    train_loss= 0  # Accumulation of training losses
    true_label_list, pred_label_list= [], []

    for data in dataloader:
        X_data, Y_data = data[0].to(device), data[1].to(device)  # Pass the training dataset and labels into the cpu or GPU
        output = model(X_data) 
        loss = model.criterion(output, Y_data)
        train_loss += loss.item()
        true_label_list.append(Y_data.cpu().detach().numpy())
        pred_label_list.append(output.argmax(dim=1).cpu().detach().numpy())
        model.optimier.zero_grad() 
        loss.backward() 
        model.optimier.step() 

    y_true = np.concatenate(true_label_list)
    y_pred = np.concatenate(pred_label_list)
    train_loss /= num_batches 
    train_acc = accuracy_score(y_true,y_pred)
    train_precision, train_recall, train_f1 = precision_recall_fscore_support(y_true, y_pred, average='weighted')[:-1]

    return train_loss, train_acc, train_precision, train_recall, train_f1


def validate(dataloader, model):
    num_batches = len(dataloader)
    model.eval()
    val_loss = 0 
    true_label_list, pred_label_list= [], []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += model.criterion(pred, y).item()
            true_label_list.append(y.cpu().detach().numpy())
            pred_label_list.append(pred.argmax(dim=1).cpu().detach().numpy())

    y_true = np.concatenate(true_label_list)
    y_pred = np.concatenate(pred_label_list)
    val_loss /= num_batches 
    val_acc = accuracy_score(y_true,y_pred)
    val_precision, val_recall, val_f1 = precision_recall_fscore_support(y_true, y_pred, average='weighted')[:-1]

    return val_loss, val_acc, val_precision, val_recall, val_f1

def test(dataloader, model, device):
    model.eval()
    true_label_list, pred_label_list= [], []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            
            true_label_list.append(y.cpu().detach().numpy())
            pred_label_list.append(pred.argmax(dim=1).cpu().detach().numpy())

    y_true = np.concatenate(true_label_list)
    y_pred = np.concatenate(pred_label_list)
    acc = accuracy_score(y_true,y_pred)
    precision, recall, f1 = precision_recall_fscore_support(y_true, y_pred, average='weighted')[:-1]
    return acc, precision, recall, f1

class Cnn(nn.Module):
    def __init__(self, conv_size_1, conv_size_2, output_size, dropout_rate, lr, loss_function, optimizer_function):
        super(Cnn, self).__init__()
        self.output_size = output_size 

        self.model1 = nn.Sequential(
            nn.Conv1d(1, conv_size_1, 2, stride=2, padding=6),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.model2 = nn.Sequential(
            nn.Conv1d(conv_size_1, conv_size_2, 2, padding=6),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Flatten(),
            nn.Dropout(dropout_rate)
        )

        
        
        
        self.learning_rate = lr
        self.fc = nn.LazyLinear(self.output_size)  # 一审
        self.loss_function = loss_function  # 一审
        if loss_function == "crossentropy":
            self.criterion = nn.CrossEntropyLoss()
        elif loss_function == "focalloss":
            if output_size == 2:
                self.criterion = FocalLoss(gamma=2, alpha=0.25, task_type='binary')
            else:
                self.criterion = FocalLoss(gamma=2, alpha=0.25, task_type='multi-class', num_classes=output_size)
        elif loss_function == "nllloss":
            self.criterion = nn.NLLLoss()

        if optimizer_function == "adam":
            self.optimier = optim.Adam(self.parameters(), lr=self.learning_rate)
        elif optimizer_function == "sgd":
            self.optimier = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0)
        elif optimizer_function == "rmsprop":
            self.optimier = optim.RMSprop(self.parameters(), lr=self.learning_rate)

    def forward(self, input):
        input = input.reshape(-1,1,input.shape[1])
        x = self.model1(input)
        x = self.model2(x)
        # 一审
        logits = self.fc(x)
        if self.loss_function == "nllloss":
            return nn.functional.log_softmax(logits, dim=1)
        return logits

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--kfold", default=0, type=int, help='a number represents kfold, 0 means not use kfold') 
    parser.add_argument("--ratio", default="0", type=str, help='a dataset is split into train, validation and test datasets by ratio, and the seperator is :, e.g. 8:1:1, 0 means not use split')  # "0" 表示不使用split，值只能以:分割，e.g. 8:1:1 新加
    parser.add_argument("--epochs", default=10, type=int, help='number of epochs')
    parser.add_argument("--es", default=10, type=int, help='early stopping patience')
    parser.add_argument("--lr", default=0.01, type=float, help='learning rate')
    parser.add_argument("--bs", default=32, type=int, help="batch size")
    parser.add_argument("--loss_function", default="crossentropy", type=str, help='Options: crossentropy, nllloss, focalloss') 
    parser.add_argument("--optimizer_function", default="adam", type=str, help='Options: adam, rmsprop, sgd') 
    parser.add_argument("--output_size", default=2, type=int, help='label number. e.g. 2 for binary classification, 4 for 4-class classification. The range is [2, 7]') 
    parser.add_argument('--random_seed', type=int, default=42, help='random seed')  # 一审新增

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
    random_seed = args.random_seed  # 一审新增

    print('model = CNN')
    print('kfold =', kfold)
    print('ratio =', ratio)
    print('epochs =', epochs)
    print('earlystopping =', es)
    print('learning rate =', lr)
    print('batch size =', batch_size)
    print('loss function =', loss_function)
    print('optimizer function =', optimizer_function)
    print('label number =', output_size)
    print('random seed =', random_seed)  # 一审新增

    # Convolution kernel size, user can change it.
    conv_size_1 = 64
    conv_size_2 = 32
    dropout_rate = 0.0  # the ratio of dropout


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seed(random_seed)  # 一审新增
    result_path = './result/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    savename = result_path + 'model.pth'
    early_stopping = EarlyStopping(es, verbose=True, savename=savename, delta=0.0001)

    # need split a dataset into train, validation and test datasets by ratio
    if (ratio != "0"):
        process(ratio=ratio)
    

    # load training datasets
    X_train, Y_train = load_data("train")
    input_size = X_train.shape[1]



    '''begin training'''
    if (kfold):
        kfscore = []
        skf = StratifiedKFold(n_splits=kfold)
        for i, (train_idx, val_idx) in enumerate(skf.split(X_train, Y_train)):
            early_stopping = EarlyStopping(es, verbose=True, savename=savename, delta=0.0001)
            print("--------The {} fold is training---------".format(i+1))
            trainset, valset = torch.FloatTensor(np.array(X_train)[train_idx]), torch.FloatTensor(np.array(X_train)[val_idx])
            traintag, valtag = torch.LongTensor(np.array(Y_train)[train_idx]), torch.LongTensor(np.array(Y_train)[val_idx])
            train_dataset = torch.utils.data.TensorDataset(trainset, traintag)
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            val_dataset =  torch.utils.data.TensorDataset(valset, valtag)
            val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

            # define model
            cnn = Cnn(conv_size_1, conv_size_2, output_size, dropout_rate, lr, loss_function, optimizer_function).to(device)
            val_acc_s = []  # Store the validation accuracy of each epoch
            val_loss_s = []  # Store the validation loss of each epoch
            train_acc_s = []  # Store the training accuracy of each epoch
            train_loss_s = []  # Store the training loss of each epoch
            for t in range(epochs):
                print("--------Begin the {} epoch training---------".format(t+1))
                train_loss, train_acc, train_precision, train_recall, train_f1 = train(dataloader=train_loader, model=cnn)
                val_loss, val_acc, val_precision, val_recall, val_f1 = validate(dataloader=val_loader, model=cnn)
                
                print("train loss = {}, acc = {}, precision = {}, recall = {}, f1 = {} ".format(round(train_loss, 4), round(train_acc, 4), round(train_precision, 4), round(train_recall, 4), round(train_f1, 4)))
                print("validation loss = {}, acc = {}, precision = {}, recall = {}, f1 = {}".format(round(val_loss, 4), round(val_acc, 4), round(val_precision, 4), round(val_recall, 4), round(val_f1, 4)))

                train_acc_s.append(train_acc)
                train_loss_s.append(train_loss)
                val_acc_s.append(val_acc)
                val_loss_s.append(val_loss)

                early_stopping(val_loss, cnn)
                if early_stopping.early_stop:  # Early stop conditions met
                    print('early stopping')
                    epochs = t+1 
                    break

            print("--------The {} fold validation result---------".format(i+1))
            val_acc, val_precision, val_recall, val_f1 = test(dataloader=val_loader, model=cnn, device=device)
            print("validation acc = {}, precision = {}, recall = {}, f1 = {}".format(round(val_acc, 4), round(val_precision, 4), round(val_recall, 4), round(val_f1, 4)))
            kfscore.append(test(dataloader=val_loader, model=cnn, device=device))
        
        # average score
        kfscore = np.array(kfscore).sum(axis= 0)/float(kfold)  # acc, precision, recall, f1
        print("--------KFold Final Average Validation Results---------")
        print("Stratified KFold mean validation acc = {}, precision = {}, recall = {}, f1 = {}".format(round(kfscore[0], 4), round(kfscore[1], 4), round(kfscore[2], 4), round(kfscore[3], 4)))



    else:
        cnn = Cnn(conv_size_1, conv_size_2, output_size, dropout_rate, lr, loss_function, optimizer_function).to(device)
        train_dataset =  torch.utils.data.TensorDataset(X_train, Y_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        X_val, Y_val = load_data("validate")
        val_dataset =  torch.utils.data.TensorDataset(X_val, Y_val)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

        val_acc_s = []
        val_loss_s = []
        train_acc_s = []
        train_loss_s = []
        for t in range(epochs):
            print("--------Begin the {} epoch training---------".format(t+1))
            train_loss, train_acc, train_precision, train_recall, train_f1 = train(dataloader=train_loader, model=cnn)
            val_loss, val_acc, val_precision, val_recall, val_f1  = validate(dataloader=val_loader, model=cnn)
            print("train loss = {}, acc = {}, precision = {}, recall = {}, f1 = {} ".format(round(train_loss, 4), round(train_acc, 4), round(train_precision, 4), round(train_recall, 4), round(train_f1, 4)))
            print("validation loss = {}, acc = {}, precision = {}, recall = {}, f1 = {}".format(round(val_loss, 4), round(val_acc, 4), round(val_precision, 4), round(val_recall, 4), round(val_f1, 4)))

            train_acc_s.append(train_acc)
            train_loss_s.append(train_loss)
            val_acc_s.append(val_acc)
            val_loss_s.append(val_loss)

            early_stopping(val_loss, cnn)
            if early_stopping.early_stop:
                print('early stopping')
                epochs = t+1 
                break



    '''save models'''
    torch.save({
        'epochs': epochs,
        'model_state_dict': cnn.state_dict(), 
        'loss_function': loss_function,
        'optimizer_function': optimizer_function,
        'output_size': output_size,
        'conv_size_1': conv_size_1,
        'conv_size_2': conv_size_2,
        'dropout_rate': dropout_rate,
        'learning_rate': lr,
        'input_size': input_size
    }, savename)
    

 
    '''plot loss-accuracy curve'''
    plt.plot(list(range(1, epochs+1)), train_loss_s, label = 'training loss')
    plt.plot(list(range(1, epochs+1)), val_loss_s, label = 'validation loss')
    plt.plot(list(range(1, epochs+1)), train_acc_s, label = 'training accuracy')
    plt.plot(list(range(1, epochs+1)), val_acc_s, label = 'validation accuracy')

    plt.xlabel("Epoch")  # x-coordinate axis
    plt.ylabel("Loss—Accuracy")  # y-coordinate axis

    plt.title('acc-loss curve')
    plt.legend(loc='upper left')
    plt.savefig(result_path + "figure.png", format='png', dpi=300)
    plt.close()

    print("Done!")


    '''test model'''
    X_test, Y_test = load_data("test")
    test_dataset =  torch.utils.data.TensorDataset(X_test, Y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    acc, precision, recall, f1 = test(dataloader=test_loader, model=cnn, device=device)
    print("test acc = {}, precision = {}, recall = {}, f1 = {}".format(acc, precision, recall, f1))

    # save test result
    with open(result_path + "test_result.txt", mode="w") as f:
        f.write("test result: \n")
        f.write("acc = " + str(round(acc, 4)) + "\n")
        f.write("precision = " + str(round(precision, 4)) + "\n")
        f.write("recall = " + str(round(recall, 4)) + "\n")
        f.write("f1 = " + str(round(f1, 4)) + "\n")
