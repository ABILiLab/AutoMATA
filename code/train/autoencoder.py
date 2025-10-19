import shutup  # ignore warning
shutup.please()
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
# os.chdir(os.path.dirname(__file__))
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import  StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from utils.earlystopping import EarlyStopping
from utils.FocalLoss import FocalLoss
import argparse
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

torch.manual_seed(2022)  # Setting the random seed. User can change it.

from DataProcess import load_data,process


class Autoencoder(nn.Module):  # 64, 32 ,16
    def __init__(self, input_dim, hidden_size_1=64, hidden_size_2=32, hidden_size_3=16, lr=0.001, dropout_rate=0.5):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size_1),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(True),
            nn.Linear(hidden_size_2, hidden_size_3),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size_3, hidden_size_2),
            nn.ReLU(True),
            nn.Linear(hidden_size_2, hidden_size_1),
            nn.ReLU(True),
            nn.Linear(hidden_size_1, input_dim),
        )

        self.criterion = nn.MSELoss()  # Autoencoder usually uses MSELoss to optimise the reconstruction error.
        self.learning_rate = lr
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


class Classifier(nn.Module):  # 16, 8, 2
    def __init__(self, hidden_size_3=16, cls_hidden_size=8, output_size=2, lr=0.001, loss_function="crossentropy", optimizer_function="adam"):
        self.output_size = output_size  # Record the output dimension of the flatten layer
        super(Classifier, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(hidden_size_3, cls_hidden_size),
            nn.ReLU(),
            nn.Linear(cls_hidden_size, output_size),
        )

        self.learning_rate = lr
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
            self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        elif optimizer_function == "sgd":
            self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0)  # User can change momentum
        elif optimizer_function == "rmsprop":
            self.optimizer = optim.RMSprop(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        # 一审
        logits = self.fc(x)
        if self.loss_function == "nllloss":
            return nn.functional.log_softmax(logits, dim=1)
        return logits



# train Autoencoder
def train_autoencoder(model, dataloader):
    train_loss = 0
    for data in dataloader:
        inputs, _ = data[0].to(device), data[1].to(device)
        model.optimizer.zero_grad()
        decoded, _ = model(inputs)
        loss = model.criterion(decoded, inputs)
        loss.backward()
        model.optimizer.step()
        train_loss += loss.item()

    train_loss /= len(dataloader)  # Loss of one epoch
    return train_loss

# validate Autoencoder
def val_autoencoder(model, val_dataloader):
    val_loss = 0
    with torch.no_grad(): # Context manager that disables gradient computation in the code blocks it wraps to reduce memory usage and speed up computation.
        for data in val_dataloader:
            inputs, _ = data[0].to(device), data[1].to(device)
            decoded, _  = model(inputs)
            val_loss += model.criterion(decoded, inputs).item() # Calculate and accumulate losses
    val_loss /= len(val_dataloader)  # Loss of one epoch
    return val_loss

# train classifier
def train_classifier(classifier, dataloader):
    true_label_list, pred_label_list= [], []
    train_loss = 0
    classifier.train()
    for data in dataloader:
        features, labels = data[0].to(device), data[1].to(device)
        classifier.optimizer.zero_grad()
        outputs = classifier(features)
        loss = classifier.criterion(outputs, labels)
        loss.backward()
        classifier.optimizer.step()
        train_loss += loss.item()
        true_label_list.append(labels.cpu().detach().numpy())
        pred_label_list.append(outputs.argmax(dim=1).cpu().detach().numpy())


    train_loss /= len(dataloader)
    y_true = np.concatenate(true_label_list)
    y_pred = np.concatenate(pred_label_list)
    train_acc = accuracy_score(y_true,y_pred)
    train_precision, train_recall, train_f1 = precision_recall_fscore_support(y_true, y_pred, average='weighted')[:-1]

    return train_loss, train_acc, train_precision, train_recall, train_f1

# validate classifier
def val_classifier(classifier, dataloader):
    true_label_list, pred_label_list= [], []
    classifier.eval()
    val_loss = 0 
    with torch.no_grad():
        for data in dataloader:
            features, labels = data[0].to(device), data[1].to(device)
            outputs = classifier(features)
            loss = classifier.criterion(outputs, labels)
            val_loss += loss.item()
            true_label_list.append(labels.cpu().detach().numpy())
            pred_label_list.append(outputs.argmax(dim=1).cpu().detach().numpy()) 

    y_true = np.concatenate(true_label_list)
    y_pred = np.concatenate(pred_label_list)
    val_loss /= len(dataloader)
    val_acc = accuracy_score(y_true,y_pred)
    val_precision, val_recall, val_f1 = precision_recall_fscore_support(y_true, y_pred, average='macro')[:-1]
    
    return val_loss, val_acc, val_precision, val_recall, val_f1


def extract_features(autoencoder, dataloader, device):
    encoder = autoencoder.encoder
    features = []
    labels = []
    for data in dataloader:
        inputs, label = data
        inputs, label  = inputs.to(device), label.to(device)
        with torch.no_grad():
            encoded_features = encoder(inputs)
        features.append(encoded_features)
        labels.append(label)
    # concatenate features and labels
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    # construct dataloader
    dataset =  torch.utils.data.TensorDataset(features, labels)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return loader


# save models
def save_model(autoencoder, classifier, autoencoder_path, classifier_path):
    torch.save(autoencoder.state_dict(), autoencoder_path)
    torch.save(classifier.state_dict(), classifier_path)
    print("model save successfully!")

# load models
def load_model(autoencoder, classifier, autoencoder_path, classifier_path):
    autoencoder.load_state_dict(torch.load(autoencoder_path))
    classifier.load_state_dict(torch.load(classifier_path))
    print("model load successfully!")

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

# test model
def test(dataloader, autoencoder, classifier, device):
    loader = extract_features(autoencoder, dataloader, device) 
    true_label_list, pred_label_list= [], []

    classifier.eval()
    with torch.no_grad():
        for data in loader:
            features, labels = data[0].to(device), data[1].to(device)
            outputs = classifier(features)
            true_label_list.append(labels.cpu().detach().numpy())
            pred_label_list.append(outputs.argmax(dim=1).cpu().detach().numpy())

    y_true = np.concatenate(true_label_list)
    y_pred = np.concatenate(pred_label_list)
    acc = accuracy_score(y_true,y_pred)
    precision, recall, f1 = precision_recall_fscore_support(y_true, y_pred, average='macro')[:-1]
    return acc, precision, recall, f1


if __name__ == '__main__':

    # cmd: nohup python rnn.py --jobID subtype_811 --epochs 150 --es 20 --lr 0.005 --bs 48 --output_size 4 --type all > /xp/www/AutoMATA/download_data/Jobs/subtype_811/result/rnn.log 2>&1
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
    parser.add_argument('--random_seed', type=int, default=42, help='随机种子')  # 一审新增

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
    
    print('model = AutoEncoder')
    print('kfold =', kfold)
    print('ratio =', ratio)
    print('epochs =', epochs)
    print('earlystopping =', es)
    print('learning rate =', lr)
    print('batch size =', batch_size)
    print('loss function =', loss_function)
    print('optimizer function =', optimizer_function)
    print('label number =', output_size)
    random_seed = args.random_seed  # 一审新增
    print('random seed =', random_seed)  # 一审新增
    set_random_seed(random_seed)  # 一审新增

    # the hidden size of the autoencoder, user can change it.
    hidden_size_1=64
    hidden_size_2=32
    hidden_size_3=16
    cls_hidden_size=8
    dropout_rate = 0.0  # the ratio of dropout

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_path = './result/'
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    savename = result_path+'model_autoencoder.pth'
    savename_cls = result_path+'model_cls.pth'
    
    early_stopping = EarlyStopping(es, verbose=True, savename=savename, delta=0.0001)
    early_stopping_cls = EarlyStopping(es, verbose=True, savename=savename_cls, delta=0.0001)
    
    # need split a dataset into train, validation and test datasets by ratio
    if (ratio != "0"):
        process(ratio=ratio)
    

    # load training datasets
    X_train, Y_train = load_data("train")
    input_dim = X_train.shape[1]

    # begin training
    if (kfold):
        kfscore = []
        skf = StratifiedKFold(n_splits=kfold)
        for i, (train_idx, val_idx) in enumerate(skf.split(X_train, Y_train)):
            early_stopping = EarlyStopping(es, verbose=True, savename=savename, delta=0.0001)  # Each fold should open a new early stop
            early_stopping_cls = EarlyStopping(es, verbose=True, savename=savename_cls, delta=0.0001)
            print("--------The {} fold is training---------".format(i+1))
            trainset, valset = torch.FloatTensor(np.array(X_train)[train_idx]), torch.FloatTensor(np.array(X_train)[val_idx])
            traintag, valtag = torch.LongTensor(np.array(Y_train)[train_idx]), torch.LongTensor(np.array(Y_train)[val_idx])

            train_dataset = torch.utils.data.TensorDataset(trainset, traintag)
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            val_dataset =  torch.utils.data.TensorDataset(valset, valtag)
            val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

            # define model
            autoencoder = Autoencoder(input_dim, hidden_size_1, hidden_size_2, hidden_size_3, lr, dropout_rate).to(device)
            classifier = Classifier(hidden_size_3, cls_hidden_size, output_size, lr, loss_function, optimizer_function).to(device)  # 输出分类概率

            val_loss_s = []  # Store the validation loss value for each epoch
            train_loss_s = []  # Store the training loss value for each epoch
            print("--------Training Autoencoder--------")
            for t in range(epochs):
                print("--------Begin the {} epoch training---------".format(t+1))
                # training Autoencoder
                train_loss = train_autoencoder(autoencoder, train_loader)
                # validate Autoencoder
                val_loss = val_autoencoder(autoencoder, val_loader)
                print("train loss = {}".format(train_loss))
                print("validation loss = {}".format(val_loss))

                train_loss_s.append(train_loss)
                val_loss_s.append(val_loss)

                early_stopping(val_loss, autoencoder)
                if early_stopping.early_stop:
                    print('early stopping')
                    epochs = t+1  # Save the number of epochs
                    break
            print("--------Autoencoder Training Ends--------")


            val_acc_s_cls = []  # Store the validation accuracy of each epoch
            val_loss_s_cls = []  # Store the validation loss of each epoch
            train_acc_s_cls = []  # Store the training accuracy of each epoch
            train_loss_s_cls = []  # Store the training loss of each epoch
            print("--------Training Classifier--------")
            # Extracting training/validation set features using trained Autoencoder
            train_loader = extract_features(autoencoder, train_loader, device)
            val_loader = extract_features(autoencoder, val_loader, device)
            
            for t in range(epochs):
                print("--------Begin the {} epoch training---------".format(t+1))
                # training classifier
                train_loss, train_acc, train_precision, train_recall, train_f1 = train_classifier(classifier, train_loader)
                # validate classifier
                val_loss, val_acc, val_precision, val_recall, val_f1 = val_classifier(classifier, val_loader)
                print("train loss = {}, acc = {}, precision = {}, recall = {}, f1 = {} ".format(round(train_loss, 4), round(train_acc, 4), round(train_precision, 4), round(train_recall, 4), round(train_f1, 4)))
                print("validation loss = {}, acc = {}, precision = {}, recall = {}, f1 = {}".format(round(val_loss, 4), round(val_acc, 4), round(val_precision, 4), round(val_recall, 4), round(val_f1, 4)))

                train_loss_s_cls.append(train_loss)
                val_loss_s_cls.append(val_loss)
                val_acc_s_cls.append(val_acc)
                train_acc_s_cls.append(train_acc)

                early_stopping_cls(val_loss, classifier)
                if early_stopping_cls.early_stop:
                    print('early stopping')
                    epochs = t+1
                    break
            print("--------Classifier Training Ends--------")

            print("--------The {} fold validation result---------".format(i+1))
            _, val_acc, val_precision, val_recall, val_f1 = val_classifier(classifier, val_loader)
            print("validation acc = {}, precision = {}, recall = {}, f1 = {}".format(round(val_acc, 4), round(val_precision, 4), round(val_recall, 4), round(val_f1, 4)))
            kfscore.append(val_classifier(classifier, val_loader))

        # average score
        kfscore = np.array(kfscore).sum(axis= 0)/float(kfold)  # acc, precision, recall, f1
        print("--------KFold Final Average Validation Results---------")
        print("Stratified KFold mean validation acc = {}, precision = {}, recall = {}, f1 = {}".format(round(kfscore[1], 4), round(kfscore[2], 4), round(kfscore[3], 4), round(kfscore[4], 4)))


    else:
        autoencoder = Autoencoder(input_dim, hidden_size_1, hidden_size_2, hidden_size_3, lr, dropout_rate).to(device)
        classifier = Classifier(hidden_size_3, cls_hidden_size, output_size, lr, loss_function, optimizer_function).to(device)  # 输出分类概率
        
        train_dataset =  torch.utils.data.TensorDataset(X_train, Y_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        X_val, Y_val = load_data("validate")
        val_dataset =  torch.utils.data.TensorDataset(X_val, Y_val)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

        # train and validate Autoencoder
        val_loss_s = []  # Store the validation loss value for each epoch
        train_loss_s = []  # Store the training loss value for each epoch
        print("--------Training Autoencoder--------")
        for t in range(epochs):
            print("--------Begin the {} epoch training---------".format(t+1))
            train_loss = train_autoencoder(autoencoder, train_loader)
            val_loss = val_autoencoder(autoencoder, val_loader)
            print("train loss = {}".format(train_loss))
            print("validation loss = {}".format(val_loss))

            train_loss_s.append(train_loss)
            val_loss_s.append(val_loss)


            early_stopping(val_loss, autoencoder)
            if early_stopping.early_stop:
                print('early stopping')
                epochs = t+1 
                break
        print("--------Autoencoder Training Ends--------")


        # train and validate classifier
        val_acc_s_cls = [] 
        val_loss_s_cls = [] 
        train_acc_s_cls = []  
        train_loss_s_cls = [] 
        print("--------Training Classifier--------")
        # Extracting training set features using trained Autoencoder
        train_loader = extract_features(autoencoder, train_loader, device)
        val_loader = extract_features(autoencoder, val_loader, device)
        
        for t in range(epochs):
            print("--------Begin the {} epoch training---------".format(t+1))
            train_loss, train_acc, train_precision, train_recall, train_f1 = train_classifier(classifier, train_loader)
            val_loss, val_acc, val_precision, val_recall, val_f1 = val_classifier(classifier, val_loader)
            print("train loss = {}, acc = {}, precision = {}, recall = {}, f1 = {} ".format(round(train_loss, 4), round(train_acc, 4), round(train_precision, 4), round(train_recall, 4), round(train_f1, 4)))
            print("validation loss = {}, acc = {}, precision = {}, recall = {}, f1 = {}".format(round(val_loss, 4), round(val_acc, 4), round(val_precision, 4), round(val_recall, 4), round(val_f1, 4)))


            train_loss_s_cls.append(train_loss)
            val_loss_s_cls.append(val_loss)
            val_acc_s_cls.append(val_acc)
            train_acc_s_cls.append(train_acc)


            early_stopping_cls(val_loss, classifier)
            if early_stopping_cls.early_stop:
                print('early stopping')
                epochs = t+1 
                break
        print("--------Classifier Training Ends--------")
    


    '''plot loss-accuracy curve'''
    plt.plot(list(range(1, epochs+1)), train_loss_s_cls, label = 'training loss')
    plt.plot(list(range(1, epochs+1)), val_loss_s_cls, label = 'validation loss') 
    plt.plot(list(range(1, epochs+1)), train_acc_s_cls, label = 'training accuracy') 
    plt.plot(list(range(1, epochs+1)), val_acc_s_cls, label = 'validation accuracy')  

    plt.xlabel("Epoch")  # x-coordinate axis
    plt.ylabel("Loss—Accuracy")  # y-coordinate axis

    plt.title('acc-loss curve')
    plt.legend(loc='upper left')
    plt.savefig(result_path + "figure.png", format='png', dpi=300)
    plt.close()

    print("Done!")  # training and plotting end

    '''save models'''
    torch.save({
        'epochs': epochs,
        'model_state_dict': autoencoder.state_dict(), 
        'input_dim': input_dim,
        'hidden_size_1': hidden_size_1,
        'hidden_size_2': hidden_size_2,
        'hidden_size_3': hidden_size_3,
        'learning_rate': lr,
        'dropout_rate': dropout_rate
    }, savename)

    torch.save({
        'model_state_dict': classifier.state_dict(), 
        'hidden_size_3': hidden_size_3,
        'cls_hidden_size': cls_hidden_size,
        'output_size': output_size,
        'learning_rate': lr,
        'loss_function': loss_function,
        'optimizer_function': optimizer_function
    }, savename_cls)


    '''test model'''
    X_test, Y_test = load_data("test")
    test_dataset =  torch.utils.data.TensorDataset(X_test, Y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    acc, precision, recall, f1 = test(dataloader=test_loader, autoencoder = autoencoder, classifier = classifier, device=device)
    print("test acc = {}, precision = {}, recall = {}, f1 = {}".format(acc, precision, recall, f1))

    # save test result
    with open(result_path + "test_result.txt", mode="w") as f:
        f.write("test result: \n")
        f.write("acc = " + str(round(acc, 4)) + "\n")
        f.write("precision = " + str(round(precision, 4)) + "\n")
        f.write("recall = " + str(round(recall, 4)) + "\n")
        f.write("f1 = " + str(round(f1, 4)) + "\n")





