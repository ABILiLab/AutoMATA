import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import os
import json
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

import warnings
warnings.filterwarnings('ignore')

class PseudoLabelNetwork(nn.Module):
    """
    Pseudo-label semi-supervised learning network
    For multi-classification tasks of semi-supervised learning
    """
    def __init__(self, input_dim, hidden_dims, num_classes, dropout_rate=0.1):
        super(PseudoLabelNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Classifier
        self.classifier = nn.Linear(prev_dim, num_classes)
        
        # Confidence threshold
        self.confidence_threshold = 0.8
    
    def forward(self, x):
        """Forward propagation"""
        encoded = self.encoder(x)
        logits = self.classifier(encoded)
        return logits
    
    def predict_with_confidence(self, x):
        """Predict and return confidence"""
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            max_probs, predictions = torch.max(probabilities, dim=1)
            return predictions, max_probs, probabilities

class PseudoLabelLoss(nn.Module):
    """
    Pseudo-label loss function
    Combine supervised loss and pseudo-label loss
    """
    def __init__(self, alpha=1.0, beta=0.5, reduction='mean'):
        super(PseudoLabelLoss, self).__init__()
        self.alpha = alpha  # Supervised loss weight
        self.beta = beta    # Pseudo-label loss weight
        self.reduction = reduction
    
    def forward(self, logits, targets, pseudo_targets, pseudo_weights):
        # Supervised loss
        if targets is not None and len(targets) > 0:
            # Separate labeled and unlabeled data
            labeled_mask = targets != -1  # -1 represents unlabeled data
            if labeled_mask.sum() > 0:  # There are labeled data
                labeled_logits = logits[labeled_mask]
                labeled_targets = targets[labeled_mask]
                supervised_loss = F.cross_entropy(labeled_logits, labeled_targets, reduction=self.reduction)
            else:
                supervised_loss = torch.tensor(0.0, device=logits.device)
        else:
            supervised_loss = torch.tensor(0.0, device=logits.device)
        
        # Pseudo-label loss
        if pseudo_targets is not None and len(pseudo_targets) > 0:
            pseudo_loss = F.cross_entropy(logits, pseudo_targets, reduction='none')
            pseudo_loss = (pseudo_loss * pseudo_weights).mean()
        else:
            pseudo_loss = torch.tensor(0.0, device=logits.device)
        
        total_loss = self.alpha * supervised_loss + self.beta * pseudo_loss
        
        return total_loss, supervised_loss, pseudo_loss

class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification"""
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Separate labeled and unlabeled data
        labeled_mask = targets != -1  # -1 represents unlabeled data
        if labeled_mask.sum() > 0:  # There are labeled data
            labeled_inputs = inputs[labeled_mask]
            labeled_targets = targets[labeled_mask]
            ce_loss = F.cross_entropy(labeled_inputs, labeled_targets, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
            
            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss
        else:
            return torch.tensor(0.0, device=inputs.device)

class LabelSmoothingLoss(nn.Module):
    """Label Smoothing Loss"""
    def __init__(self, num_classes, smoothing=0.1, reduction='mean'):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Separate labeled and unlabeled data
        labeled_mask = targets != -1  # -1 represents unlabeled data
        if labeled_mask.sum() > 0:  # There are labeled data
            labeled_inputs = inputs[labeled_mask]
            labeled_targets = targets[labeled_mask]
            
            log_preds = F.log_softmax(labeled_inputs, dim=1)
            true_dist = torch.zeros_like(log_preds)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, labeled_targets.unsqueeze(1), 1 - self.smoothing)
            
            loss = -torch.sum(true_dist * log_preds, dim=1)
            
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss
        else:
            return torch.tensor(0.0, device=inputs.device)

class ContrastiveLoss(nn.Module):
    """Contrastive Loss for semi-supervised learning"""
    def __init__(self, margin=1.0, temperature=0.1, reduction='mean'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, features, labels):
        # Separate labeled and unlabeled data
        labeled_mask = labels != -1  # -1 represents unlabeled data
        if labeled_mask.sum() > 0:  # There are labeled data
            labeled_features = features[labeled_mask]
            labeled_labels = labels[labeled_mask]
            
            features_norm = F.normalize(labeled_features, p=2, dim=1)
            similarity_matrix = torch.mm(features_norm, features_norm.t()) / self.temperature
            
            labels_equal = labeled_labels.unsqueeze(0) == labeled_labels.unsqueeze(1)
            
            exp_sim = torch.exp(similarity_matrix)
            pos_sim = exp_sim * labels_equal.float()
            neg_sim = exp_sim * (1 - labels_equal.float())
            
            pos_sum = pos_sim.sum(dim=1, keepdim=True)
            neg_sum = neg_sim.sum(dim=1, keepdim=True)
            
            loss = -torch.log(pos_sum / (pos_sum + neg_sum + 1e-8))
            
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss
        else:
            return torch.tensor(0.0, device=features.device)

class ModelEvaluator:
    """Semi-supervised learning model evaluator"""
    def __init__(self, device):
        self.device = device
        self.metrics = {}
    
    def evaluate_classification(self, model, data_loader, scaler):
        """Evaluate classification performance"""
        model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, targets in data_loader:
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                logits = model(data)
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate classification metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='weighted')
        recall = recall_score(all_targets, all_predictions, average='weighted')
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        
        # Calculate metrics for each class
        precision_per_class = precision_score(all_targets, all_predictions, average=None)
        recall_per_class = recall_score(all_targets, all_predictions, average=None)
        f1_per_class = f1_score(all_targets, all_predictions, average=None)
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        classification_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }
        
        return classification_metrics
    
    def compute_comprehensive_metrics(self, model, data_loader, scaler):
        """
        Calculate comprehensive evaluation metrics
        """
        print("Start comprehensive model evaluation...")
        
        # evaluate classification performance
        print("Evaluate classification performance...")
        classification_metrics = self.evaluate_classification(model, data_loader, scaler)
        
                
        comprehensive_metrics = {
            'classification': classification_metrics
        }
        
        return comprehensive_metrics
    
def visualize_results(metrics, save_path=None):
    """Visualize evaluation results"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Pseudo-label semi-supervised learning evaluation results', fontsize=16, fontweight='bold')
    
    # 1. Confusion matrix
    ax1 = axes[0]
    cm = metrics['classification']['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title(f'Confusion matrix (accuracy: {metrics["classification"]["accuracy"]:.3f})')
    ax1.set_xlabel('Predicted label')
    ax1.set_ylabel('True label')
    
    # 2. Classification performance metrics
    ax2 = axes[1]
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 score']
    metrics_values = [
        metrics['classification']['accuracy'],
        metrics['classification']['precision'],
        metrics['classification']['recall'],
        metrics['classification']['f1_score']
    ]
    bars = ax2.bar(metrics_names, metrics_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    ax2.set_ylabel('Scores')
    ax2.set_title('Classification performance metrics')
    ax2.set_ylim(0, 1)
    
    # Add numerical labels
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. F1 score per class
    ax3 = axes[2]
    f1_per_class = metrics['classification']['f1_per_class']
    class_names = [f'Class {i}' for i in range(len(f1_per_class))]
    bars = ax3.bar(class_names, f1_per_class, color='lightblue')
    ax3.set_ylabel('F1 score')
    ax3.set_title('F1 score per class')
    ax3.set_ylim(0, 1)
    ax3.tick_params(axis='x', rotation=45)
    
    # Add numerical labels
    for bar, value in zip(bars, f1_per_class):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()

def print_evaluation_report(metrics):
    """
    Print detailed evaluation report
    """
    print("\n" + "="*60)
    print("Pseudo-label semi-supervised learning evaluation report")
    print("="*60)
    
    # classification performance
    print("\n Classification performance evaluation:")
    print("-" * 30)
    print(f"Accuracy: {metrics['classification']['accuracy']:.6f}")
    print(f"Precision: {metrics['classification']['precision']:.6f}")
    print(f"Recall: {metrics['classification']['recall']:.6f}")
    print(f"F1-Score: {metrics['classification']['f1_score']:.6f}")
    
    print("\nDetailed metrics for each class:")
    print("-" * 30)
    for i, (precision, recall, f1) in enumerate(zip(
        metrics['classification']['precision_per_class'],
        metrics['classification']['recall_per_class'],
        metrics['classification']['f1_per_class']
    )):
        print(f"Class {i}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    
    

class EarlyStopping:
    """Early stopping mechanism"""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """Save best model weights"""
        self.best_weights = model.state_dict().copy()

def set_random_seed(seed):
    """Set random seed to ensure reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.Generator().manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_data(seed, batch_size, ratio="8:1:1", jobid="example_pseudo", use_unlabeled=True, kfold=0): 
    """
    Prepare training, validation and test data
    The first column is the sample name, the last column is the label
    Support semi-supervised learning: use unlabeled data
    Training set has unlabeled samples (labeled as 'Unknown'), validation set and test set must have labels
    """
    if ratio == "0" and kfold == 0: 
        scaler = StandardScaler()
        label_encoder = LabelEncoder()
        
        # Directly load training-validation-test set file
        # train_data = pd.read_csv(f"./train_example/{jobid}_data.txt", sep="\t")
        # train_data = pd.read_csv("D:\\wamp\www\\multi_omics_own\\download_data\\Jobs\\"+jobid+"\\"+jobid+"_data.txt", sep="\t")
        train_data = pd.read_csv("../../data/train_example_semi/train_example_semi_data.txt", sep="\t")
        train_data = train_data.dropna()
        train_features = train_data.iloc[:, 1:-1].values.astype(float)  # the middle columns are features
        train_labels = train_data.iloc[:, -1].values  # the last column is label
        
        # Separate labeled and unlabeled data
        labeled_mask = train_labels != 'Unknown'
        unlabeled_mask = train_labels == 'Unknown'
        
        labeled_features = train_features[labeled_mask]
        labeled_labels = train_labels[labeled_mask]
        unlabeled_features = train_features[unlabeled_mask]
        
        # Encode labels
        labeled_labels_encoded = label_encoder.fit_transform(labeled_labels)
        
        # Standardize features
        scaler.fit(labeled_features)  # Only use labeled data to fit scaler
        labeled_features_scaled = scaler.transform(labeled_features)
        unlabeled_features_scaled = scaler.transform(unlabeled_features)
        
        # Create labeled data loader
        labeled_data_tensor = torch.FloatTensor(labeled_features_scaled)
        labeled_labels_tensor = torch.LongTensor(labeled_labels_encoded)
        labeled_loader = DataLoader(TensorDataset(labeled_data_tensor, labeled_labels_tensor), 
                                  batch_size=batch_size, shuffle=True)
        
        # Create unlabeled data loader
        unlabeled_data_tensor = torch.FloatTensor(unlabeled_features_scaled)
        unlabeled_loader = DataLoader(TensorDataset(unlabeled_data_tensor), 
                                    batch_size=batch_size, shuffle=True)
        
        # Merge labeled and unlabeled data for training
        if use_unlabeled and len(unlabeled_features) > 0:
            # Create semi-supervised training data loader
            all_features = np.vstack([labeled_features_scaled, unlabeled_features_scaled])
            all_labels = np.concatenate([labeled_labels_encoded, 
                                       np.full(len(unlabeled_features), -1)])  # -1 represents unlabeled data
            
            all_data_tensor = torch.FloatTensor(all_features)
            all_labels_tensor = torch.LongTensor(all_labels)
            train_loader = DataLoader(TensorDataset(all_data_tensor, all_labels_tensor), 
                                    batch_size=batch_size, shuffle=True)
        else:
            train_loader = labeled_loader
        
        # Validation and test data
        # val_data = pd.read_csv(f"./train_example/{jobid}_val.txt", sep="\t")
        # val_data = pd.read_csv("D:\\wamp\www\\multi_omics_own\\download_data\\Jobs\\"+jobid+"\\"+jobid+"_val.txt", sep="\t")
        val_data = pd.read_csv("../../data/train_example_semi/train_example_semi_val.txt", sep="\t")
        val_data = val_data.dropna()
        val_features = val_data.iloc[:, 1:-1].values.astype(float)
        val_labels = val_data.iloc[:, -1].values
        
        # Process validation set labels
        try:
            val_labels_encoded = label_encoder.transform(val_labels)
        except ValueError as e:
            val_labels_str = val_labels.astype(str)
            all_labels = np.concatenate([labeled_labels, val_labels_str])
            label_encoder.fit(all_labels)
            val_labels_encoded = label_encoder.transform(val_labels_str)
        
        val_features_scaled = scaler.transform(val_features)
        val_data_tensor = torch.FloatTensor(val_features_scaled)
        val_labels_tensor = torch.LongTensor(val_labels_encoded)
        val_loader = DataLoader(TensorDataset(val_data_tensor, val_labels_tensor), 
                              batch_size=batch_size, shuffle=False)

        # test_data = pd.read_csv(f"./train_example/{jobid}_test.txt", sep="\t")
        # test_data = pd.read_csv("D:\\wamp\www\\multi_omics_own\\download_data\\Jobs\\"+jobid+"\\"+jobid+"_test.txt", sep="\t")
        test_data = pd.read_csv("../../data/train_example_semi/train_example_semi_test.txt", sep="\t")
        test_data = test_data.dropna()
        test_features = test_data.iloc[:, 1:-1].values.astype(float)
        test_labels = test_data.iloc[:, -1].values
        
        # Process test set labels
        try:
            test_labels_encoded = label_encoder.transform(test_labels)
        except ValueError as e:
            test_labels_str = test_labels.astype(str)
            all_labels = np.concatenate([labeled_labels, test_labels_str])
            label_encoder.fit(all_labels)
            test_labels_encoded = label_encoder.transform(test_labels_str)
        
        test_features_scaled = scaler.transform(test_features)
        test_data_tensor = torch.FloatTensor(test_features_scaled)
        test_labels_tensor = torch.LongTensor(test_labels_encoded)
        test_loader = DataLoader(TensorDataset(test_data_tensor, test_labels_tensor), 
                               batch_size=batch_size, shuffle=False)
        
        input_dim = train_features.shape[1]
    elif ratio != "0" and kfold == 0:  
        # Split training-validation-test set
        # data = pd.read_csv(f"./train_example/{jobid}_data.txt", sep="\t")
        data = pd.read_csv("../../data/train_example_semi/train_example_semi_data.txt", sep="\t")
        sample_names = data.iloc[:, 0].values  # the first column is sample name
        features = data.iloc[:, 1:-1].values.astype(float)  # the middle columns are features
        labels = data.iloc[:, -1].values  # the last column is label
        
        scaler = StandardScaler()
        label_encoder = LabelEncoder()
        
        # Separate labeled and unlabeled data
        labeled_mask = labels != 'Unknown'
        unlabeled_mask = labels == 'Unknown'
        
        labeled_features = features[labeled_mask]
        labeled_labels = labels[labeled_mask]
        unlabeled_features = features[unlabeled_mask]
        
        # Encode labels
        labeled_labels_encoded = label_encoder.fit_transform(labeled_labels)
        
        # Standardize features
        scaler.fit(labeled_features)  # Only use labeled data to fit scaler
        labeled_features_scaled = scaler.transform(labeled_features)
        unlabeled_features_scaled = scaler.transform(unlabeled_features)
        
        # Separate labeled data
        ratio_str = ratio.split(":")
        ratio_num = list(map(int, ratio_str))
        train_ratio = ratio_num[0] / sum(ratio_num)
        test_ratio = ratio_num[2] / sum(ratio_num[1:])
        
        # Check if the data amount is enough for stratified sampling
        unique_labels, label_counts = np.unique(labeled_labels_encoded, return_counts=True)
        min_class_count = np.min(label_counts)
        
        if min_class_count < 2:
            print(f"Warning: Some classes have too few samples (minimum {min_class_count} samples), using random split instead of stratified sampling")
            stratify_train = None
            stratify_val = None
        else:
            stratify_train = labeled_labels_encoded
            stratify_val = None
        
        train_features, res_features, train_labels, res_labels = train_test_split(
            labeled_features_scaled, labeled_labels_encoded, test_size=1-train_ratio, 
            random_state=seed, stratify=stratify_train
        )
        
        # Check if the validation set split needs stratified sampling
        if len(res_labels) > 0:
            unique_res_labels, res_label_counts = np.unique(res_labels, return_counts=True)
            min_res_class_count = np.min(res_label_counts)
            
            if min_res_class_count < 2:
                print(f"Warning: Some classes have too few samples (minimum {min_res_class_count} samples) during validation set split, using random split instead of stratified sampling")
                stratify_val = None
            else:
                stratify_val = res_labels
        else:
            stratify_val = None
        
        val_features, test_features, val_labels, test_labels = train_test_split(
            res_features, res_labels, test_size=test_ratio, random_state=seed, stratify=stratify_val
        )
        
        # Create training data loader (contains unlabeled data)
        if use_unlabeled and len(unlabeled_features) > 0:
            # Add unlabeled data to the training set
            all_train_features = np.vstack([train_features, unlabeled_features_scaled])
            all_train_labels = np.concatenate([train_labels, np.full(len(unlabeled_features), -1)])
            
            train_data_tensor = torch.FloatTensor(all_train_features)
            train_labels_tensor = torch.LongTensor(all_train_labels)
        else:
            train_data_tensor = torch.FloatTensor(train_features)
            train_labels_tensor = torch.LongTensor(train_labels)
        
        train_loader = DataLoader(TensorDataset(train_data_tensor, train_labels_tensor), 
                                batch_size=batch_size, shuffle=True)
        
        val_data_tensor = torch.FloatTensor(val_features)
        val_labels_tensor = torch.LongTensor(val_labels)
        val_loader = DataLoader(TensorDataset(val_data_tensor, val_labels_tensor), 
                              batch_size=batch_size, shuffle=False)
        
        test_data_tensor = torch.FloatTensor(test_features)
        test_labels_tensor = torch.LongTensor(test_labels)
        test_loader = DataLoader(TensorDataset(test_data_tensor, test_labels_tensor), 
                               batch_size=batch_size, shuffle=False)
        
        # Create unlabeled data loader
        if use_unlabeled and len(unlabeled_features) > 0:
            unlabeled_data_tensor = torch.FloatTensor(unlabeled_features_scaled)
            unlabeled_loader = DataLoader(TensorDataset(unlabeled_data_tensor), 
                                        batch_size=batch_size, shuffle=True)
        else:
            # Create an empty DataLoader
            empty_data = torch.FloatTensor([])
            unlabeled_loader = DataLoader(TensorDataset(empty_data), 
                                        batch_size=batch_size, shuffle=True)
        
        input_dim = features.shape[1]

    else:  
        scaler = StandardScaler()
        label_encoder = LabelEncoder()
        
        # Directly load training-validation-test set file
        # train_data = pd.read_csv(f"./train_example/{jobid}_data.txt", sep="\t")
        # train_data = pd.read_csv("D:\\wamp\www\\multi_omics_own\\download_data\\Jobs\\"+jobid+"\\"+jobid+"_data.txt", sep="\t")
        train_data = pd.read_csv("../../data/train_example_semi/train_example_semi_data.txt", sep="\t")
        train_data = train_data.dropna()
        train_features = train_data.iloc[:, 1:-1].values.astype(float)  # the middle columns are features
        train_labels = train_data.iloc[:, -1].values  # the last column is label
        
        # Separate labeled and unlabeled data
        labeled_mask = train_labels != 'Unknown'
        unlabeled_mask = train_labels == 'Unknown'
        
        labeled_features = train_features[labeled_mask]
        labeled_labels = train_labels[labeled_mask]
        unlabeled_features = train_features[unlabeled_mask]
        
        # Encode labels
        labeled_labels_encoded = label_encoder.fit_transform(labeled_labels)
        
        # Standardize features
        scaler.fit(labeled_features)  # Only use labeled data to fit scaler
        labeled_features_scaled = scaler.transform(labeled_features)
        unlabeled_features_scaled = scaler.transform(unlabeled_features)
        
        # Create labeled data loader
        labeled_data_tensor = torch.FloatTensor(labeled_features_scaled)
        labeled_labels_tensor = torch.LongTensor(labeled_labels_encoded)
        labeled_loader = DataLoader(TensorDataset(labeled_data_tensor, labeled_labels_tensor), 
                                  batch_size=batch_size, shuffle=True)
        
        # Create unlabeled data loader
        unlabeled_data_tensor = torch.FloatTensor(unlabeled_features_scaled)
        unlabeled_loader = DataLoader(TensorDataset(unlabeled_data_tensor), 
                                    batch_size=batch_size, shuffle=True)
        
        # Merge labeled and unlabeled data for training
        if use_unlabeled and len(unlabeled_features) > 0:
            # Create semi-supervised training data loader
            all_features = np.vstack([labeled_features_scaled, unlabeled_features_scaled])
            all_labels = np.concatenate([labeled_labels_encoded, 
                                       np.full(len(unlabeled_features), -1)])  # -1 represents unlabeled data
            
            all_data_tensor = torch.FloatTensor(all_features)
            all_labels_tensor = torch.LongTensor(all_labels)
            train_loader = DataLoader(TensorDataset(all_data_tensor, all_labels_tensor), 
                                    batch_size=batch_size, shuffle=True)
        else:
            train_loader = labeled_loader

        val_loader = 0
        test_data = pd.read_csv("../../data/train_example_semi/train_example_semi_test.txt", sep="\t")
        test_data = test_data.dropna()
        test_features = test_data.iloc[:, 1:-1].values.astype(float)
        test_labels = test_data.iloc[:, -1].values
        
        # Process test set labels
        try:
            test_labels_encoded = label_encoder.transform(test_labels)
        except ValueError as e:
            test_labels_str = test_labels.astype(str)
            all_labels = np.concatenate([labeled_labels, test_labels_str])
            label_encoder.fit(all_labels)
            test_labels_encoded = label_encoder.transform(test_labels_str)
        
        test_features_scaled = scaler.transform(test_features)
        test_data_tensor = torch.FloatTensor(test_features_scaled)
        test_labels_tensor = torch.LongTensor(test_labels_encoded)
        test_loader = DataLoader(TensorDataset(test_data_tensor, test_labels_tensor), 
                               batch_size=batch_size, shuffle=False)
        
        input_dim = train_features.shape[1]
        
    num_classes = len(np.unique(labeled_labels_encoded))
    
    return input_dim, num_classes, train_loader, val_loader, test_loader, scaler, unlabeled_loader

def generate_pseudo_labels(model, unlabeled_loader, device, confidence_threshold=0.8):
    """Generate pseudo labels"""
    model.eval()
    pseudo_data = []
    pseudo_labels = []
    pseudo_weights = []
    
    with torch.no_grad():
        for data in unlabeled_loader:
            data = data[0].to(device)  # Unlabeled data only has features
            predictions, confidences, probabilities = model.predict_with_confidence(data)
            
            # Select high confidence predictions as pseudo labels
            high_conf_mask = confidences >= confidence_threshold
            
            if high_conf_mask.sum() > 0:
                pseudo_data.append(data[high_conf_mask].cpu())
                pseudo_labels.append(predictions[high_conf_mask].cpu())
                pseudo_weights.append(confidences[high_conf_mask].cpu())
    
    if pseudo_data:
        pseudo_data = torch.cat(pseudo_data, dim=0)
        pseudo_labels = torch.cat(pseudo_labels, dim=0)
        pseudo_weights = torch.cat(pseudo_weights, dim=0)
        
        return pseudo_data, pseudo_labels, pseudo_weights
    else:
        return None, None, None

def train_epoch(model, train_loader, unlabeled_loader, optimizer, device, loss_function='pseudo_label', 
                alpha=1.0, beta=0.5, confidence_threshold=0.8, pseudo_ratio=0.5):
    """Train one epoch"""
    model.train()
    total_loss = 0
    total_supervised_loss = 0
    total_pseudo_loss = 0
    
    # Generate pseudo labels
    pseudo_data, pseudo_labels, pseudo_weights = generate_pseudo_labels(
        model, unlabeled_loader, device, confidence_threshold
    )
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        
        # Separate labeled and unlabeled data
        labeled_mask = targets != -1  # -1 represents unlabeled data
        unlabeled_mask = targets == -1
        
        if labeled_mask.sum() > 0:  # There are labeled data
            labeled_data = data[labeled_mask]
            labeled_targets = targets[labeled_mask]
        else:
            labeled_data = None
            labeled_targets = None
        
        if unlabeled_mask.sum() > 0:  # There are unlabeled data
            unlabeled_data = data[unlabeled_mask]
        else:
            unlabeled_data = None
        
        # Calculate loss
        if loss_function.lower() == 'pseudo_label':
            # Pseudo label loss
            loss_fn = PseudoLabelLoss(alpha=alpha, beta=beta)
            
            # Merge labeled data and pseudo data
            if labeled_data is not None and pseudo_data is not None:
                # Randomly select part of the pseudo data
                n_pseudo = min(len(pseudo_data), int(len(labeled_data) * pseudo_ratio))
                if n_pseudo > 0:
                    pseudo_indices = torch.randperm(len(pseudo_data))[:n_pseudo]
                    selected_pseudo_data = pseudo_data[pseudo_indices].to(device)
                    selected_pseudo_labels = pseudo_labels[pseudo_indices].to(device)
                    selected_pseudo_weights = pseudo_weights[pseudo_indices].to(device)
                    
                    # Merge data
                    combined_data = torch.cat([labeled_data, selected_pseudo_data], dim=0)
                    combined_targets = torch.cat([labeled_targets, selected_pseudo_labels], dim=0)
                    combined_weights = torch.cat([torch.ones(len(labeled_targets), device=device), 
                                                selected_pseudo_weights], dim=0)
                else:
                    combined_data = labeled_data
                    combined_targets = labeled_targets
                    combined_weights = torch.ones(len(labeled_targets), device=device)
            elif labeled_data is not None:
                combined_data = labeled_data
                combined_targets = labeled_targets
                combined_weights = torch.ones(len(labeled_targets), device=device)
            else:
                combined_data = None
                combined_targets = None
                combined_weights = None
            
            if combined_data is not None:
                logits = model(combined_data)
                loss, supervised_loss, pseudo_loss = loss_fn(logits, combined_targets, 
                                                           combined_targets, combined_weights)
            else:
                loss = torch.tensor(0.0, device=device)
                supervised_loss = torch.tensor(0.0, device=device)
                pseudo_loss = torch.tensor(0.0, device=device)
                
        elif loss_function.lower() == 'focal':
            if labeled_data is not None:
                logits = model(labeled_data)
                focal_loss_fn = FocalLoss(alpha=1.0, gamma=2.0)
                supervised_loss = focal_loss_fn(logits, labeled_targets)
            else:
                supervised_loss = torch.tensor(0.0, device=device)
            
            pseudo_loss = torch.tensor(0.0, device=device)
            loss = supervised_loss
            
        elif loss_function.lower() == 'label_smoothing':
            if labeled_data is not None:
                logits = model(labeled_data)
                label_smooth_loss_fn = LabelSmoothingLoss(num_classes=model.num_classes, smoothing=0.1)
                supervised_loss = label_smooth_loss_fn(logits, labeled_targets)
            else:
                supervised_loss = torch.tensor(0.0, device=device)
            
            pseudo_loss = torch.tensor(0.0, device=device)
            loss = supervised_loss
            
        else:
            # Default use pseudo label loss
            if labeled_data is not None:
                logits = model(labeled_data)
                supervised_loss = F.cross_entropy(logits, labeled_targets)
            else:
                supervised_loss = torch.tensor(0.0, device=device)
            
            pseudo_loss = torch.tensor(0.0, device=device)
            loss = supervised_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_supervised_loss += supervised_loss.item()
        total_pseudo_loss += pseudo_loss.item()
    
    return (total_loss / len(train_loader), 
            total_supervised_loss / len(train_loader), 
            total_pseudo_loss / len(train_loader))

def validate_epoch(model, val_loader, device, loss_function='pseudo_label', alpha=1.0, beta=0.5):
    """Validate one epoch"""
    model.eval()
    total_loss = 0
    total_supervised_loss = 0
    total_pseudo_loss = 0
    
    with torch.no_grad():
        for data, targets in val_loader:
            data = data.to(device)
            targets = targets.to(device)
            
            logits = model(data)
            
            # Calculate loss according to the loss function type
            if loss_function.lower() == 'pseudo_label':
                loss_fn = PseudoLabelLoss(alpha=alpha, beta=beta)
                loss, supervised_loss, pseudo_loss = loss_fn(logits, targets, None, None)
            elif loss_function.lower() == 'focal':
                focal_loss_fn = FocalLoss(alpha=1.0, gamma=2.0)
                supervised_loss = focal_loss_fn(logits, targets)
                pseudo_loss = torch.tensor(0.0, device=device)
                loss = supervised_loss
            elif loss_function.lower() == 'label_smoothing':
                label_smooth_loss_fn = LabelSmoothingLoss(num_classes=model.num_classes, smoothing=0.1)
                supervised_loss = label_smooth_loss_fn(logits, targets)
                pseudo_loss = torch.tensor(0.0, device=device)
                loss = supervised_loss
            else:
                supervised_loss = F.cross_entropy(logits, targets)
                pseudo_loss = torch.tensor(0.0, device=device)
                loss = supervised_loss
            
            total_loss += loss.item()
            total_supervised_loss += supervised_loss.item()
            total_pseudo_loss += pseudo_loss.item()
    
    return (total_loss / len(val_loader), 
            total_supervised_loss / len(val_loader), 
            total_pseudo_loss / len(val_loader))

def train_model(model, train_loader, val_loader, unlabeled_loader, optimizer, device, epochs, patience, model_name, 
                loss_function='pseudo_label', alpha=1.0, beta=0.5, confidence_threshold=0.8, pseudo_ratio=0.5):
    """Train model"""
    early_stopping = EarlyStopping(patience=patience)
    train_losses = []
    val_losses = []
    
    print(f"Start training model: {model_name}")
    print(f"Use loss function: {loss_function}")
    print(f"Supervised loss weight: {alpha}, Pseudo-label loss weight: {beta}")
    print(f"Confidence threshold: {confidence_threshold}, Pseudo-label ratio: {pseudo_ratio}")
    print("-" * 50)
    
    for epoch in range(epochs):
        train_loss, train_sup, train_pseudo = train_epoch(
            model, train_loader, unlabeled_loader, optimizer, device, loss_function, 
            alpha, beta, confidence_threshold, pseudo_ratio
        )
        val_loss, val_sup, val_pseudo = validate_epoch(
            model, val_loader, device, loss_function, alpha, beta
        )
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  train loss: {train_loss:.4f} (supervised: {train_sup:.4f}, pseudo-label: {train_pseudo:.4f})')
        print(f'  validation loss: {val_loss:.4f} (supervised: {val_sup:.4f}, pseudo-label: {val_pseudo:.4f})')
        
        if early_stopping(val_loss, model):
            print(f'Early stopping triggered, stopping training at epoch {epoch+1}')
            break
    
    return train_losses, val_losses

def test_model(model, test_loader, device, loss_function='pseudo_label', alpha=1.0, beta=0.5):
    """Test model"""
    model.eval()
    total_loss = 0
    total_supervised_loss = 0
    total_pseudo_loss = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)
            
            logits = model(data)
            
            # Calculate loss according to the loss function type
            if loss_function.lower() == 'pseudo_label':
                loss_fn = PseudoLabelLoss(alpha=alpha, beta=beta)
                loss, supervised_loss, pseudo_loss = loss_fn(logits, targets, None, None)
            elif loss_function.lower() == 'focal':
                focal_loss_fn = FocalLoss(alpha=1.0, gamma=2.0)
                supervised_loss = focal_loss_fn(logits, targets)
                pseudo_loss = torch.tensor(0.0, device=device)
                loss = supervised_loss
            elif loss_function.lower() == 'label_smoothing':
                label_smooth_loss_fn = LabelSmoothingLoss(num_classes=model.num_classes, smoothing=0.1)
                supervised_loss = label_smooth_loss_fn(logits, targets)
                pseudo_loss = torch.tensor(0.0, device=device)
                loss = supervised_loss
            else:
                supervised_loss = F.cross_entropy(logits, targets)
                pseudo_loss = torch.tensor(0.0, device=device)
                loss = supervised_loss
            
            total_loss += loss.item()
            total_supervised_loss += supervised_loss.item()
            total_pseudo_loss += pseudo_loss.item()
    
    avg_loss = total_loss / len(test_loader)
    avg_supervised_loss = total_supervised_loss / len(test_loader)
    avg_pseudo_loss = total_pseudo_loss / len(test_loader)
    
    print(f"Test results:")
    print(f"  Total loss: {avg_loss:.4f}")
    print(f"  Supervised loss: {avg_supervised_loss:.4f}")
    print(f"  Pseudo-label loss: {avg_pseudo_loss:.4f}")
    
    return avg_loss, avg_supervised_loss, avg_pseudo_loss

def kfold_cross_validation(args, k_folds=5):
    """k-fold cross-validation"""
    print(f"\nStart {k_folds}-fold cross-validation")
    print("=" * 60)
    
    input_dim, num_classes, train_loader, _, test_loader, scaler, unlabeled_loader = prepare_data(
        args.random_seed, args.batch_size, args.ratio, args.jobid, args.use_unlabeled, args.k_folds 
    )
    
    # Get training data for k-fold
    train_data_list = []
    train_labels_list = []
    for data, labels in train_loader:
        train_data_list.append(data.numpy())
        train_labels_list.append(labels.numpy())
    train_data_scaled = np.vstack(train_data_list)
    train_labels_scaled = np.concatenate(train_labels_list)
    
    # Perform k-fold cross-validation on training data
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=args.random_seed)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data_scaled)):
        print(f"\nFold {fold + 1}/{k_folds}")
        print("-" * 30)
        
        # Split data
        fold_train_data = torch.FloatTensor(train_data_scaled[train_idx])
        fold_train_labels = torch.LongTensor(train_labels_scaled[train_idx])
        fold_val_data = torch.FloatTensor(train_data_scaled[val_idx])
        fold_val_labels = torch.LongTensor(train_labels_scaled[val_idx])
        
        fold_train_loader = DataLoader(TensorDataset(fold_train_data, fold_train_labels), 
                                     batch_size=args.batch_size, shuffle=True)
        fold_val_loader = DataLoader(TensorDataset(fold_val_data, fold_val_labels), 
                                   batch_size=args.batch_size, shuffle=False)
        
        # Create model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = PseudoLabelNetwork(
            input_dim=input_dim,
            hidden_dims=[512, 256, 128],
            num_classes=num_classes,
            dropout_rate=args.dropout
        ).to(device)
        
        # Create optimizer
        if args.optimizer_function.lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        elif args.optimizer_function.lower() == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
        elif args.optimizer_function.lower() == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
        elif args.optimizer_function.lower() == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate)
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        
        # Train model
        train_losses, val_losses = train_model(
            model, fold_train_loader, fold_val_loader, unlabeled_loader, optimizer, device, 
            args.epochs, args.early_stopping_patience, f"Fold_{fold+1}", 
            args.loss_function, args.alpha, args.beta, args.confidence_threshold, args.pseudo_ratio
        )
        
        # Test model on validation set
        test_loss, test_sup, test_pseudo = test_model(
            model, fold_val_loader, device, args.loss_function, args.alpha, args.beta
        )
        fold_results.append({
            'fold': fold + 1,
            'test_loss': test_loss,
            'test_supervised_loss': test_sup,
            'test_pseudo_loss': test_pseudo,
            'train_losses': train_losses,
            'val_losses': val_losses
        })
    
    # Calculate average results
    avg_test_loss = np.mean([r['test_loss'] for r in fold_results])
    avg_test_sup = np.mean([r['test_supervised_loss'] for r in fold_results])
    avg_test_pseudo = np.mean([r['test_pseudo_loss'] for r in fold_results])
    
    print(f"\n{k_folds}-fold cross-validation results:")
    print("=" * 40)
    print(f"Average validation loss: {avg_test_loss:.4f} ± {np.std([r['test_loss'] for r in fold_results]):.4f}")
    print(f"Average supervised loss: {avg_test_sup:.4f} ± {np.std([r['test_supervised_loss'] for r in fold_results]):.4f}")
    print(f"Average pseudo-label loss: {avg_test_pseudo:.4f} ± {np.std([r['test_pseudo_loss'] for r in fold_results]):.4f}")
    
    return fold_results, model, scaler, test_loader, input_dim, num_classes

def save_model(model, scaler, args, model_path, scaler_path, input_dim, num_classes):
    """Save model and preprocessor"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': input_dim,
            'hidden_dims': [512, 256, 128],
            'num_classes': num_classes,
            'dropout_rate': args.dropout
        },
        'args': vars(args)
    }, model_path)
    
    # Save scaler and label_encoder
    import pickle
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

 

def load_model(model_path, scaler_path, device):
    """Load model and preprocessor"""
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['model_config']
    
    model = PseudoLabelNetwork(
        input_dim=model_config['input_dim'],
        hidden_dims=model_config['hidden_dims'],
        num_classes=model_config['num_classes'],
        dropout_rate=model_config['dropout_rate']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load scaler and label_encoder
    import pickle
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    

    
    return model, scaler

def predict(model, scaler, data, device):
    """Use trained model to predict"""
    model.eval()
    
    # Data preprocessing
    data_scaled = scaler.transform(data)
    data_tensor = torch.FloatTensor(data_scaled).to(device)
    
    with torch.no_grad():
        logits = model(data_tensor)
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1) 
    
    return (predictions.cpu().numpy(), 
            probabilities.cpu().numpy())

def main():
    parser = argparse.ArgumentParser(description='Pseudo label semi-supervised learning training script')

    # Data path
    parser.add_argument('--ratio', type=str, default='8:1:1', help='Data split ratio')
    parser.add_argument('--jobid', type=str, default='train_example_semi', help='Data set ID')
    
    # Model parameters
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience')
    
    # Loss function parameters
    parser.add_argument('--loss_function', type=str, default='pseudo_label', 
                       choices=['pseudo_label', 'focal', 'label_smoothing'], 
                       help='Loss function type')
    parser.add_argument('--alpha', type=float, default=1.0, help='Supervised loss weight')
    parser.add_argument('--beta', type=float, default=0.5, help='Pseudo label loss weight')
    parser.add_argument('--confidence_threshold', type=float, default=0.8, help='Pseudo label confidence threshold')
    parser.add_argument('--pseudo_ratio', type=float, default=0.5, help='Pseudo label ratio')
    
    # Optimizer
    parser.add_argument('--optimizer_function', type=str, default='adam', 
                       choices=['adam', 'sgd', 'adamw', 'rmsprop'], help='Optimizer type')
    
    # Semi-supervised learning parameters
    parser.add_argument('--use_unlabeled', action='store_true', default=1, help='Whether to use unlabeled data for semi-supervised learning')
    
    # Other parameters
    parser.add_argument('--random_seed', type=int, default=42, help='random seed')
    parser.add_argument('--k_folds', type=int, default=0, help='k-fold number')
    parser.add_argument('--save_model', action='store_true', default=1, help='Whether to save model')
    parser.add_argument('--model_path', type=str, default='pseudo_model.pth', help='Model save path')
    parser.add_argument('--scaler_path', type=str, default='pseudo_scaler.pkl', help='Preprocessor save path')
    
    # Evaluation related parameters
    parser.add_argument('--evaluate_model', action='store_true', default=1, help='Whether to evaluate model')
    parser.add_argument('--save_evaluation', action='store_true', default=1, help='Whether to save evaluation results')
    parser.add_argument('--evaluation_path', type=str, default='pseudo_evaluation_results.png', help='Evaluation results save path')
    parser.add_argument('--show_plots', action='store_true', default=1, help='Whether to show plots')
    
    args = parser.parse_args()

    print("Pseudo model")
    print('ratio =', args.ratio)
    print('k_folds =', args.k_folds)
    print('epochs =', args.epochs)
    print('batch_size =', args.batch_size)
    print('learning_rate =', args.learning_rate)
    print('early_stopping_patience =', args.early_stopping_patience)
    print('optimizer_function =', args.optimizer_function)
    print('loss_function =', args.loss_function)
    print('random_seed =', args.random_seed)
    print('alpha =', args.alpha)
    print('beta =', args.beta)
    print('confidence_threshold =', args.confidence_threshold)
    print('pseudo_ratio =', args.pseudo_ratio)
    

    # Set random seed
    set_random_seed(args.random_seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    if args.k_folds:
        # k-fold cross-validation
        fold_results, model, scaler, test_loader, input_dim, num_classes = kfold_cross_validation(args, args.k_folds)
    else:
        # Regular training-validation-test
        print("\nStart regular training process")
        print("=" * 50)
        
        # Prepare data
        input_dim, num_classes, train_loader, val_loader, test_loader, scaler, unlabeled_loader = prepare_data(
            args.random_seed, args.batch_size, args.ratio, args.jobid, args.use_unlabeled, args.k_folds
        )
        
        # Create model
        model = PseudoLabelNetwork(
            input_dim=input_dim,
            hidden_dims=[512, 256, 128],
            num_classes=num_classes,
            dropout_rate=args.dropout
        ).to(device)
        
        # Create optimizer
        if args.optimizer_function.lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        elif args.optimizer_function.lower() == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
        elif args.optimizer_function.lower() == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
        elif args.optimizer_function.lower() == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate)
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        
        # Train model
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, unlabeled_loader, optimizer, device, 
            args.epochs, args.early_stopping_patience, "Pseudo_Model", 
            loss_function=args.loss_function, alpha=args.alpha, beta=args.beta, 
            confidence_threshold=args.confidence_threshold, pseudo_ratio=args.pseudo_ratio
        )
        
        # Test model
        test_loss, test_sup, test_pseudo = test_model(
            model, test_loader, device, loss_function=args.loss_function, 
            alpha=args.alpha, beta=args.beta
        )
    
    # Save model
    if args.save_model:
        save_model(model, scaler, args, args.model_path, 
                  args.scaler_path, input_dim, num_classes)
    
    # Model evaluation
    if args.evaluate_model:
        print("\n" + "="*60)
        print("Start model evaluation")
        print("="*60)
        
        # Create evaluator
        evaluator = ModelEvaluator(device)
        
        # Perform comprehensive evaluation
        metrics = evaluator.compute_comprehensive_metrics(model, test_loader, scaler)
        
        # Print evaluation report
        print_evaluation_report(metrics)
        
        # Visualize results
        if args.show_plots:
            visualize_results(metrics, args.evaluation_path if args.save_evaluation else None)
        
        # Save evaluation results
        if args.save_evaluation:
            # Save evaluation metrics to JSON file
            evaluation_data = {
                'classification_metrics': {
                    'accuracy': float(metrics['classification']['accuracy']),
                    'precision': float(metrics['classification']['precision']),
                    'recall': float(metrics['classification']['recall']),
                    'f1_score': float(metrics['classification']['f1_score']),
                    # 'precision_per_class': metrics['classification']['precision_per_class'].tolist(),
                    # 'recall_per_class': metrics['classification']['recall_per_class'].tolist(),
                    # 'f1_per_class': metrics['classification']['f1_per_class'].tolist()
                }

            }
            
            json_path = args.evaluation_path.replace('.png', '.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # cmd: pythonpseudo.py --ratio 0 --jobid train_example_semi
    # cmd: pythonpseudo.py --ratio 8:1:1 --k_folds 5 --jobid train_example_semi
    # output: pseudo_evaluation_results.json, pseudo_evaluation_results.png, pseudo_model.pth, pseudo_scaler.pkl
    main()
