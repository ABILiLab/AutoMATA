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

class LadderNetwork(nn.Module):
    """
    Multi-classification tasks for semi-supervised learning
    """
    def __init__(self, input_dim, hidden_dims, num_classes, dropout_rate=0.1):
        super(LadderNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # encoder
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
        
        # decoder
        decoder_layers = []
        decoder_dims = list(reversed(hidden_dims))
        
        for i, hidden_dim in enumerate(decoder_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # final output layer
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # classifier
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)
        
        # noise layer (for semi-supervised learning)
        self.noise_layers = nn.ModuleList([
            nn.Linear(hidden_dims[i], hidden_dims[i]) for i in range(len(hidden_dims))
        ])
    
    def forward(self, x, add_noise=False):
        """forward propagation"""
        # encoder
        encoded = self.encoder(x)
        
        # add noise (if enabled)
        if add_noise:
            noise = torch.randn_like(encoded) * 0.1
            encoded = encoded + noise
        
        # classifier
        logits = self.classifier(encoded)
        
        # decoder
        decoded = self.decoder(encoded)
        
        return logits, decoded, [encoded]
    
    def decode(self, h, encoded_features):
        return self.decoder(h)

class SemiSupervisedLoss(nn.Module):
    """
    Semi-supervised learning loss function
    Combine supervised loss and reconstruction loss
    """
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.1, reduction='mean'):
        super(SemiSupervisedLoss, self).__init__()
        self.alpha = alpha  # supervised loss weight
        self.beta = beta    # reconstruction loss weight
        self.gamma = gamma  # regularization loss weight
        self.reduction = reduction
    
    def forward(self, logits, targets, reconstructed, original, encoded_features):
        # supervised loss (cross entropy) - only for labeled data
        if targets is not None:
            # supervised_loss = F.cross_entropy(logits, targets, reduction=self.reduction)
            # Separate labeled and unlabeled data
            labeled_mask = targets != -1  # -1 means unlabeled
            if labeled_mask.sum() > 0:  # there are labeled data
                labeled_logits = logits[labeled_mask]
                labeled_targets = targets[labeled_mask]
                supervised_loss = F.cross_entropy(labeled_logits, labeled_targets, reduction=self.reduction)
            else:
                supervised_loss = torch.tensor(0.0, device=logits.device)
        else:
            supervised_loss = torch.tensor(0.0, device=logits.device)
        
        # reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(reconstructed, original, reduction=self.reduction)
        
        # regularization loss (L2)
        l2_reg = 0
        for features in encoded_features:
            l2_reg += torch.norm(features, p=2)
        regularization_loss = l2_reg / len(encoded_features)
        
        total_loss = (self.alpha * supervised_loss + 
                     self.beta * reconstruction_loss + 
                     self.gamma * regularization_loss)
        
        return total_loss, supervised_loss, reconstruction_loss, regularization_loss

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Separate labeled and unlabeled data
        labeled_mask = targets != -1  # -1 means unlabeled
        if labeled_mask.sum() > 0:  # there are labeled data
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
    """
    Label Smoothing Loss
    """
    def __init__(self, num_classes, smoothing=0.1, reduction='mean'):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Separate labeled and unlabeled data
        labeled_mask = targets != -1  # -1 means unlabeled
        if labeled_mask.sum() > 0:  # there are labeled data
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
    """
    Contrastive Loss for semi-supervised learning
    """
    def __init__(self, margin=1.0, temperature=0.1, reduction='mean'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, features, labels):
        # Separate labeled and unlabeled data
        labeled_mask = labels != -1  # -1 means unlabeled
        if labeled_mask.sum() > 0:  # there are labeled data
            labeled_features = features[labeled_mask]
            labeled_labels = labels[labeled_mask]
            
            # calculate similarity matrix
            features_norm = F.normalize(labeled_features, p=2, dim=1)
            similarity_matrix = torch.mm(features_norm, features_norm.t()) / self.temperature
            
            # create positive sample mask
            labels_equal = labeled_labels.unsqueeze(0) == labeled_labels.unsqueeze(1)
            
            # calculate contrastive loss
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
    """
    Semi-supervised learning model evaluator
    """
    def __init__(self, device):
        self.device = device
        self.metrics = {}
    
    def evaluate_classification(self, model, data_loader, scaler):
        """
        Evaluate classification performance
        """
        model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, targets in data_loader:
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                logits, _, _ = model(data)
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        
        # calculate classification metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='weighted')
        recall = recall_score(all_targets, all_predictions, average='weighted')
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        
        # calculate each class metrics
        precision_per_class = precision_score(all_targets, all_predictions, average=None)
        recall_per_class = recall_score(all_targets, all_predictions, average=None)
        f1_per_class = f1_score(all_targets, all_predictions, average=None)
        
        # calculate confusion matrix
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
    
    def evaluate_reconstruction(self, model, data_loader, scaler):
        """
        Evaluate reconstruction performance
        """
        model.eval()
        all_original = []
        all_reconstructed = []
        
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.device)
                _, reconstructed, _ = model(data)
                
                all_original.append(data.cpu().numpy())
                all_reconstructed.append(reconstructed.cpu().numpy())
        
        original = np.vstack(all_original)
        reconstructed = np.vstack(all_reconstructed)
        
        # unstandardize
        original_unscaled = scaler.inverse_transform(original)
        reconstructed_unscaled = scaler.inverse_transform(reconstructed)
        
        # calculate reconstruction metrics
        mse = np.mean((original_unscaled - reconstructed_unscaled) ** 2)
        mae = np.mean(np.abs(original_unscaled - reconstructed_unscaled))
        
        # calculate correlation coefficient
        correlation = np.corrcoef(original_unscaled.flatten(), reconstructed_unscaled.flatten())[0, 1]
        
        reconstruction_metrics = {
            'mse': mse,
            'mae': mae,
            'correlation': correlation,
            'original_data': original_unscaled,
            'reconstructed_data': reconstructed_unscaled
        }
        
        return reconstruction_metrics
    
    def compute_comprehensive_metrics(self, model, data_loader, scaler):
        """
        Calculate comprehensive evaluation metrics
        """
        print("Start comprehensive model evaluation...")
        
        # evaluate classification performance
        print("Evaluate classification performance...")
        classification_metrics = self.evaluate_classification(model, data_loader, scaler)
        
        # evaluate reconstruction performance
        print("Evaluate reconstruction performance...")
        reconstruction_metrics = self.evaluate_reconstruction(model, data_loader, scaler)
                
        comprehensive_metrics = {
            'classification': classification_metrics,
            'reconstruction': reconstruction_metrics
        }
        
        return comprehensive_metrics
 
def visualize_results(metrics, save_path=None):
    """
    Visualize evaluation results
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Ladder network semi-supervised learning evaluation results', fontsize=16, fontweight='bold')
    
    # 1. confusion matrix
    ax1 = axes[0, 0]
    cm = metrics['classification']['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title(f'Confusion matrix (accuracy: {metrics["classification"]["accuracy"]:.3f})')
    ax1.set_xlabel('Predicted labels')
    ax1.set_ylabel('True labels')
    
    # 2. classification performance metrics
    ax2 = axes[0, 1]
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
    
    # add numerical labels
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. reconstruction quality scatter plot
    ax3 = axes[1, 0]
    original = metrics['reconstruction']['original_data'].flatten()
    reconstructed = metrics['reconstruction']['reconstructed_data'].flatten()

    if len(original) > 1000:
        indices = np.random.choice(len(original), 1000, replace=False)
        original_sample = original[indices]
        reconstructed_sample = reconstructed[indices]
    else:
        original_sample = original
        reconstructed_sample = reconstructed
    
    ax3.scatter(original_sample, reconstructed_sample, alpha=0.5, s=1)
    ax3.plot([original_sample.min(), original_sample.max()], 
             [original_sample.min(), original_sample.max()], 'r--', lw=2)
    ax3.set_xlabel('Original data')
    ax3.set_ylabel('Reconstructed data')
    ax3.set_title(f'Reconstruction quality (correlation: {metrics["reconstruction"]["correlation"]:.3f})')
    ax3.grid(True, alpha=0.3)
    

    # 4. reconstruction error distribution
    ax5 = axes[1, 1]
    reconstruction_error = np.abs(original - reconstructed)
    ax5.hist(reconstruction_error, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax5.set_xlabel('Reconstruction error')
    ax5.set_ylabel('Frequency')
    ax5.set_title(f'Reconstruction error distribution (MSE: {metrics["reconstruction"]["mse"]:.3f})')
    ax5.grid(True, alpha=0.3)
    
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # plt.show()
    plt.close()

def print_evaluation_report(metrics):
    print("\n" + "="*60)
    print("Ladder network semi-supervised learning evaluation report")
    print("="*60)
    
    # classification performance
    print("\n Classification performance evaluation:")
    print("-" * 30)
    print(f"Accuracy: {metrics['classification']['accuracy']:.6f}")
    print(f"Precision: {metrics['classification']['precision']:.6f}")
    print(f"Recall: {metrics['classification']['recall']:.6f}")
    print(f"F1 Score: {metrics['classification']['f1_score']:.6f}")
    
    print("\n Detailed metrics for each class:")
    print("-" * 30)
    for i, (precision, recall, f1) in enumerate(zip(
        metrics['classification']['precision_per_class'],
        metrics['classification']['recall_per_class'],
        metrics['classification']['f1_per_class']
    )):
        print(f"Class {i}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    
    # reconstruction performance
    print("\n Reconstruction performance evaluation:")
    print("-" * 30)
    print(f"MSE: {metrics['reconstruction']['mse']:.6f}")
    print(f"MAE: {metrics['reconstruction']['mae']:.6f}")
    print(f"Correlation: {metrics['reconstruction']['correlation']:.6f}")
    
    

class EarlyStopping:
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

def prepare_data(seed, batch_size, ratio="8:1:1", jobid="20240808232043_OtJF37SH", use_unlabeled=True, kfold=0):
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
        
        # Create unlabeled data loader (for reconstruction task)
        unlabeled_data_tensor = torch.FloatTensor(unlabeled_features_scaled)
        unlabeled_loader = DataLoader(TensorDataset(unlabeled_data_tensor), 
                                    batch_size=batch_size, shuffle=True)
        
        # Merge labeled and unlabeled data for training
        if use_unlabeled and len(unlabeled_features) > 0:
            # Create semi-supervised training data loader
            all_features = np.vstack([labeled_features_scaled, unlabeled_features_scaled])
            all_labels = np.concatenate([labeled_labels_encoded, 
                                       np.full(len(unlabeled_features), -1)])  # -1 indicates no label
            
            all_data_tensor = torch.FloatTensor(all_features)
            all_labels_tensor = torch.LongTensor(all_labels)
            train_loader = DataLoader(TensorDataset(all_data_tensor, all_labels_tensor), 
                                    batch_size=batch_size, shuffle=True)
        else:
            train_loader = labeled_loader
        
        # Validation and test data
        # val_data = pd.read_csv(f"./train_example/{jobid}_val.txt", sep="\t")
        val_data = pd.read_csv("../../data/train_example_semi/train_example_semi_val.txt", sep="\t")
        val_data = val_data.dropna()
        val_features = val_data.iloc[:, 1:-1].values.astype(float)
        val_labels = val_data.iloc[:, -1].values
        
        # Process validation set labels - ensure label format consistency
        try:
            # Convert all labels to strings first to avoid type issues
            val_labels_str = val_labels.astype(str)
            # Try using label_encoder to convert
            val_labels_encoded = label_encoder.transform(val_labels_str)
        except ValueError as e:
            # If conversion fails, it means the label format does not match
            # Refit label_encoder to include all labels
            all_labels = np.concatenate([labeled_labels, val_labels_str])
            label_encoder.fit(all_labels)
            val_labels_encoded = label_encoder.transform(val_labels_str)
        
        val_features_scaled = scaler.transform(val_features)
        val_data_tensor = torch.FloatTensor(val_features_scaled)
        val_labels_tensor = torch.LongTensor(val_labels_encoded)
        val_loader = DataLoader(TensorDataset(val_data_tensor, val_labels_tensor), 
                              batch_size=batch_size, shuffle=False)

        # test_data = pd.read_csv(f"./train_example/{jobid}_test.txt", sep="\t")
        test_data = pd.read_csv("../../data/train_example_semi/train_example_semi_test.txt", sep="\t")
        test_data = test_data.dropna()
        test_features = test_data.iloc[:, 1:-1].values.astype(float)
        test_labels = test_data.iloc[:, -1].values
        
        # Process test set labels - ensure label format consistency
        try:
            # Convert all labels to strings first to avoid type issues
            test_labels_str = test_labels.astype(str)
            # Try using label_encoder to convert
            test_labels_encoded = label_encoder.transform(test_labels_str)
        except ValueError as e:
            # If conversion fails, it means the label format does not match
            # Refit label_encoder to include all labels
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
        # print("kfold")
        # Split training-validation-test set
        # data = pd.read_csv(f"./train_example/{jobid}_data.txt", sep="\t")
        data = pd.read_csv("../../data/train_example_semi/train_example_semi_data.txt", sep="\t")
        data = data.dropna()
        # sample_names = data.iloc[:, 0].values  # the first column is sample name
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
        
        # Split labeled data
        ratio_str = ratio.split(":")
        ratio_num = list(map(int, ratio_str))
        train_ratio = ratio_num[0] / sum(ratio_num)
        test_ratio = ratio_num[2] / sum(ratio_num[1:])
        
        # Check if the data amount is enough for stratified sampling
        unique_labels, label_counts = np.unique(labeled_labels_encoded, return_counts=True)
        min_class_count = np.min(label_counts)
        
        if min_class_count < 2:
            print(f"Warning: If the sample size of certain categories is too small (at least {min_class_count}), random segmentation will be used instead of stratified sampling")  
            stratify_train = None
            stratify_val = None
        else:
            stratify_train = labeled_labels_encoded
            stratify_val = None
        
        train_features, res_features, train_labels, res_labels = train_test_split(
            labeled_features_scaled, labeled_labels_encoded, test_size=1-train_ratio, 
            random_state=seed, stratify=stratify_train
        )
        
        # Check if stratified sampling is needed when splitting the validation set
        if len(res_labels) > 0:
            unique_res_labels, res_label_counts = np.unique(res_labels, return_counts=True)
            min_res_class_count = np.min(res_label_counts)
            
            if min_res_class_count < 2:
                print(f"Warning: If the sample size of certain categories is too small (at least {min_res_class_count}) during validation set segmentation, random segmentation will be used")
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
            # Add unlabeled data to training set
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
        
        input_dim = features.shape[1]

    else: 
        # kfold training
        scaler = StandardScaler()
        label_encoder = LabelEncoder()
        
        # Directly load training-validation-test set file
        # train_data = pd.read_csv(f"./train_example/{jobid}_data.txt", sep="\t")
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
        
        # Create unlabeled data loader (for reconstruction task)
        unlabeled_data_tensor = torch.FloatTensor(unlabeled_features_scaled)
        unlabeled_loader = DataLoader(TensorDataset(unlabeled_data_tensor), 
                                    batch_size=batch_size, shuffle=True)
        
        # Merge labeled and unlabeled data for training
        if use_unlabeled and len(unlabeled_features) > 0:
            # Create semi-supervised training data loader
            all_features = np.vstack([labeled_features_scaled, unlabeled_features_scaled])
            all_labels = np.concatenate([labeled_labels_encoded, 
                                       np.full(len(unlabeled_features), -1)])
            
            all_data_tensor = torch.FloatTensor(all_features)
            all_labels_tensor = torch.LongTensor(all_labels)
            train_loader = DataLoader(TensorDataset(all_data_tensor, all_labels_tensor), 
                                    batch_size=batch_size, shuffle=True)
        else:
            train_loader = labeled_loader
        val_loader = 0
        # test_data = pd.read_csv(f"./train_example/{jobid}_test.txt", sep="\t")
        test_data = pd.read_csv("../../data/train_example_semi/train_example_semi_test.txt", sep="\t")
        test_data = test_data.dropna()
        test_features = test_data.iloc[:, 1:-1].values.astype(float)
        test_labels = test_data.iloc[:, -1].values
        
        # Process test set labels - ensure label format consistency
        try:
            # Convert all labels to strings first to avoid type issues
            test_labels_str = test_labels.astype(str)
            # Try using label_encoder to convert
            test_labels_encoded = label_encoder.transform(test_labels_str)
        except ValueError as e:
            # If conversion fails, it means the label format does not match
            # Refit label_encoder to include all labels
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
    
    return input_dim, num_classes, train_loader, val_loader, test_loader, scaler

def train_epoch(model, train_loader, optimizer, device, loss_function='semi_supervised', 
                alpha=1.0, beta=1.0, gamma=0.1, add_noise=False):
    """Train one epoch"""
    model.train()
    total_loss = 0
    total_supervised_loss = 0
    total_reconstruction_loss = 0
    total_regularization_loss = 0
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        
        logits, reconstructed, encoded_features = model(data, add_noise=add_noise)
        
        # Separate labeled and unlabeled data
        labeled_mask = targets != -1  # -1 means unlabeled
        unlabeled_mask = targets == -1
        
        if labeled_mask.sum() > 0:  # there are labeled data
            labeled_data = data[labeled_mask]
            labeled_targets = targets[labeled_mask]
            labeled_logits = logits[labeled_mask]
            labeled_reconstructed = reconstructed[labeled_mask]
            labeled_encoded_features = [feat[labeled_mask] for feat in encoded_features]
        else:
            labeled_data = None
            labeled_targets = None
            labeled_logits = None
            labeled_reconstructed = None
            labeled_encoded_features = None
        
        if unlabeled_mask.sum() > 0:  # there are unlabeled data
            unlabeled_data = data[unlabeled_mask]
            unlabeled_reconstructed = reconstructed[unlabeled_mask]
            unlabeled_encoded_features = [feat[unlabeled_mask] for feat in encoded_features]
        else:
            unlabeled_data = None
            unlabeled_reconstructed = None
            unlabeled_encoded_features = None
        
        # Calculate loss
        if loss_function.lower() == 'semi_supervised':
            # Semi-supervised loss: calculate supervised loss for labeled data, calculate reconstruction loss for all data
            supervised_loss = torch.tensor(0.0, device=device)
            if labeled_data is not None:
                supervised_loss = F.cross_entropy(labeled_logits, labeled_targets)
            
            reconstruction_loss = F.mse_loss(reconstructed, data)
            
            regularization_loss = sum(torch.norm(feat, p=2) for feat in encoded_features) / len(encoded_features)
            
            loss = alpha * supervised_loss + beta * reconstruction_loss + gamma * regularization_loss
            
        elif loss_function.lower() == 'focal':
            supervised_loss = torch.tensor(0.0, device=device)
            if labeled_data is not None:
                focal_loss_fn = FocalLoss(alpha=1.0, gamma=2.0)
                supervised_loss = focal_loss_fn(labeled_logits, labeled_targets)
            
            reconstruction_loss = F.mse_loss(reconstructed, data)
            regularization_loss = sum(torch.norm(feat, p=2) for feat in encoded_features) / len(encoded_features)
            loss = alpha * supervised_loss + beta * reconstruction_loss + gamma * regularization_loss
            
        elif loss_function.lower() == 'label_smoothing':
            supervised_loss = torch.tensor(0.0, device=device)
            if labeled_data is not None:
                label_smooth_loss_fn = LabelSmoothingLoss(num_classes=model.num_classes, smoothing=0.1)
                supervised_loss = label_smooth_loss_fn(labeled_logits, labeled_targets)
            
            reconstruction_loss = F.mse_loss(reconstructed, data)
            regularization_loss = sum(torch.norm(feat, p=2) for feat in encoded_features) / len(encoded_features)
            loss = alpha * supervised_loss + beta * reconstruction_loss + gamma * regularization_loss
            
        elif loss_function.lower() == 'contrastive':
            supervised_loss = torch.tensor(0.0, device=device)
            if labeled_data is not None:
                supervised_loss = F.cross_entropy(labeled_logits, labeled_targets)
                contrastive_loss_fn = ContrastiveLoss()
                contrastive_loss = contrastive_loss_fn(labeled_encoded_features[-1], labeled_targets)
            else:
                contrastive_loss = torch.tensor(0.0, device=device)
            
            reconstruction_loss = F.mse_loss(reconstructed, data)
            regularization_loss = sum(torch.norm(feat, p=2) for feat in encoded_features) / len(encoded_features)
            loss = alpha * supervised_loss + beta * reconstruction_loss + gamma * contrastive_loss + 0.1 * regularization_loss
            
        else:
            # Default use semi-supervised loss
            supervised_loss = torch.tensor(0.0, device=device)
            if labeled_data is not None:
                supervised_loss = F.cross_entropy(labeled_logits, labeled_targets)
            
            reconstruction_loss = F.mse_loss(reconstructed, data)
            regularization_loss = sum(torch.norm(feat, p=2) for feat in encoded_features) / len(encoded_features)
            loss = alpha * supervised_loss + beta * reconstruction_loss + gamma * regularization_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_supervised_loss += supervised_loss.item()
        total_reconstruction_loss += reconstruction_loss.item()
        total_regularization_loss += regularization_loss.item()
    
    return (total_loss / len(train_loader), 
            total_supervised_loss / len(train_loader), 
            total_reconstruction_loss / len(train_loader), 
            total_regularization_loss / len(train_loader))

def validate_epoch(model, val_loader, device, loss_function='semi_supervised', 
                  alpha=1.0, beta=1.0, gamma=0.1):
    """Validate one epoch"""
    model.eval()
    total_loss = 0
    total_supervised_loss = 0
    total_reconstruction_loss = 0
    total_regularization_loss = 0
    
    with torch.no_grad():
        for data, targets in val_loader:
            data = data.to(device)
            targets = targets.to(device)
            
            logits, reconstructed, encoded_features = model(data)
            
            # Calculate loss according to the loss function type
            if loss_function.lower() == 'semi_supervised':
                loss_fn = SemiSupervisedLoss(alpha=alpha, beta=beta, gamma=gamma)
                loss, supervised_loss, reconstruction_loss, regularization_loss = loss_fn(
                    logits, targets, reconstructed, data, encoded_features
                )
            elif loss_function.lower() == 'focal':
                focal_loss_fn = FocalLoss(alpha=1.0, gamma=2.0)
                supervised_loss = focal_loss_fn(logits, targets)
                reconstruction_loss = F.mse_loss(reconstructed, data)
                regularization_loss = sum(torch.norm(feat, p=2) for feat in encoded_features) / len(encoded_features)
                loss = alpha * supervised_loss + beta * reconstruction_loss + gamma * regularization_loss
            elif loss_function.lower() == 'label_smoothing':
                label_smooth_loss_fn = LabelSmoothingLoss(num_classes=model.num_classes, smoothing=0.1)
                supervised_loss = label_smooth_loss_fn(logits, targets)
                reconstruction_loss = F.mse_loss(reconstructed, data)
                regularization_loss = sum(torch.norm(feat, p=2) for feat in encoded_features) / len(encoded_features)
                loss = alpha * supervised_loss + beta * reconstruction_loss + gamma * regularization_loss
            elif loss_function.lower() == 'contrastive':
                contrastive_loss_fn = ContrastiveLoss()
                supervised_loss = F.cross_entropy(logits, targets)
                reconstruction_loss = F.mse_loss(reconstructed, data)
                contrastive_loss = contrastive_loss_fn(encoded_features[-1], targets)
                regularization_loss = sum(torch.norm(feat, p=2) for feat in encoded_features) / len(encoded_features)
                loss = alpha * supervised_loss + beta * reconstruction_loss + gamma * contrastive_loss + 0.1 * regularization_loss
            else:
                loss_fn = SemiSupervisedLoss(alpha=alpha, beta=beta, gamma=gamma)
                loss, supervised_loss, reconstruction_loss, regularization_loss = loss_fn(
                    logits, targets, reconstructed, data, encoded_features
                )
            
            total_loss += loss.item()
            total_supervised_loss += supervised_loss.item()
            total_reconstruction_loss += reconstruction_loss.item()
            total_regularization_loss += regularization_loss.item()
    
    return (total_loss / len(val_loader), 
            total_supervised_loss / len(val_loader), 
            total_reconstruction_loss / len(val_loader), 
            total_regularization_loss / len(val_loader))

def train_model(model, train_loader, val_loader, optimizer, device, epochs, patience, model_name, 
                loss_function='semi_supervised', alpha=1.0, beta=1.0, gamma=0.1, add_noise=False):
    """训练模型"""
    early_stopping = EarlyStopping(patience=patience)
    train_losses = []
    val_losses = []
    
    print(f"Start training model: {model_name}")
    print(f"Use loss function: {loss_function}")
    print(f"Supervised loss weight: {alpha}, Reconstruction loss weight: {beta}, Regularization weight: {gamma}")
    print("-" * 50)
    
    for epoch in range(epochs):
        train_loss, train_sup, train_recon, train_reg = train_epoch(
            model, train_loader, optimizer, device, loss_function, alpha, beta, gamma, add_noise
        )
        val_loss, val_sup, val_recon, val_reg = validate_epoch(
            model, val_loader, device, loss_function, alpha, beta, gamma
        )
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  train loss: {train_loss:.4f} (supervised: {train_sup:.4f}, reconstruction: {train_recon:.4f}, regularization: {train_reg:.4f})')
        print(f'  validation loss: {val_loss:.4f} (supervised: {val_sup:.4f}, reconstruction: {val_recon:.4f}, regularization: {val_reg:.4f})')
        
        if early_stopping(val_loss, model):
            print(f'Early stopping triggered, stopping training at epoch {epoch+1}')
            break
    
    return train_losses, val_losses

def test_model(model, test_loader, device, loss_function='semi_supervised', 
               alpha=1.0, beta=1.0, gamma=0.1):
    """Test model"""
    model.eval()
    total_loss = 0
    total_supervised_loss = 0
    total_reconstruction_loss = 0
    total_regularization_loss = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)
            
            logits, reconstructed, encoded_features = model(data)
            
            # Calculate loss according to the loss function type
            if loss_function.lower() == 'semi_supervised':
                loss_fn = SemiSupervisedLoss(alpha=alpha, beta=beta, gamma=gamma)
                loss, supervised_loss, reconstruction_loss, regularization_loss = loss_fn(
                    logits, targets, reconstructed, data, encoded_features
                )
            elif loss_function.lower() == 'focal':
                focal_loss_fn = FocalLoss(alpha=1.0, gamma=2.0)
                supervised_loss = focal_loss_fn(logits, targets)
                reconstruction_loss = F.mse_loss(reconstructed, data)
                regularization_loss = sum(torch.norm(feat, p=2) for feat in encoded_features) / len(encoded_features)
                loss = alpha * supervised_loss + beta * reconstruction_loss + gamma * regularization_loss
            elif loss_function.lower() == 'label_smoothing':
                label_smooth_loss_fn = LabelSmoothingLoss(num_classes=model.num_classes, smoothing=0.1)
                supervised_loss = label_smooth_loss_fn(logits, targets)
                reconstruction_loss = F.mse_loss(reconstructed, data)
                regularization_loss = sum(torch.norm(feat, p=2) for feat in encoded_features) / len(encoded_features)
                loss = alpha * supervised_loss + beta * reconstruction_loss + gamma * regularization_loss
            elif loss_function.lower() == 'contrastive':
                contrastive_loss_fn = ContrastiveLoss()
                supervised_loss = F.cross_entropy(logits, targets)
                reconstruction_loss = F.mse_loss(reconstructed, data)
                contrastive_loss = contrastive_loss_fn(encoded_features[-1], targets)
                regularization_loss = sum(torch.norm(feat, p=2) for feat in encoded_features) / len(encoded_features)
                loss = alpha * supervised_loss + beta * reconstruction_loss + gamma * contrastive_loss + 0.1 * regularization_loss
            else:
                loss_fn = SemiSupervisedLoss(alpha=alpha, beta=beta, gamma=gamma)
                loss, supervised_loss, reconstruction_loss, regularization_loss = loss_fn(
                    logits, targets, reconstructed, data, encoded_features
                )
            
            total_loss += loss.item()
            total_supervised_loss += supervised_loss.item()
            total_reconstruction_loss += reconstruction_loss.item()
            total_regularization_loss += regularization_loss.item()
    
    avg_loss = total_loss / len(test_loader)
    avg_supervised_loss = total_supervised_loss / len(test_loader)
    avg_reconstruction_loss = total_reconstruction_loss / len(test_loader)
    avg_regularization_loss = total_regularization_loss / len(test_loader)
    
    print(f"Test results:")
    print(f"  Total loss: {avg_loss:.4f}")
    print(f"  Supervised loss: {avg_supervised_loss:.4f}")
    print(f"  Reconstruction loss: {avg_reconstruction_loss:.4f}")
    print(f"  Regularization loss: {avg_regularization_loss:.4f}")
    
    return avg_loss, avg_supervised_loss, avg_reconstruction_loss, avg_regularization_loss

def kfold_cross_validation(args, k_folds=5):
    """k-fold cross-validation"""
    print(f"\nStart {k_folds}-fold cross-validation")
    print("=" * 60)
    
    input_dim, num_classes, train_loader, _, test_loader, scaler = prepare_data(
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
        model = LadderNetwork(
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
            model, fold_train_loader, fold_val_loader, optimizer, device, 
            args.epochs, args.early_stopping_patience, f"Fold_{fold+1}", 
            args.loss_function, args.alpha, args.beta, args.gamma, args.add_noise
        )
        
        # Test model on validation set
        test_loss, test_sup, test_recon, test_reg = test_model(
            model, fold_val_loader, device, args.loss_function, args.alpha, args.beta, args.gamma
        )
        fold_results.append({
            'fold': fold + 1,
            'test_loss': test_loss,
            'test_supervised_loss': test_sup,
            'test_reconstruction_loss': test_recon,
            'test_regularization_loss': test_reg,
            'train_losses': train_losses,
            'val_losses': val_losses
        })
    
    # Calculate average result
    avg_test_loss = np.mean([r['test_loss'] for r in fold_results])
    avg_test_sup = np.mean([r['test_supervised_loss'] for r in fold_results])
    avg_test_recon = np.mean([r['test_reconstruction_loss'] for r in fold_results])
    avg_test_reg = np.mean([r['test_regularization_loss'] for r in fold_results])
    
    print(f"\n{k_folds}-fold cross-validation results:")
    print("=" * 40)
    print(f"Validation loss: {avg_test_loss:.4f} ± {np.std([r['test_loss'] for r in fold_results]):.4f}")
    print(f"Supervised loss: {avg_test_sup:.4f} ± {np.std([r['test_supervised_loss'] for r in fold_results]):.4f}")
    print(f"Reconstruction loss: {avg_test_recon:.4f} ± {np.std([r['test_reconstruction_loss'] for r in fold_results]):.4f}")
    print(f"Regularization loss: {avg_test_reg:.4f} ± {np.std([r['test_regularization_loss'] for r in fold_results]):.4f}")
    
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
    
    model = LadderNetwork(
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
        logits, reconstructed, encoded_features = model(data_tensor)
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)
    
    return (predictions.cpu().numpy(), 
            probabilities.cpu().numpy(), 
            reconstructed.cpu().numpy(), 
            [feat.cpu().numpy() for feat in encoded_features])

def main():
    parser = argparse.ArgumentParser(description='Ladder network semi-supervised learning training script')

    # Data path
    parser.add_argument('--ratio', type=str, default='0', help='Data split ratio')
    parser.add_argument('--jobid', type=str, default='train_example_semi', help='Data set ID')
    
    # Model parameters
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience')
    
    # Loss function parameters
    parser.add_argument('--loss_function', type=str, default='semi_supervised', 
                       choices=['semi_supervised', 'focal', 'label_smoothing', 'contrastive'], 
                       help='Loss function type:\n'
                            'semi_supervised: Semi-supervised learning loss\n'
                            'focal: Focal loss\n'
                            'label_smoothing: Label smoothing loss\n'
                            'contrastive: Contrastive loss')
    parser.add_argument('--alpha', type=float, default=1.0, help='Supervised loss weight')
    parser.add_argument('--beta', type=float, default=1.0, help='Reconstruction loss weight')
    parser.add_argument('--gamma', type=float, default=0.1, help='Regularization loss weight')
    parser.add_argument('--add_noise', action='store_true', default=0, help='Whether to add noise')  # Train model
    
    # Optimizer
    parser.add_argument('--optimizer_function', type=str, default='adam', 
                       choices=['adam', 'sgd', 'adamw', 'rmsprop'], help='Optimizer type')
    
    # Semi-supervised learning parameters
    parser.add_argument('--use_unlabeled', action='store_true', default=1, help='Whether to use unlabeled data for semi-supervised learning')
    
    # Other parameters
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--k_folds', type=int, default=0, help='k-fold number')
    parser.add_argument('--save_model', action='store_true', default=1, help='Whether to save model')
    parser.add_argument('--model_path', type=str, default='ladder_model.pth', help='Model save path')
    parser.add_argument('--scaler_path', type=str, default='ladder_scaler.pkl', help='Preprocessor save path')
    
    # Evaluation related parameters
    parser.add_argument('--evaluate_model', action='store_true', default=1, help='Whether to evaluate model')
    parser.add_argument('--save_evaluation', action='store_true', default=1, help='Whether to save evaluation results')
    parser.add_argument('--evaluation_path', type=str, default='ladder_evaluation_results.png', help='Evaluation results save path')
    parser.add_argument('--show_plots', action='store_true', default=1, help='Whether to show plots')
    
    args = parser.parse_args()
    
    print("Ladder model")
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
    print('gamma =', args.gamma)



    # Set random seed
    set_random_seed(args.random_seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.k_folds:
        # k-fold cross-validation
        fold_results, model, scaler, test_loader, input_dim, num_classes = kfold_cross_validation(args, args.k_folds)
    else:
        # Regular training-validation-test
        print("=" * 50)
        
        # Prepare data
        input_dim, num_classes, train_loader, val_loader, test_loader, scaler = prepare_data(
            args.random_seed, args.batch_size, args.ratio, args.jobid, args.use_unlabeled, args.k_folds
        )
        
        # Create model
        model = LadderNetwork(
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
            model, train_loader, val_loader, optimizer, device, 
            args.epochs, args.early_stopping_patience, "Ladder_Model", 
            loss_function=args.loss_function, alpha=args.alpha, beta=args.beta, 
            gamma=args.gamma, add_noise=args.add_noise
        )
        
        # Test model
        test_loss, test_sup, test_recon, test_reg = test_model(
            model, test_loader, device, loss_function=args.loss_function, 
            alpha=args.alpha, beta=args.beta, gamma=args.gamma
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
                    'precision_per_class': metrics['classification']['precision_per_class'].tolist(),
                    'recall_per_class': metrics['classification']['recall_per_class'].tolist(),
                    'f1_per_class': metrics['classification']['f1_per_class'].tolist()
                },
                'reconstruction_metrics': {
                    'mse': float(metrics['reconstruction']['mse']),
                    'mae': float(metrics['reconstruction']['mae']),
                    'correlation': float(metrics['reconstruction']['correlation'])
                }
            }
            
            json_path = args.evaluation_path.replace('.png', '.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # cmd: python ladder.py --ratio 0 --jobid train_example_semi
    # cmd: python ladder.py --ratio 8:1:1 --k_folds 5 --jobid train_example_semi
    # output: ladder_evaluation_results.json, ladder_evaluation_results.png, ladder_model.pth, ladder_scaler.pkl
    main()
