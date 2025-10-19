import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

import warnings
warnings.filterwarnings('ignore')

class DeepClusterLoss(nn.Module):

    def __init__(self, alpha=1.0, beta=1.0, reduction='sum'):
        super(DeepClusterLoss, self).__init__()
        self.alpha = alpha  # Reconstruct the loss weight
        self.beta = beta    # Cluster loss weight
        self.reduction = reduction
    
    def forward(self, recon_x, x, cluster_assignments, cluster_centers):
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='none')
        if self.reduction == 'sum':
            recon_loss = recon_loss.sum()
        else:
            recon_loss = recon_loss.mean()
        
        # Clustering loss (K-means target function)
        # Calculate the distance of each sample to its assigned cluster center
        cluster_loss = 0
        for i, center in enumerate(cluster_centers):
            mask = (cluster_assignments == i)
            if mask.sum() > 0:
                samples_in_cluster = x[mask]
                distances = torch.sum((samples_in_cluster - center) ** 2, dim=1)
                cluster_loss += distances.sum()
        
        if self.reduction == 'sum':
            cluster_loss = cluster_loss
        else:
            cluster_loss = cluster_loss / x.size(0)
        
        total_loss = self.alpha * recon_loss + self.beta * cluster_loss
        
        return total_loss, recon_loss, cluster_loss

class TripletLoss(nn.Module):
    """
    Triplet loss function
    Used to learn better cluster representations
    """
    def __init__(self, margin=1.0, reduction='sum'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        if self.reduction == 'sum':
            return loss.sum()
        else:
            return loss.mean()

class CenterLoss(nn.Module):
    """
    Center loss function
    Force samples to be close to their cluster centers
    """
    def __init__(self, num_classes, feat_dim, use_gpu=True, reduction='sum'):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.reduction = reduction
        
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
    
    def forward(self, x, labels):
        batch_size = x.size(0)
        centers_batch = self.centers[labels]
        
        loss = F.mse_loss(x, centers_batch, reduction='none')
        
        if self.reduction == 'sum':
            return loss.sum()
        else:
            return loss.mean()

class ContrastiveClusteringLoss(nn.Module):
    """
    Contrastive clustering loss function
    Combine contrastive learning and clustering
    """
    def __init__(self, temperature=0.1, reduction='sum'):
        super(ContrastiveClusteringLoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, features, cluster_assignments):
        # Calculate the similarity matrix
        features_norm = F.normalize(features, p=2, dim=1)
        similarity_matrix = torch.mm(features_norm, features_norm.t()) / self.temperature
        
        #  Create a positive sample mask (same cluster)
        cluster_mask = cluster_assignments.unsqueeze(0) == cluster_assignments.unsqueeze(1)
        
        # Calculate the contrastive loss
        exp_sim = torch.exp(similarity_matrix)
        
        # Positive sample similarity
        pos_sim = exp_sim * cluster_mask.float()
        
        # Negative sample similarity
        neg_sim = exp_sim * (1 - cluster_mask.float())
        
        # Avoid division by zero
        pos_sum = pos_sim.sum(dim=1, keepdim=True)
        neg_sum = neg_sim.sum(dim=1, keepdim=True)
        
        # Calculate the loss
        loss = -torch.log(pos_sum / (pos_sum + neg_sum + 1e-8))
        
        if self.reduction == 'sum':
            return loss.sum()
        else:
            return loss.mean()

class SpectralClusteringLoss(nn.Module):
    """
    Spectral clustering loss function
    Based on the Laplacian matrix of clustering loss
    """
    def __init__(self, reduction='sum'):
        super(SpectralClusteringLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, features, cluster_assignments):
        """
        features: feature vector
        cluster_assignments: cluster assignment
        """
        # Calculate the similarity matrix
        similarity = torch.mm(features, features.t())
        
        # Create a cluster indicator matrix
        n_samples = features.size(0)
        n_clusters = len(torch.unique(cluster_assignments))
        
        # Create a cluster indicator matrix
        cluster_indicator = torch.zeros(n_samples, n_clusters)
        for i, cluster_id in enumerate(cluster_assignments):
            cluster_indicator[i, cluster_id] = 1
        
        # Calculate the Laplacian matrix
        degree = torch.diag(torch.sum(similarity, dim=1))
        laplacian = degree - similarity
        
        # Calculate the spectral clustering loss
        loss = torch.trace(torch.mm(torch.mm(cluster_indicator.t(), laplacian), cluster_indicator))
        
        if self.reduction == 'sum':
            return loss
        else:
            return loss / n_samples

class EntropyLoss(nn.Module):
    """
    Entropy loss function
    Encourage the uniformity of cluster assignments
    """
    def __init__(self, reduction='sum'):
        super(EntropyLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, cluster_assignments, n_clusters):
        """
        cluster_assignments: cluster assignment
        n_clusters: cluster number
        """
        # Calculate the number of samples in each cluster
        cluster_counts = torch.bincount(cluster_assignments, minlength=n_clusters).float()
        
        # Calculate the probability distribution
        probabilities = cluster_counts / cluster_counts.sum()
        
        # Calculate the entropy
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8))
        
        if self.reduction == 'sum':
            return entropy
        else:
            return entropy / n_clusters

class CompactnessLoss(nn.Module):
    """
    Compactness loss function
    Encourage the samples in the cluster to be tightly clustered
    """
    def __init__(self, reduction='sum'):
        super(CompactnessLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, features, cluster_assignments):
        """
        features: feature vector
        cluster_assignments: cluster assignment
        """
        loss = 0
        n_clusters = len(torch.unique(cluster_assignments))
        
        for i in range(n_clusters):
            mask = (cluster_assignments == i)
            if mask.sum() > 1:
                cluster_features = features[mask]
                cluster_center = cluster_features.mean(dim=0)
                distances = torch.sum((cluster_features - cluster_center) ** 2, dim=1)
                loss += distances.sum()
        
        if self.reduction == 'sum':
            return loss
        else:
            return loss / features.size(0)

class SeparationLoss(nn.Module):
    """
    Separation loss function
    Encourage the distance between different cluster centers to be maximized
    """
    def __init__(self, reduction='sum'):
        super(SeparationLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, features, cluster_assignments):
        n_clusters = len(torch.unique(cluster_assignments))
        cluster_centers = []
        
        # Calculate the center of each cluster
        for i in range(n_clusters):
            mask = (cluster_assignments == i)
            if mask.sum() > 0:
                cluster_center = features[mask].mean(dim=0)
                cluster_centers.append(cluster_center)
        
        if len(cluster_centers) < 2:
            return torch.tensor(0.0, device=features.device)
        
        # Calculate the distance between cluster centers
        total_distance = 0
        for i in range(len(cluster_centers)):
            for j in range(i + 1, len(cluster_centers)):
                distance = torch.norm(cluster_centers[i] - cluster_centers[j], p=2)
                total_distance += distance
        
        # Separation loss (negative distance, encourage maximization)
        loss = -total_distance
        
        if self.reduction == 'sum':
            return loss
        else:
            return loss / len(cluster_centers)

class CombinedClusteringLoss(nn.Module):
    """
    Combined clustering loss function
    Combine multiple clustering losses
    """
    def __init__(self, recon_weight=1.0, cluster_weight=1.0, 
                 center_weight=0.1, contrastive_weight=0.1, 
                 entropy_weight=0.05, compactness_weight=0.1, 
                 separation_weight=0.1, reduction='sum'):
        super(CombinedClusteringLoss, self).__init__()
        self.recon_weight = recon_weight
        self.cluster_weight = cluster_weight
        self.center_weight = center_weight
        self.contrastive_weight = contrastive_weight
        self.entropy_weight = entropy_weight
        self.compactness_weight = compactness_weight
        self.separation_weight = separation_weight
        self.reduction = reduction
        
        # Initialize various loss functions
        self.center_loss = CenterLoss(num_classes=10, feat_dim=128, use_gpu=True)
        self.contrastive_loss = ContrastiveClusteringLoss()
        self.entropy_loss = EntropyLoss()
        self.compactness_loss = CompactnessLoss()
        self.separation_loss = SeparationLoss()
    
    def forward(self, recon_x, x, features, cluster_assignments, cluster_centers):
        """
        Calculate the combined loss
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='none')
        if self.reduction == 'sum':
            recon_loss = recon_loss.sum()
        else:
            recon_loss = recon_loss.mean()
        
        # Basic clustering loss
        cluster_loss = 0
        for i, center in enumerate(cluster_centers):
            mask = (cluster_assignments == i)
            if mask.sum() > 0:
                samples_in_cluster = x[mask]
                distances = torch.sum((samples_in_cluster - center) ** 2, dim=1)
                cluster_loss += distances.sum()
        
        if self.reduction == 'sum':
            cluster_loss = cluster_loss
        else:
            cluster_loss = cluster_loss / x.size(0)
        
        # Center loss
        center_loss = self.center_loss(features, cluster_assignments)
        
        # Contrastive clustering loss
        contrastive_loss = self.contrastive_loss(features, cluster_assignments)
        
        # Entropy loss
        entropy_loss = self.entropy_loss(cluster_assignments, len(cluster_centers))
        
        # Compactness loss
        compactness_loss = self.compactness_loss(features, cluster_assignments)
        
        # Separation loss
        separation_loss = self.separation_loss(features, cluster_assignments)
        
        # Combined total loss
        total_loss = (self.recon_weight * recon_loss + 
                     self.cluster_weight * cluster_loss +
                     self.center_weight * center_loss +
                     self.contrastive_weight * contrastive_loss +
                     self.entropy_weight * entropy_loss +
                     self.compactness_weight * compactness_loss +
                     self.separation_weight * separation_loss)
        
        return total_loss, recon_loss, cluster_loss

class DeepCluster(nn.Module):
    """
    Deep clustering model
    Combine autoencoder and K-means clustering
    """
    def __init__(self, input_dim, hidden_dims, latent_dim, n_clusters, dropout_rate=0.1):
        super(DeepCluster, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent representation layer
        self.latent_layer = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Cluster centers (learnable parameters)
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, latent_dim))
        
    def encode(self, x):
        """Encoder: map the input to the latent space"""
        h = self.encoder(x)
        z = self.latent_layer(h)
        return z
    
    def decode(self, z):
        """Decoder: map the latent variable back to the original space"""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward propagation"""
        z = self.encode(x)
        recon_x = self.decode(z)
        return recon_x, z
    
    def get_cluster_assignments(self, x):
        """Get cluster assignments"""
        with torch.no_grad():
            z = self.encode(x)
            # Calculate the distance to each cluster center
            distances = torch.cdist(z, self.cluster_centers)
            assignments = torch.argmin(distances, dim=1)
        return assignments
    
    def update_cluster_centers(self, dataloader, device):
        """Update cluster centers"""
        self.eval()
        all_latent = []
        
        with torch.no_grad():
            for data, in dataloader:
                data = data.to(device)
                z = self.encode(data)
                all_latent.append(z.cpu())
        
        all_latent = torch.cat(all_latent, dim=0)
        
        # Update cluster centers using K-means
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        kmeans.fit(all_latent.numpy())
        
        # Update cluster center parameters
        with torch.no_grad():
            self.cluster_centers.data = torch.FloatTensor(kmeans.cluster_centers_).to(device)
        
        return kmeans.labels_

class ModelEvaluator:
    """
    Deep clustering model evaluator
    Provide comprehensive model performance evaluation indicators
    """
    
    def __init__(self, device):
        self.device = device
        self.metrics = {}
    
    def evaluate_reconstruction(self, model, data_loader, scaler):
        """
        Evaluate reconstruction quality
        """
        model.eval()
        all_original = []
        all_reconstructed = []
        all_latent = []
        
        with torch.no_grad():
            for data, in data_loader:
                data = data.to(self.device)
                recon_data, latent = model(data)
                
                all_original.append(data.cpu().numpy())
                all_reconstructed.append(recon_data.cpu().numpy())
                all_latent.append(latent.cpu().numpy())
        
        original = np.vstack(all_original)
        reconstructed = np.vstack(all_reconstructed)
        latent = np.vstack(all_latent)
        
        # Unstandardization
        original_unscaled = scaler.inverse_transform(original)
        reconstructed_unscaled = scaler.inverse_transform(reconstructed)
        
        # Calculate reconstruction indicators
        mse = mean_squared_error(original_unscaled.flatten(), reconstructed_unscaled.flatten())
        mae = mean_absolute_error(original_unscaled.flatten(), reconstructed_unscaled.flatten())
        r2 = r2_score(original_unscaled.flatten(), reconstructed_unscaled.flatten())
        
        # Calculate correlation coefficients
        correlation = np.corrcoef(original_unscaled.flatten(), reconstructed_unscaled.flatten())[0, 1]
        
        # Calculate reconstruction error
        reconstruction_error = np.mean(np.abs(original_unscaled - reconstructed_unscaled))
        
        # Calculate reconstruction accuracy
        threshold = 0.1 * np.std(original_unscaled)
        reconstruction_accuracy = np.mean(np.abs(original_unscaled - reconstructed_unscaled) < threshold)
        
        reconstruction_metrics = {
            'mse': mse,
            'mae': mae,
            'r2_score': r2,
            'correlation': correlation,
            'reconstruction_error': reconstruction_error,
            'reconstruction_accuracy': reconstruction_accuracy,
            'original_data': original_unscaled,
            'reconstructed_data': reconstructed_unscaled,
            'latent_representations': latent
        }
        
        return reconstruction_metrics
    
    def evaluate_clustering(self, model, data_loader, true_labels=None):
        """
        Evaluate clustering quality
        """
        model.eval()
        all_latent = []
        all_assignments = []
        
        with torch.no_grad():
            for data, in data_loader:
                data = data.to(self.device)
                latent = model.encode(data)
                assignments = model.get_cluster_assignments(data)
                
                all_latent.append(latent.cpu().numpy())
                all_assignments.append(assignments.cpu().numpy())
        
        latent = np.vstack(all_latent)
        assignments = np.concatenate(all_assignments)
        
        # Calculate clustering indicators
        n_clusters = len(np.unique(assignments))
        cluster_sizes = np.bincount(assignments)
        
        # Calculate intra-cluster distance
        intra_cluster_distances = []
        for i in range(n_clusters):
            cluster_mask = (assignments == i)
            if cluster_mask.sum() > 1:
                cluster_points = latent[cluster_mask]
                distances = np.linalg.norm(cluster_points - cluster_points.mean(axis=0), axis=1)
                intra_cluster_distances.extend(distances)
        
        avg_intra_cluster_distance = np.mean(intra_cluster_distances) if intra_cluster_distances else 0
        
        # Calculate inter-cluster distance
        inter_cluster_distances = []
        cluster_centers = []
        for i in range(n_clusters):
            cluster_mask = (assignments == i)
            if cluster_mask.sum() > 0:
                cluster_centers.append(latent[cluster_mask].mean(axis=0))
        
        if len(cluster_centers) > 1:
            for i in range(len(cluster_centers)):
                for j in range(i + 1, len(cluster_centers)):
                    distance = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                    inter_cluster_distances.append(distance)
        
        avg_inter_cluster_distance = np.mean(inter_cluster_distances) if inter_cluster_distances else 0
        
        # Calculate silhouette coefficient
        from sklearn.metrics import silhouette_score
        if len(np.unique(assignments)) > 1:
            silhouette_avg = silhouette_score(latent, assignments)
        else:
            silhouette_avg = 0
        
        # Calculate cluster balance
        cluster_balance = 1 - np.std(cluster_sizes) / np.mean(cluster_sizes) if np.mean(cluster_sizes) > 0 else 0
        
        clustering_metrics = {
            'n_clusters': n_clusters,
            'cluster_sizes': cluster_sizes,
            'avg_intra_cluster_distance': avg_intra_cluster_distance,
            'avg_inter_cluster_distance': avg_inter_cluster_distance,
            'silhouette_score': silhouette_avg,
            'cluster_balance': cluster_balance,
            'assignments': assignments,
            'latent_representations': latent,
            'cluster_centers': np.array(cluster_centers)
        }
        
        return clustering_metrics
    
    def compute_comprehensive_metrics(self, model, data_loader, scaler, true_labels=None):
        """
        Calculate comprehensive evaluation indicators
        """
        print("Start the comprehensive model evaluation...")
        
        # Evaluate reconstruction quality
        print("Evaluate the quality of reconstruction...")
        recon_metrics = self.evaluate_reconstruction(model, data_loader, scaler)
        
        # Evaluate clustering quality
        print("Evaluate clustering quality...")
        cluster_metrics = self.evaluate_clustering(model, data_loader, true_labels)

        comprehensive_metrics = {
            'reconstruction': recon_metrics,
            'clustering': cluster_metrics
        }
        
        return comprehensive_metrics
    

def visualize_results(metrics, save_path=None):
    """
    Visualize evaluation results
    """
    # fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    fig.suptitle('Deep clustering model evaluation results', fontsize=16, fontweight='bold')
    
    # 1. Reconstruction quality scatter plot
    ax1 = axes[0, 0]
    original = metrics['reconstruction']['original_data'].flatten()
    reconstructed = metrics['reconstruction']['reconstructed_data'].flatten()
    
    if len(original) > 1000:
        indices = np.random.choice(len(original), 1000, replace=False)
        original_sample = original[indices]
        reconstructed_sample = reconstructed[indices]
    else:
        original_sample = original
        reconstructed_sample = reconstructed
    
    ax1.scatter(original_sample, reconstructed_sample, alpha=0.5, s=1)
    ax1.plot([original_sample.min(), original_sample.max()], 
             [original_sample.min(), original_sample.max()], 'r--', lw=2)
    ax1.set_xlabel('Original data')
    ax1.set_ylabel('Reconstructed data')  
    ax1.set_title(f'Reconstruction metrics (R² = {metrics["reconstruction"]["r2_score"]:.3f})')
    ax1.grid(True, alpha=0.3)
    
    # 2. Reconstruction error distribution
    ax2 = axes[0, 1]
    reconstruction_error = np.abs(original - reconstructed)
    ax2.hist(reconstruction_error, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Reconstruction error')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Reconstruction error distribution (MAE = {metrics["reconstruction"]["mae"]:.3f})')
    ax2.grid(True, alpha=0.3)
    
    # 3. Clustering result visualization (t-SNE)
    ax3 = axes[1, 0]
    # ax3 = axes[0, 2]
    latent = metrics['clustering']['latent_representations']
    assignments = metrics['clustering']['assignments']
    
    # If the latent dimension > 2, use t-SNE to reduce dimensionality
    if latent.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=42)
        latent_2d = tsne.fit_transform(latent[:1000])  # Only use the first 1000 samples
        assignments_2d = assignments[:1000]
    else:
        latent_2d = latent[:1000]
        assignments_2d = assignments[:1000]
    
    scatter = ax3.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                        c=assignments_2d, cmap='tab10', alpha=0.6)
    ax3.set_xlabel('t-SNE 1')
    ax3.set_ylabel('t-SNE 2')
    ax3.set_title('Clustering result visualization')
    plt.colorbar(scatter, ax=ax3, label='Clustering labels')
    
    # 4. Clustering size distribution
    # ax4 = axes[1, 0]
    ax4 = axes[1, 1]
    cluster_sizes = metrics['clustering']['cluster_sizes']
    ax4.bar(range(len(cluster_sizes)), cluster_sizes, alpha=0.7, color='lightcoral')
    ax4.set_xlabel('Clustering labels')
    ax4.set_ylabel('Sample number')
    ax4.set_title(f'Clustering size distribution (Balance = {metrics["clustering"]["cluster_balance"]:.3f})')
    ax4.grid(True, alpha=0.3)
    
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()

def print_evaluation_report(metrics):
    """
    Print detailed evaluation report
    """
    print("\n" + "="*60)
    print("Deep clustering model evaluation report")
    print("="*60)
    
    # Reconstruction quality
    print("\n Reconstruction quality evaluation:")
    print("-" * 30)
    print(f"MSE: {metrics['reconstruction']['mse']:.6f}")
    print(f"MAE: {metrics['reconstruction']['mae']:.6f}")
    print(f"R²: {metrics['reconstruction']['r2_score']:.6f}")
    print(f"Correlation: {metrics['reconstruction']['correlation']:.6f}")
    print(f"Reconstruction error: {metrics['reconstruction']['reconstruction_error']:.6f}")
    print(f"Reconstruction accuracy: {metrics['reconstruction']['reconstruction_accuracy']:.4f}")
    
    # Clustering quality
    print("\n Clustering quality evaluation:")
    print("-" * 30)
    print(f"Clustering number: {metrics['clustering']['n_clusters']}")
    print(f"Silhouette score: {metrics['clustering']['silhouette_score']:.6f}")
    print(f"Clustering balance: {metrics['clustering']['cluster_balance']:.6f}")
    print(f"Average intra-cluster distance: {metrics['clustering']['avg_intra_cluster_distance']:.6f}")
    print(f"Average inter-cluster distance: {metrics['clustering']['avg_inter_cluster_distance']:.6f}")
    print(f"Clustering size: {metrics['clustering']['cluster_sizes']}")


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
        """Save the best model weights"""
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

def prepare_data(seed, batch_size, ratio="8:1:1", jobid="20240808232043_OtJF37SH", kfold=0):
    """
    Prepare training, validation and test data
    First row is column name
    """
    if ratio == "0" and kfold == 0: 
        scaler = StandardScaler()
        # Directly load training-validation-test set file
        # data = pd.read_csv("D:\\wamp\www\\multi_omics_own\\download_data\\Jobs\\"+jobid+"\\"+jobid+"_data.txt",sep="\t")   # the path of training dataset
        data = pd.read_csv("../../data/train_exmaple_un/train_exmaple_un_data.txt", sep="\t")   # the path of training dataset
        data = data.dropna().values.astype(float)
        data = scaler.fit_transform(data)
        data = torch.FloatTensor(data)
        train_loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)
        
        # data = pd.read_csv("D:\\wamp\www\\multi_omics_own\\download_data\\Jobs\\"+jobid+"\\"+jobid+"_val.txt",sep="\t")  # the path of validation dataset
        data = pd.read_csv("../../data/train_exmaple_un/train_exmaple_un_val.txt", sep="\t")  # the path of validation dataset
        data = data.dropna().values.astype(float) 
        data = scaler.transform(data)
        data = torch.FloatTensor(data)
        val_loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)

        # data = pd.read_csv("D:\\wamp\www\\multi_omics_own\\download_data\\Jobs\\"+jobid+"\\"+jobid+"_test.txt",sep="\t")   # the path of testing dataset
        data = pd.read_csv("../../data/train_exmaple_un/train_exmaple_un_test.txt",sep="\t")   # the path of testing dataset
        data = data.dropna().values.astype(float)
        data = scaler.transform(data)
        data = torch.FloatTensor(data)
        test_loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)

        
    elif ratio != "0" and kfold == 0:   
        # Split training-validation-test set
        # data = pd.read_csv("D:\\wamp\www\\multi_omics_own\\download_data\\Jobs\\"+jobid+"\\"+jobid+"_data.txt",sep="\t")   # the path of training dataset
        data = pd.read_csv("../../data/train_exmaple_un/train_exmaple_un_data.txt", sep="\t")   # the path of training dataset
        
        data = data.dropna().values.astype(float)
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

        ratio_str = ratio.split(":")
        ratio_num = list(map(int, ratio_str))  # [8, 1, 1]
        train_ratio = ratio_num[0] / sum(ratio_num)
        test_ratio = ratio_num[2] / sum(ratio_num[1:])

        train_data, res_data = train_test_split(data, test_size=1-train_ratio, random_state=seed)
        val_data, test_data = train_test_split(res_data, test_size=test_ratio, random_state=seed)

        train_data = torch.FloatTensor(train_data)
        val_data = torch.FloatTensor(val_data)
        test_data = torch.FloatTensor(test_data)

        train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_data), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(test_data), batch_size=batch_size, shuffle=True)
    else: 
        # kfold training
        scaler = StandardScaler()
        # Directly load training-validation-test set file
        # data = pd.read_csv("D:\\wamp\www\\multi_omics_own\\download_data\\Jobs\\"+jobid+"\\"+jobid+"_data.txt",sep="\t")   # the path of training dataset
        data = pd.read_csv("../../data/train_exmaple_un/train_exmaple_un_data.txt", sep="\t")   # the path of training dataset
        data = data.dropna().values.astype(float)
        data = scaler.fit_transform(data)
        data = torch.FloatTensor(data)
        train_loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)

        val_loader = 0
        data = pd.read_csv("../../data/train_exmaple_un/train_exmaple_un_test.txt",sep="\t")   # the path of testing dataset
        data = data.dropna().values.astype(float)
        data = scaler.transform(data)
        data = torch.FloatTensor(data)
        test_loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)


    input_dim = data.shape[1]
    return input_dim, train_loader, val_loader, test_loader, scaler

def train_epoch(model, train_loader, optimizer, device, loss_function='mse', alpha=1.0, beta=1.0):
    """Train one epoch"""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_cluster_loss = 0
    
    for batch_idx, (data,) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, latent = model(data)
        
        # Get cluster assignments
        cluster_assignments = model.get_cluster_assignments(data)
        
        # Calculate loss based on loss function type
        if loss_function.lower() == 'deepcluster':
            loss_fn = DeepClusterLoss(alpha=alpha, beta=beta, reduction='sum')
            loss, recon_loss, cluster_loss = loss_fn(recon_batch, data, cluster_assignments, model.cluster_centers)
        elif loss_function.lower() == 'combined':
            loss_fn = CombinedClusteringLoss(
                recon_weight=alpha, cluster_weight=beta, 
                center_weight=0.1, contrastive_weight=0.1,
                entropy_weight=0.05, compactness_weight=0.1,
                separation_weight=0.1, reduction='sum'
            )
            loss, recon_loss, cluster_loss = loss_fn(recon_batch, data, latent, cluster_assignments, model.cluster_centers)
        elif loss_function.lower() == 'center':
            # Center loss
            center_loss_fn = CenterLoss(num_classes=model.n_clusters, feat_dim=model.latent_dim, use_gpu=device.type=='cuda')
            recon_loss = F.mse_loss(recon_batch, data, reduction='sum')
            cluster_loss = center_loss_fn(latent, cluster_assignments)
            loss = alpha * recon_loss + beta * cluster_loss
        elif loss_function.lower() == 'contrastive':
            # Contrastive clustering loss
            contrastive_loss_fn = ContrastiveClusteringLoss(temperature=0.1)
            recon_loss = F.mse_loss(recon_batch, data, reduction='sum')
            cluster_loss = contrastive_loss_fn(latent, cluster_assignments)
            loss = alpha * recon_loss + beta * cluster_loss
        elif loss_function.lower() == 'spectral':
            # Spectral clustering loss
            spectral_loss_fn = SpectralClusteringLoss()
            recon_loss = F.mse_loss(recon_batch, data, reduction='sum')
            cluster_loss = spectral_loss_fn(latent, cluster_assignments)
            loss = alpha * recon_loss + beta * cluster_loss
        elif loss_function.lower() == 'entropy':
            # Entropy loss
            entropy_loss_fn = EntropyLoss()
            recon_loss = F.mse_loss(recon_batch, data, reduction='sum')
            cluster_loss = entropy_loss_fn(cluster_assignments, model.n_clusters)
            loss = alpha * recon_loss + beta * cluster_loss
        elif loss_function.lower() == 'compactness':
            # Compactness loss
            compactness_loss_fn = CompactnessLoss()
            recon_loss = F.mse_loss(recon_batch, data, reduction='sum')
            cluster_loss = compactness_loss_fn(latent, cluster_assignments)
            loss = alpha * recon_loss + beta * cluster_loss
        elif loss_function.lower() == 'separation':
            # Separation loss
            separation_loss_fn = SeparationLoss()
            recon_loss = F.mse_loss(recon_batch, data, reduction='sum')
            cluster_loss = separation_loss_fn(latent, cluster_assignments)
            loss = alpha * recon_loss + beta * cluster_loss
        else:
            # Default use basic deep clustering loss
            loss_fn = DeepClusterLoss(alpha=alpha, beta=beta, reduction='sum')
            loss, recon_loss, cluster_loss = loss_fn(recon_batch, data, cluster_assignments, model.cluster_centers)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_cluster_loss += cluster_loss.item()
    
    return total_loss / len(train_loader), total_recon_loss / len(train_loader), total_cluster_loss / len(train_loader)

def validate_epoch(model, val_loader, device, loss_function='mse', alpha=1.0, beta=1.0):
    """Validate one epoch"""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_cluster_loss = 0
    
    with torch.no_grad():
        for data, in val_loader:
            data = data.to(device)
            recon_batch, latent = model(data)
            
            # Get cluster assignments
            cluster_assignments = model.get_cluster_assignments(data)
            
            # Calculate loss based on loss function type
            if loss_function.lower() == 'deepcluster':
                loss_fn = DeepClusterLoss(alpha=alpha, beta=beta, reduction='sum')
                loss, recon_loss, cluster_loss = loss_fn(recon_batch, data, cluster_assignments, model.cluster_centers)
            elif loss_function.lower() == 'combined':
                loss_fn = CombinedClusteringLoss(
                    recon_weight=alpha, cluster_weight=beta, 
                    center_weight=0.1, contrastive_weight=0.1,
                    entropy_weight=0.05, compactness_weight=0.1,
                    separation_weight=0.1, reduction='sum'
                )
                loss, recon_loss, cluster_loss = loss_fn(recon_batch, data, latent, cluster_assignments, model.cluster_centers)
            elif loss_function.lower() == 'center':
                center_loss_fn = CenterLoss(num_classes=model.n_clusters, feat_dim=model.latent_dim, use_gpu=device.type=='cuda')
                recon_loss = F.mse_loss(recon_batch, data, reduction='sum')
                cluster_loss = center_loss_fn(latent, cluster_assignments)
                loss = alpha * recon_loss + beta * cluster_loss
            elif loss_function.lower() == 'contrastive':
                contrastive_loss_fn = ContrastiveClusteringLoss(temperature=0.1)
                recon_loss = F.mse_loss(recon_batch, data, reduction='sum')
                cluster_loss = contrastive_loss_fn(latent, cluster_assignments)
                loss = alpha * recon_loss + beta * cluster_loss
            elif loss_function.lower() == 'spectral':
                spectral_loss_fn = SpectralClusteringLoss()
                recon_loss = F.mse_loss(recon_batch, data, reduction='sum')
                cluster_loss = spectral_loss_fn(latent, cluster_assignments)
                loss = alpha * recon_loss + beta * cluster_loss
            elif loss_function.lower() == 'entropy':
                entropy_loss_fn = EntropyLoss()
                recon_loss = F.mse_loss(recon_batch, data, reduction='sum')
                cluster_loss = entropy_loss_fn(cluster_assignments, model.n_clusters)
                loss = alpha * recon_loss + beta * cluster_loss
            elif loss_function.lower() == 'compactness':
                compactness_loss_fn = CompactnessLoss()
                recon_loss = F.mse_loss(recon_batch, data, reduction='sum')
                cluster_loss = compactness_loss_fn(latent, cluster_assignments)
                loss = alpha * recon_loss + beta * cluster_loss
            elif loss_function.lower() == 'separation':
                separation_loss_fn = SeparationLoss()
                recon_loss = F.mse_loss(recon_batch, data, reduction='sum')
                cluster_loss = separation_loss_fn(latent, cluster_assignments)
                loss = alpha * recon_loss + beta * cluster_loss
            else:
                loss_fn = DeepClusterLoss(alpha=alpha, beta=beta, reduction='sum')
                loss, recon_loss, cluster_loss = loss_fn(recon_batch, data, cluster_assignments, model.cluster_centers)
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_cluster_loss += cluster_loss.item()
    
    return total_loss / len(val_loader), total_recon_loss / len(val_loader), total_cluster_loss / len(val_loader)

def train_model(model, train_loader, val_loader, optimizer, device, epochs, patience, model_name, 
                loss_function='mse', alpha=1.0, beta=1.0, update_cluster_centers_freq=10):
    """Train model"""
    early_stopping = EarlyStopping(patience=patience)
    train_losses = []
    val_losses = []
    
    print(f"Start training model: {model_name}")
    print(f"Loss function: {loss_function}")
    print(f"Reconstruction loss weight: {alpha}, Cluster loss weight: {beta}")
    print("-" * 50)
    
    for epoch in range(epochs):
        train_loss, train_recon, train_cluster = train_epoch(
            model, train_loader, optimizer, device, loss_function, alpha, beta
        )
        val_loss, val_recon, val_cluster = validate_epoch(
            model, val_loader, device, loss_function, alpha, beta
        )
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  train loss: {train_loss:.4f} (reconstruction: {train_recon:.4f}, clustering: {train_cluster:.4f})')  
        print(f'  validation loss: {val_loss:.4f} (reconstruction: {val_recon:.4f}, clustering: {val_cluster:.4f})')
        
        # Update cluster centers periodically
        if (epoch + 1) % update_cluster_centers_freq == 0:
            print(f'  Update cluster centers...')
            model.update_cluster_centers(train_loader, device)
        
        if early_stopping(val_loss, model):
            print(f'Early stopping triggered, stopping training at epoch {epoch+1}')
            break
    
    return train_losses, val_losses

def test_model(model, test_loader, device, loss_function='mse', alpha=1.0, beta=1.0):
    """Test model"""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_cluster_loss = 0
    
    with torch.no_grad():
        for data, in test_loader:
            data = data.to(device)
            recon_batch, latent = model(data)
            
            # Get cluster assignments
            cluster_assignments = model.get_cluster_assignments(data)
            
            # Calculate loss based on loss function type
            if loss_function.lower() == 'deepcluster':
                loss_fn = DeepClusterLoss(alpha=alpha, beta=beta, reduction='sum')
                loss, recon_loss, cluster_loss = loss_fn(recon_batch, data, cluster_assignments, model.cluster_centers)
            elif loss_function.lower() == 'combined':
                loss_fn = CombinedClusteringLoss(
                    recon_weight=alpha, cluster_weight=beta, 
                    center_weight=0.1, contrastive_weight=0.1,
                    entropy_weight=0.05, compactness_weight=0.1,
                    separation_weight=0.1, reduction='sum'
                )
                loss, recon_loss, cluster_loss = loss_fn(recon_batch, data, latent, cluster_assignments, model.cluster_centers)
            elif loss_function.lower() == 'center':
                center_loss_fn = CenterLoss(num_classes=model.n_clusters, feat_dim=model.latent_dim, use_gpu=device.type=='cuda')
                recon_loss = F.mse_loss(recon_batch, data, reduction='sum')
                cluster_loss = center_loss_fn(latent, cluster_assignments)
                loss = alpha * recon_loss + beta * cluster_loss
            elif loss_function.lower() == 'contrastive':
                contrastive_loss_fn = ContrastiveClusteringLoss(temperature=0.1)
                recon_loss = F.mse_loss(recon_batch, data, reduction='sum')
                cluster_loss = contrastive_loss_fn(latent, cluster_assignments)
                loss = alpha * recon_loss + beta * cluster_loss
            elif loss_function.lower() == 'spectral':
                spectral_loss_fn = SpectralClusteringLoss()
                recon_loss = F.mse_loss(recon_batch, data, reduction='sum')
                cluster_loss = spectral_loss_fn(latent, cluster_assignments)
                loss = alpha * recon_loss + beta * cluster_loss
            elif loss_function.lower() == 'entropy':
                entropy_loss_fn = EntropyLoss()
                recon_loss = F.mse_loss(recon_batch, data, reduction='sum')
                cluster_loss = entropy_loss_fn(cluster_assignments, model.n_clusters)
                loss = alpha * recon_loss + beta * cluster_loss
            elif loss_function.lower() == 'compactness':
                compactness_loss_fn = CompactnessLoss()
                recon_loss = F.mse_loss(recon_batch, data, reduction='sum')
                cluster_loss = compactness_loss_fn(latent, cluster_assignments)
                loss = alpha * recon_loss + beta * cluster_loss
            elif loss_function.lower() == 'separation':
                separation_loss_fn = SeparationLoss()
                recon_loss = F.mse_loss(recon_batch, data, reduction='sum')
                cluster_loss = separation_loss_fn(latent, cluster_assignments)
                loss = alpha * recon_loss + beta * cluster_loss
            else:
                loss_fn = DeepClusterLoss(alpha=alpha, beta=beta, reduction='sum')
                loss, recon_loss, cluster_loss = loss_fn(recon_batch, data, cluster_assignments, model.cluster_centers)
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_cluster_loss += cluster_loss.item()
    
    avg_loss = total_loss / len(test_loader)
    avg_recon_loss = total_recon_loss / len(test_loader)
    avg_cluster_loss = total_cluster_loss / len(test_loader)
    
    print(f"Test results:")
    print(f"  Total loss: {avg_loss:.4f}")
    print(f"  Reconstruction loss: {avg_recon_loss:.4f}")
    print(f"  Clustering loss: {avg_cluster_loss:.4f}")
    
    return avg_loss, avg_recon_loss, avg_cluster_loss

def kfold_cross_validation(args, k_folds=5):
    """k-fold cross-validation"""
    print(f"\nStart {k_folds}-fold cross-validation")
    print("=" * 60)
    

    # Prepare data
    dim, train_loader, _, test_loader, scaler = prepare_data(
        args.random_seed, args.batch_size, args.ratio, args.jobid, k_folds
    )
    
    
    # Get training data for k-fold
    train_data_list = []
    for batch_data, in train_loader:
        train_data_list.append(batch_data.numpy())
    train_data_scaled = np.vstack(train_data_list)
    
    # On training data for k-fold cross-validation
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=args.random_seed)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data_scaled)):
        print(f"\nFold {fold + 1}/{k_folds}")
        print("-" * 30)
        
        # Split data
        fold_train_data = torch.FloatTensor(train_data_scaled[train_idx])
        fold_val_data = torch.FloatTensor(train_data_scaled[val_idx])
        
        fold_train_loader = DataLoader(TensorDataset(fold_train_data), batch_size=args.batch_size, shuffle=True)
        fold_val_loader = DataLoader(TensorDataset(fold_val_data), batch_size=args.batch_size, shuffle=False)
        
        # Create model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DeepCluster(
            input_dim=dim,
            hidden_dims=[512, 256, 128],
            latent_dim=args.latent_dim,
            n_clusters=args.n_clusters,
            dropout_rate=args.dropout
        ).to(device)
        
        # Create optimizer
        if args.optimizer_function.lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        elif args.optimizer_function.lower() == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
        elif args.optimizer_function.lower() == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        

        # Train model
        train_losses, val_losses = train_model(   
            model, fold_train_loader, fold_val_loader, optimizer, device, 
            args.epochs, args.early_stopping_patience, f"Fold_{fold+1}", 
            loss_function=args.loss_function, alpha=args.alpha, beta=args.beta, 
            update_cluster_centers_freq=args.update_cluster_freq
        )
        
        # Test model
        test_loss, test_recon, test_cluster = test_model(
            model, test_loader, device, loss_function=args.loss_function, 
            alpha=args.alpha, beta=args.beta
        )

        fold_results.append({
            'fold': fold + 1,
            'test_loss': test_loss,
            'test_recon_loss': test_recon,
            'test_cluster': test_cluster,
            'train_losses': train_losses,
            'val_losses': val_losses
        })
    
    # Calculate average results
    avg_test_loss = np.mean([r['test_loss'] for r in fold_results])
    avg_test_recon = np.mean([r['test_recon_loss'] for r in fold_results])
    avg_test_cluster= np.mean([r['test_cluster'] for r in fold_results])
    
    print(f"\n{k_folds}-fold cross-validation results:")
    print("=" * 40)
    print(f"Average validation loss: {avg_test_loss:.4f} ± {np.std([r['test_loss'] for r in fold_results]):.4f}")
    print(f"Average reconstruction loss: {avg_test_recon:.4f} ± {np.std([r['test_recon_loss'] for r in fold_results]):.4f}")
    print(f"Average clustering loss: {test_cluster:.4f} ± {np.std([r['test_cluster'] for r in fold_results]):.4f}")
    
    return fold_results, model, scaler, test_loader, dim





def save_model(model, scaler, args, model_path, scaler_path, input_dim):
    """Save model and preprocessor"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': input_dim,
            'hidden_dims': [512, 256, 128],
            'latent_dim': args.latent_dim,
            'n_clusters': args.n_clusters,
            'dropout_rate': args.dropout
        },
        'args': vars(args)
    }, model_path)
    
    # Save scaler
    import pickle
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # print(f"Model saved to: {model_path}")
    # print(f"Preprocessor saved to: {scaler_path}")

def load_model(model_path, scaler_path, device):
    """Load model and preprocessor"""
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['model_config']
    
    model = DeepCluster(
        input_dim=model_config['input_dim'],
        hidden_dims=model_config['hidden_dims'],
        latent_dim=model_config['latent_dim'],
        n_clusters=model_config['n_clusters'],
        dropout_rate=model_config['dropout_rate']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load scaler
    import pickle
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # print(f"Model loaded from: {model_path}")
    # print(f"Preprocessor loaded from: {scaler_path}")
    
    return model, scaler

def predict(model, scaler, data, device):
    """Use trained model to predict"""
    model.eval()
    
    # Data preprocessing
    data_scaled = scaler.transform(data)
    data_tensor = torch.FloatTensor(data_scaled).to(device)
    
    with torch.no_grad():
        recon_data, latent = model(data_tensor)
        cluster_assignments = model.get_cluster_assignments(data_tensor)
    
    return recon_data.cpu().numpy(), latent.cpu().numpy(), cluster_assignments.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description='Deep clustering model training script')

    # Data path
    parser.add_argument('--ratio', type=str, default='0', help='Data split ratio')
    parser.add_argument('--jobid', type=str, default='20240808232043_OtJF37SH', help='Dataset ID')
    
    # Model parameters
    parser.add_argument('--latent_dim', type=int, default=20, help='Latent space dimension')
    parser.add_argument('--n_clusters', type=int, default=5, help='Cluster number')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience')
    
    # Loss function parameters
    parser.add_argument('--loss_function', type=str, default='deepcluster', 
                       choices=['deepcluster', 'combined', 'center', 'contrastive', 
                               'spectral', 'entropy', 'compactness', 'separation'], 
                       help='Loss function type:\n'
                            'deepcluster: Basic deep clustering loss\n'
                            'combined: Combined multiple clustering losses\n'
                            'center: Center loss\n'
                            'contrastive: Contrastive clustering loss\n'
                            'spectral: Spectral clustering loss\n'
                            'entropy: Entropy loss\n'
                            'compactness: Compactness loss\n'
                            'separation: Separation loss\n')
    parser.add_argument('--alpha', type=float, default=1.0, help='Reconstruction loss weight')
    parser.add_argument('--beta', type=float, default=1.0, help='Cluster loss weight')
    parser.add_argument('--update_cluster_freq', type=int, default=10, help='Update cluster center frequency')
    
    # Optimizer
    parser.add_argument('--optimizer_function', type=str, default='adam', 
                       choices=['adam', 'sgd', 'adamw'], help='Optimizer type')
    
    # Other parameters
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--k_folds', type=int, default=0, help='k-fold number')
    parser.add_argument('--save_model', action='store_true', default=1, help='Save model')
    parser.add_argument('--model_path', type=str, default='deepcluster_model.pth', help='Model save path')
    parser.add_argument('--scaler_path', type=str, default='deepcluster_scaler.pkl', help='Preprocessor save path')
    
    # Evaluation related parameters
    parser.add_argument('--evaluate_model', action='store_true', default=1, help='Evaluate model')
    parser.add_argument('--save_evaluation', action='store_true', default=1, help='Save evaluation results')
    parser.add_argument('--evaluation_path', type=str, default='deepcluster_evaluation_results.png', help='Evaluation results save path')
    parser.add_argument('--show_plots', action='store_true', default=1, help='Show plots')
    
    args = parser.parse_args()

    print("DeepCluster model")
    print('ratio =', args.ratio)
    print('k_folds =', args.k_folds)
    print('latent_dim =', args.latent_dim)
    print('n_clusters =', args.n_clusters)
    print('dropout =', args.dropout)
    print('epochs =', args.epochs)
    print('batch_size =', args.batch_size)
    print('learning_rate =', args.learning_rate)
    print('early_stopping_patience =', args.early_stopping_patience)
    print('optimizer_function =', args.optimizer_function)
    print('loss_function =', args.loss_function)
    print('random_seed =', args.random_seed)
    print('alpha =', args.alpha)
    print('beta =', args.beta)
    print('update_cluster_freq =', args.update_cluster_freq)
    

    

    # Set random seed
    set_random_seed(args.random_seed)
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.k_folds:
        # kfold training
        fold_results, model, scaler, test_loader, dim = kfold_cross_validation(args, args.k_folds)


    else:
        # Prepare data
        dim, train_loader, val_loader, test_loader, scaler = prepare_data(
            args.random_seed, args.batch_size, args.ratio, args.jobid, args.k_folds
        )
        
        # Create model
        model = DeepCluster(
            input_dim=dim,
            hidden_dims=[512, 256, 128],
            latent_dim=args.latent_dim,
            n_clusters=args.n_clusters,
            dropout_rate=args.dropout
        ).to(device)
        
        # Create optimizer
        if args.optimizer_function.lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        elif args.optimizer_function.lower() == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
        elif args.optimizer_function.lower() == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        
        # Train model
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, optimizer, device, 
            args.epochs, args.early_stopping_patience, "DeepCluster_Model", 
            loss_function=args.loss_function, alpha=args.alpha, beta=args.beta, 
            update_cluster_centers_freq=args.update_cluster_freq
        )
        
        # Test model
        test_loss, test_recon, test_cluster = test_model(
            model, test_loader, device, loss_function=args.loss_function, 
            alpha=args.alpha, beta=args.beta
        )


    
    # Save model
    if args.save_model:
        save_model(model, scaler, args, args.model_path, args.scaler_path, dim)
    
    # Model evaluation
    if args.evaluate_model:
        print("\n" + "="*60)
        print("Start model evaluation")
        print("="*60)
        

        evaluator = ModelEvaluator(device)
        metrics = evaluator.compute_comprehensive_metrics(model, test_loader, scaler)
        print_evaluation_report(metrics)
        
        # Visualize results
        if args.show_plots:
            visualize_results(metrics, args.evaluation_path if args.save_evaluation else None)
        
        # Save evaluation results
        if args.save_evaluation:
            # Save evaluation indicators to JSON file
            evaluation_data = {
                'reconstruction_metrics': {
                    'mse': float(metrics['reconstruction']['mse']),
                    'mae': float(metrics['reconstruction']['mae']),
                    'r2_score': float(metrics['reconstruction']['r2_score']),
                    'correlation': float(metrics['reconstruction']['correlation']),
                    'reconstruction_error': float(metrics['reconstruction']['reconstruction_error']),
                    'reconstruction_accuracy': float(metrics['reconstruction']['reconstruction_accuracy'])
                },
                'clustering_metrics': {
                    'n_clusters': int(metrics['clustering']['n_clusters']),
                    'silhouette_score': float(metrics['clustering']['silhouette_score']),
                    'cluster_balance': float(metrics['clustering']['cluster_balance']),
                    'avg_intra_cluster_distance': float(metrics['clustering']['avg_intra_cluster_distance']),
                    'avg_inter_cluster_distance': float(metrics['clustering']['avg_inter_cluster_distance'])
                }
            }
            
            json_path = args.evaluation_path.replace('.png', '.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # cmd: python deepcluster.py --ratio 0 --jobid jobid
    # cmd: python deepcluster.py --ratio 8:1:1 --n_clusters 5 --jobid jobid
    # output: deepcluster_evaluation_results.json、deepcluster_evaluation_results.png、deepcluster_model.pth、deepcluster_scaler.pkl
    main()
