import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import os
import json
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
import warnings
warnings.filterwarnings('ignore')

class FocalLoss(nn.Module):
    """
    Focal Loss implementation
    For handling class imbalance problem
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='sum'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # calculate probability
        pt = torch.exp(-bce_loss)
        
        # calculate Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function
    For learning the representation of similar samples
    """
    def __init__(self, margin=1.0, reduction='sum'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, x1, x2, labels):
        """
        x1, x2: input sample pairs
        labels: 1 represents similarity, 0 represents dissimilarity
        """
        euclidean_distance = F.pairwise_distance(x1, x2)
        loss_contrastive = torch.mean((1-labels) * torch.pow(euclidean_distance, 2) +
                                     (labels) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        
        if self.reduction == 'sum':
            return loss_contrastive * x1.size(0)
        else:
            return loss_contrastive

class SpectralLoss(nn.Module):
    """
    Spectral loss function
    Based on feature value decomposition
    """
    def __init__(self, reduction='sum'):
        super(SpectralLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, x, recon_x):
        # calculate covariance matrix
        x_centered = x - x.mean(dim=0)
        recon_centered = recon_x - recon_x.mean(dim=0)
        
        cov_x = torch.mm(x_centered.t(), x_centered) / (x.size(0) - 1)
        cov_recon = torch.mm(recon_centered.t(), recon_centered) / (recon_x.size(0) - 1)
        
        # calculate Frobenius norm
        spectral_loss = torch.norm(cov_x - cov_recon, p='fro')
        
        if self.reduction == 'sum':
            return spectral_loss * x.size(0)
        else:
            return spectral_loss

class WassersteinLoss(nn.Module):
    """
    Wasserstein distance loss
    For measuring the distance between two distributions
    """
    def __init__(self, reduction='sum'):
        super(WassersteinLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, x, recon_x):
        # simplified Wasserstein distance calculation
        # calculate the first moment of the empirical distribution
        mean_x = x.mean(dim=0)
        mean_recon = recon_x.mean(dim=0)
        
        # calculate the second moment of the empirical distribution
        var_x = x.var(dim=0)
        var_recon = recon_x.var(dim=0)
        
        # Wasserstein distance approximation
        wasserstein_loss = torch.norm(mean_x - mean_recon, p=2) + torch.norm(var_x - var_recon, p=2)
        
        if self.reduction == 'sum':
            return wasserstein_loss * x.size(0)
        else:
            return wasserstein_loss

class PerceptualLoss(nn.Module):
    """
    Perceptual loss function
    Based on feature similarity
    """
    def __init__(self, reduction='sum'):
        super(PerceptualLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, x, recon_x):
        # calculate feature similarity (using L2 distance)
        perceptual_loss = F.mse_loss(x, recon_x, reduction='none')
        
        # add gradient penalty
        grad_x = torch.autograd.grad(x.sum(), x, create_graph=True)[0]
        grad_recon = torch.autograd.grad(recon_x.sum(), recon_x, create_graph=True)[0]
        grad_loss = F.mse_loss(grad_x, grad_recon, reduction='none')
        
        total_loss = perceptual_loss + 0.1 * grad_loss
        
        if self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss.mean()

class CosineSimilarityLoss(nn.Module):
    """
    Cosine similarity loss
    Based on cosine similarity
    """
    def __init__(self, reduction='sum'):
        super(CosineSimilarityLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, x, recon_x):
        # calculate cosine similarity
        cosine_sim = F.cosine_similarity(x, recon_x, dim=1)
        # convert to loss (1 - similarity)
        cosine_loss = 1 - cosine_sim
        
        if self.reduction == 'sum':
            return cosine_loss.sum()
        else:
            return cosine_loss.mean()

class KLDivergenceLoss(nn.Module):
    """
    KL divergence loss (for reconstruction)
    For calculating the KL divergence between the reconstruction distribution and the original distribution
    """
    def __init__(self, reduction='sum'):
        super(KLDivergenceLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, x, recon_x):
        # convert data to probability distribution
        x_prob = F.softmax(x, dim=1)
        recon_prob = F.softmax(recon_x, dim=1)
        
        # calculate KL divergence
        kl_loss = F.kl_div(F.log_softmax(recon_x, dim=1), x_prob, reduction='none')
        
        if self.reduction == 'sum':
            return kl_loss.sum()
        else:
            return kl_loss.mean()

class RegularizationLoss(nn.Module):
    """
    Regularization loss function
    Contains L1, L2, and elastic network regularization
    """
    def __init__(self, l1_weight=0.01, l2_weight=0.01, elastic_weight=0.01, reduction='sum'):
        super(RegularizationLoss, self).__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.elastic_weight = elastic_weight
        self.reduction = reduction
    
    def forward(self, x, recon_x):
        # L1 regularization
        l1_loss = self.l1_weight * torch.norm(x - recon_x, p=1)
        
        # L2 regularization
        l2_loss = self.l2_weight * torch.norm(x - recon_x, p=2)
        
        # elastic network regularization
        elastic_loss = self.elastic_weight * (torch.norm(x - recon_x, p=1) + torch.norm(x - recon_x, p=2))
        
        total_loss = l1_loss + l2_loss + elastic_loss
        
        if self.reduction == 'sum':
            return total_loss
        else:
            return total_loss / x.size(0)

class InfoNCELoss(nn.Module):
    """
    InfoNCE loss function
    For contrastive learning, maximizing mutual information
    """
    def __init__(self, temperature=0.07, reduction='sum'):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, x, recon_x):
        # calculate similarity matrix
        x_norm = F.normalize(x, p=2, dim=1)
        recon_norm = F.normalize(recon_x, p=2, dim=1)
        
        # calculate similarity
        similarity = torch.mm(x_norm, recon_norm.t()) / self.temperature
        
        # create positive sample labels (diagonal)
        labels = torch.arange(x.size(0)).to(x.device)
        
        # calculate InfoNCE loss
        loss = F.cross_entropy(similarity, labels)
        
        if self.reduction == 'sum':
            return loss * x.size(0)
        else:
            return loss


class VAE(nn.Module):
    """
    Variational autoencoder model
    For unsupervised learning of one-dimensional feature data
    """
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout_rate=0.1):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # encoder
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
        
        # mean and variance layers
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # decoder
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
    
    def encode(self, x):
        """encoder: map input to mean and variance of latent space"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """reparameterization trick"""
        # limit logvar range to avoid numerical instability
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """decoder: map latent variable back to original space"""
        return self.decoder(z)
    
    def forward(self, x):
        """forward propagation"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def vae_loss(recon_x, x, mu, logvar, loss_function='mse', beta=1.0):
    """
    VAE loss function: reconstruction loss + KL divergence loss
    Supports multiple reconstruction loss functions
    """
    # reconstruction loss
    if loss_function.lower() == 'mse':
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    elif loss_function.lower() == 'bce':
        # for BCE, need to normalize data to [0,1] range
        recon_x_sigmoid = torch.sigmoid(recon_x)
        recon_loss = F.binary_cross_entropy(recon_x_sigmoid, torch.sigmoid(x), reduction='sum')
    elif loss_function.lower() == 'mae' or loss_function.lower() == 'l1':
        recon_loss = F.l1_loss(recon_x, x, reduction='sum')
    elif loss_function.lower() == 'smooth_l1':
        recon_loss = F.smooth_l1_loss(recon_x, x, reduction='sum')
    elif loss_function.lower() == 'huber':
        recon_loss = F.huber_loss(recon_x, x, reduction='sum', delta=1.0)
    elif loss_function.lower() == 'focal':
        # use Focal Loss
        focal_loss_fn = FocalLoss(alpha=1.0, gamma=2.0, reduction='sum')
        recon_loss = focal_loss_fn(recon_x, torch.sigmoid(x))
    elif loss_function.lower() == 'contrastive':
        # use contrastive loss (need to generate positive and negative sample pairs)
        contrastive_loss_fn = ContrastiveLoss(margin=1.0, reduction='sum')
        # create similar labels (here simplified to all samples are similar)
        labels = torch.ones(x.size(0)).to(x.device)
        recon_loss = contrastive_loss_fn(x, recon_x, labels)
    elif loss_function.lower() == 'spectral':
        # use spectral loss
        spectral_loss_fn = SpectralLoss(reduction='sum')
        recon_loss = spectral_loss_fn(x, recon_x)
    elif loss_function.lower() == 'wasserstein':
        # use Wasserstein distance loss
        wasserstein_loss_fn = WassersteinLoss(reduction='sum')
        recon_loss = wasserstein_loss_fn(x, recon_x)
    elif loss_function.lower() == 'perceptual':
        # use perceptual loss
        perceptual_loss_fn = PerceptualLoss(reduction='sum')
        recon_loss = perceptual_loss_fn(x, recon_x)
    elif loss_function.lower() == 'cosine':
        # use cosine similarity loss
        cosine_loss_fn = CosineSimilarityLoss(reduction='sum')
        recon_loss = cosine_loss_fn(x, recon_x)
    elif loss_function.lower() == 'kl_div':
        # use KL divergence loss (for reconstruction)
        kl_div_loss_fn = KLDivergenceLoss(reduction='sum')
        recon_loss = kl_div_loss_fn(x, recon_x)
    elif loss_function.lower() == 'regularization':
        # use regularization loss
        reg_loss_fn = RegularizationLoss(reduction='sum')
        recon_loss = reg_loss_fn(x, recon_x)
    elif loss_function.lower() == 'infonce':
        # use InfoNCE loss
        infonce_loss_fn = InfoNCELoss(temperature=0.07, reduction='sum')
        recon_loss = infonce_loss_fn(x, recon_x)
    else:
        # default use MSE
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence loss (this part must be calculated manually, PyTorch does not have a ready-made function)
    # limit logvar range to avoid numerical instability
    logvar = torch.clamp(logvar, min=-10, max=10)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # check if loss is finite
    if not torch.isfinite(recon_loss):
        print("Warning: Reconstruction loss contains NaN or infinite values, using default values")
        recon_loss = torch.tensor(1e6, device=recon_loss.device)
    
    if not torch.isfinite(kl_loss):
        print("Warning: Reconstruction loss contains NaN or infinite values, using default values")
        kl_loss = torch.tensor(1e6, device=kl_loss.device)
    
    total_loss = recon_loss + beta * kl_loss
    
    if not torch.isfinite(total_loss):
        print("Warning: Reconstruction loss contains NaN or infinite values, using default values")
        total_loss = torch.tensor(1e6, device=total_loss.device)
    
    return total_loss, recon_loss, kl_loss

class ModelEvaluator:

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
        all_latent_mu = []
        all_latent_logvar = []
        
        with torch.no_grad():
            for data, in data_loader:
                data = data.to(self.device)
                recon_data, mu, logvar = model(data)
                
                all_original.append(data.cpu().numpy())
                all_reconstructed.append(recon_data.cpu().numpy())
                all_latent_mu.append(mu.cpu().numpy())
                all_latent_logvar.append(logvar.cpu().numpy())
        
        original = np.vstack(all_original)
        reconstructed = np.vstack(all_reconstructed)
        latent_mu = np.vstack(all_latent_mu)
        latent_logvar = np.vstack(all_latent_logvar)
        
        # unstandardization
        original_unscaled = scaler.inverse_transform(original)
        reconstructed_unscaled = scaler.inverse_transform(reconstructed)
        
        # check and handle NaN and infinite values
        original_flat = original_unscaled.flatten()
        reconstructed_flat = reconstructed_unscaled.flatten()
        
        # remove NaN and infinite values
        valid_mask = np.isfinite(original_flat) & np.isfinite(reconstructed_flat)
        if not np.any(valid_mask):
            print("Warning: All data contains NaN or infinite values, using original data to calculate metrics")
            original_flat = original.flatten()
            reconstructed_flat = reconstructed.flatten()
            valid_mask = np.isfinite(original_flat) & np.isfinite(reconstructed_flat)
        
        if not np.any(valid_mask):
            print("Warning: No valid data found, returning default values")
            mse = float('inf')
            mae = float('inf')
            r2 = -float('inf')
            correlation = 0.0
        else:
            original_valid = original_flat[valid_mask]
            reconstructed_valid = reconstructed_flat[valid_mask]
            
            # calculate reconstruction metrics
            mse = mean_squared_error(original_valid, reconstructed_valid)
            mae = mean_absolute_error(original_valid, reconstructed_valid)
            r2 = r2_score(original_valid, reconstructed_valid)
        
        # calculate correlation coefficient
        if not np.any(valid_mask):
            correlation = 0.0
        else:
            try:
                correlation = np.corrcoef(original_valid, reconstructed_valid)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
            except:
                correlation = 0.0
        
        # calculate reconstruction error
        if not np.any(valid_mask):
            reconstruction_error = float('inf')
            reconstruction_accuracy = 0.0
        else:
            reconstruction_error = np.mean(np.abs(original_valid - reconstructed_valid))
            
            # calculate reconstruction accuracy (the proportion of error less than the threshold)
            threshold = 0.1 * np.std(original_valid)
            reconstruction_accuracy = np.mean(np.abs(original_valid - reconstructed_valid) < threshold)
        
        reconstruction_metrics = {
            'mse': mse,
            'mae': mae,
            'r2_score': r2,
            'correlation': correlation,
            'reconstruction_error': reconstruction_error,
            'reconstruction_accuracy': reconstruction_accuracy,
            'original_data': original_unscaled,
            'reconstructed_data': reconstructed_unscaled,
            'latent_mu': latent_mu,
            'latent_logvar': latent_logvar
        }
        
        return reconstruction_metrics
    
    def evaluate_latent_space(self, latent_mu, latent_logvar):
        """
        Evaluate latent space quality
        """
        # KL divergence
        kl_divergence = -0.5 * np.sum(1 + latent_logvar - latent_mu**2 - np.exp(latent_logvar), axis=1)
        
        # filter out infinite and NaN values
        kl_divergence_finite = kl_divergence[np.isfinite(kl_divergence)]
        
        if len(kl_divergence_finite) > 0:
            avg_kl_divergence = np.mean(kl_divergence_finite)
        else:
            avg_kl_divergence = 0.0
            print("Warning: All KL divergence values are invalid")
        
        # latent space dimension utilization
        latent_variance = np.var(latent_mu, axis=0)
        active_dimensions = np.sum(latent_variance > 0.01)
        dimension_utilization = active_dimensions / latent_mu.shape[1]
        
        # latent space continuity
        latent_distances = []
        for i in range(len(latent_mu) - 1):
            dist = np.linalg.norm(latent_mu[i+1] - latent_mu[i])
            if np.isfinite(dist):
                latent_distances.append(dist)
        
        if len(latent_distances) > 0:
            latent_continuity = np.mean(latent_distances)
        else:
            latent_continuity = 0.0
        
        # latent space distribution test (normality)
        normality_tests = []
        for i in range(latent_mu.shape[1]):
            try:
                _, p_value = stats.normaltest(latent_mu[:, i])
                if np.isfinite(p_value):
                    normality_tests.append(p_value)
            except:
                continue
        
        if len(normality_tests) > 0:
            normality_score = np.mean(normality_tests)
        else:
            normality_score = 0.0
        
        latent_metrics = {
            'avg_kl_divergence': avg_kl_divergence,
            'dimension_utilization': dimension_utilization,
            'active_dimensions': active_dimensions,
            'latent_continuity': latent_continuity,
            'normality_score': normality_score,
            'latent_variance': latent_variance,
            'kl_divergence_per_sample': kl_divergence
        }
        
        return latent_metrics
    
    def evaluate_generation_quality(self, model, n_samples=1000):
        """
        Evaluate generation quality
        """
        model.eval()
        
        with torch.no_grad():
            # sample from prior distribution
            z = torch.randn(n_samples, model.latent_dim).to(self.device)
            generated_samples = model.decode(z).cpu().numpy()
        
        # calculate statistical characteristics of generated samples
        generation_metrics = {
            'generated_samples': generated_samples,
            'generation_mean': np.mean(generated_samples),
            'generation_std': np.std(generated_samples),
            'generation_range': np.ptp(generated_samples)
        }
        
        return generation_metrics
    
    def compute_comprehensive_metrics(self, model, data_loader, scaler):
        """
        Calculate comprehensive evaluation indicators
        """
        print("Starting comprehensive model evaluation...")
        
        # reconstruction quality evaluation
        print("Evaluating reconstruction quality...")
        recon_metrics = self.evaluate_reconstruction(model, data_loader, scaler)
        
        # latent space evaluation
        print("Evaluating latent space quality...")
        latent_metrics = self.evaluate_latent_space(
            recon_metrics['latent_mu'], 
            recon_metrics['latent_logvar']
        )
        
        # generation quality evaluation
        print("Evaluating generation quality...")
        gen_metrics = self.evaluate_generation_quality(model)
        
        
        comprehensive_metrics = {
            'reconstruction': recon_metrics,
            'latent_space': latent_metrics,
            'generation': gen_metrics
        }
        
        return comprehensive_metrics
    


def visualize_results(metrics, save_path=None):
    """
    Visualize evaluation results
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('VAE model evaluation results', fontsize=16, fontweight='bold')
    
    # 1. reconstruction quality scatter plot
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
    ax1.set_title(f'Reconstruction quality (R² = {metrics["reconstruction"]["r2_score"]:.3f})')
    ax1.grid(True, alpha=0.3)
    
    # 2. Reconstruction error distribution
    ax2 = axes[0, 1]
    reconstruction_error = np.abs(original - reconstructed)
    ax2.hist(reconstruction_error, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Reconstruction error')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Reconstruction error distribution (MAE = {metrics["reconstruction"]["mae"]:.3f})')
    ax2.grid(True, alpha=0.3)
    
    # 3. Latent space visualization (t-SNE)
    ax3 = axes[1, 0]
    latent_mu = metrics['reconstruction']['latent_mu']
    
    # If latent dimension > 2, use t-SNE to reduce dimension
    if latent_mu.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=42)
        latent_2d = tsne.fit_transform(latent_mu[:1000])
    else:
        latent_2d = latent_mu[:1000]
    
    # Process KL divergence data for color mapping
    kl_div_for_plot = metrics['latent_space']['kl_divergence_per_sample'][:1000]
    kl_div_finite = kl_div_for_plot[np.isfinite(kl_div_for_plot)]
    
    if len(kl_div_finite) > 0:
        # Limit the range of KL divergence
        kl_div_clipped = np.clip(kl_div_for_plot, 0, np.percentile(kl_div_finite, 95))
        kl_div_clipped = np.where(np.isfinite(kl_div_clipped), kl_div_clipped, 0)
    else:
        kl_div_clipped = np.zeros(len(kl_div_for_plot))
    
    scatter = ax3.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                        c=kl_div_clipped, 
                        cmap='viridis', alpha=0.6)
    ax3.set_xlabel('t-SNE 1')
    ax3.set_ylabel('t-SNE 2')
    ax3.set_title('Latent space visualization')
    plt.colorbar(scatter, ax=ax3, label='KL divergence')
    
    
    
    # 4. Latent dimension utilization
    ax5 = axes[1, 1]
    latent_variance = metrics['latent_space']['latent_variance']
    dimensions = range(len(latent_variance))
    ax5.bar(dimensions, latent_variance, alpha=0.7, color='lightgreen')
    ax5.set_xlabel('Latent dimension')
    ax5.set_ylabel('Variance')
    ax5.set_title(f'Dimension utilization ({metrics["latent_space"]["active_dimensions"]}/{len(latent_variance)})')
    ax5.grid(True, alpha=0.3)
    
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # plt.show()
    plt.close()


def print_evaluation_report(metrics):
    """
    Print detailed evaluation report
    """
    print("\n" + "="*60)
    print("VAE model evaluation report")
    print("="*60)
    
    # Reconstruction quality
    print("\n Reconstruction quality evaluation:")
    print("-" * 30)
    print(f"Mean squared error (MSE): {metrics['reconstruction']['mse']:.6f}")
    print(f"Mean absolute error (MAE): {metrics['reconstruction']['mae']:.6f}")
    print(f"Coefficient of determination (R²): {metrics['reconstruction']['r2_score']:.6f}")
    print(f"Correlation coefficient: {metrics['reconstruction']['correlation']:.6f}")
    print(f"Reconstruction error: {metrics['reconstruction']['reconstruction_error']:.6f}")
    print(f"Reconstruction accuracy: {metrics['reconstruction']['reconstruction_accuracy']:.4f}")
    
    # Latent space quality
    print("\n Latent space quality evaluation:")
    print("-" * 30)
    print(f"Average KL divergence: {metrics['latent_space']['avg_kl_divergence']:.6f}")
    print(f"Dimension utilization: {metrics['latent_space']['dimension_utilization']:.4f}")
    print(f"Active dimensions: {metrics['latent_space']['active_dimensions']}")
    print(f"Latent space continuity: {metrics['latent_space']['latent_continuity']:.6f}")
    print(f"Normality score: {metrics['latent_space']['normality_score']:.6f}")
    
    # Generation quality
    print("\n Generation quality evaluation:")
    print("-" * 30)
    print(f"Generation sample mean: {metrics['generation']['generation_mean']:.6f}")
    print(f"Generation sample standard deviation: {metrics['generation']['generation_std']:.6f}")
    print(f"Generation sample range: {metrics['generation']['generation_range']:.6f}")
    
    

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


def prepare_data(seed, batch_size, ratio="8:1:1", jobid="20240808232043_OtJF37SH", kfold=0):
    """
    Prepare training, validation and test data
    The first row is column name
    """
    if ratio == "0" and kfold == 0: 
        scaler = StandardScaler()
        # Directly load training-validation-test set file
        # data = pd.read_csv(f"./train_example/{jobid}_data.txt", sep="\t").values.astype(float)   # the path of training dataset
        # data = pd.read_csv("D:\\wamp\www\\multi_omics_own\\download_data\\Jobs\\"+jobid+"\\"+jobid+"_data.txt",sep="\t")   # the path of training dataset
        data = pd.read_csv("../../data/train_exmaple_un/train_exmaple_un_data.txt", sep="\t")
        data = data.dropna().values.astype(float)
        data = scaler.fit_transform(data)
        data = torch.FloatTensor(data)
        train_loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)
        
        # data = pd.read_csv(f"./train_example/{jobid}_val.txt", sep="\t").values.astype(float)   # the path of training dataset
        # data = pd.read_csv("D:\\wamp\www\\multi_omics_own\\download_data\\Jobs\\"+jobid+"\\"+jobid+"_val.txt",sep="\t")  # the path of training dataset
        data = pd.read_csv("../../data/train_exmaple_un/train_exmaple_un_val.txt", sep="\t")
        data = data.dropna().values.astype(float) 
        data = scaler.transform(data)
        data = torch.FloatTensor(data)
        val_loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)

        # data = pd.read_csv(f"./train_example/{jobid}_test.txt", sep="\t").values.astype(float)   # the path of training dataset
        # data = pd.read_csv("D:\\wamp\www\\multi_omics_own\\download_data\\Jobs\\"+jobid+"\\"+jobid+"_test.txt",sep="\t")   # the path of training dataset
        data = pd.read_csv("../../data/train_exmaple_un/train_exmaple_un_test.txt", sep="\t")
        data = data.dropna().values.astype(float) 
        data = scaler.transform(data)
        data = torch.FloatTensor(data)
        test_loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)

        
    elif ratio != "0" and kfold == 0:  
        # Split training-validation-test set
        # data = pd.read_csv("D:\\wamp\www\\multi_omics_own\\download_data\\Jobs\\"+jobid+"\\"+jobid+"_data.txt",sep="\t")   # the path of training dataset
        data = pd.read_csv("../../data/train_exmaple_un/train_exmaple_un_data.txt", sep="\t")
        data = data.dropna().values.astype(float) 
        # data = pd.read_csv(f"./train_example/{jobid}_data.txt", sep="\t")  # the path of dataset
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
    




def train_epoch(model, train_loader, optimizer, device, loss_function='mse'):
    """Train one epoch"""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    for batch_idx, (data,) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, logvar = model(data)
        loss, recon_loss, kl_loss = vae_loss(recon_batch, data, mu, logvar, loss_function)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
    
    return total_loss / len(train_loader), total_recon_loss / len(train_loader), total_kl_loss / len(train_loader)

def validate_epoch(model, val_loader, device, loss_function='mse'):
    """Validate one epoch"""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    with torch.no_grad():
        for data, in val_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss, recon_loss, kl_loss = vae_loss(recon_batch, data, mu, logvar, loss_function)
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
    
    return total_loss / len(val_loader), total_recon_loss / len(val_loader), total_kl_loss / len(val_loader)

def train_model(model, train_loader, val_loader, optimizer, device, epochs, patience, model_name, loss_function='mse'):
    """Train model"""
    early_stopping = EarlyStopping(patience=patience)
    train_losses = []
    val_losses = []
    
    print(f"Start training model: {model_name}")
    print(f"Use loss function: {loss_function}")
    print("-" * 50)
    
    for epoch in range(epochs):
        train_loss, train_recon, train_kl = train_epoch(model, train_loader, optimizer, device, loss_function)
        val_loss, val_recon, val_kl = validate_epoch(model, val_loader, device, loss_function)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  train loss: {train_loss:.4f} (reconstruction: {train_recon:.4f}, KL: {train_kl:.4f})')
        print(f'  validation loss: {val_loss:.4f} (reconstruction: {val_recon:.4f}, KL: {val_kl:.4f})')
        
        if early_stopping(val_loss, model):
            print(f'Early stopping triggered, stopping training at epoch {epoch+1}')
            break
    
    return train_losses, val_losses

def test_model(model, test_loader, device, loss_function='mse'):
    """Test model"""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    with torch.no_grad():
        for data, in test_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss, recon_loss, kl_loss = vae_loss(recon_batch, data, mu, logvar, loss_function)
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
    
    avg_loss = total_loss / len(test_loader)
    avg_recon_loss = total_recon_loss / len(test_loader)
    avg_kl_loss = total_kl_loss / len(test_loader)
    
    print(f"Test results:")
    print(f"  Total loss: {avg_loss:.4f}")
    print(f"  Reconstruction loss: {avg_recon_loss:.4f}")
    print(f"  KL divergence loss: {avg_kl_loss:.4f}")
    
    return avg_loss, avg_recon_loss, avg_kl_loss


def kfold_cross_validation(args, k_folds=5):
    """k-fold cross validation"""
    print(f"\nStart {k_folds}-fold cross validation")
    print("=" * 60)

    dim, train_loader, _, test_loader, scaler = prepare_data(
        args.random_seed, args.batch_size, args.ratio, args.jobid, k_folds
    )
    
    
    # Get training data for k-fold
    train_data_list = []
    for batch_data, in train_loader:
        train_data_list.append(batch_data.numpy())
    train_data_scaled = np.vstack(train_data_list)
    
    # Perform k-fold cross validation on training data
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
        model = VAE(
            input_dim=dim,
            hidden_dims=[512, 256, 128],
            latent_dim=args.latent_dim,
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
            args.epochs, args.early_stopping_patience, f"Fold_{fold+1}", args.loss_function
        )
        
        # Test model on validation set
        test_loss, test_recon, test_kl = test_model(model, fold_val_loader, device, args.loss_function)
        fold_results.append({
            'fold': fold + 1,
            'test_loss': test_loss,
            'test_recon_loss': test_recon,
            'test_kl_loss': test_kl,
            'train_losses': train_losses,
            'val_losses': val_losses
        })
    
    # Calculate average result
    avg_test_loss = np.mean([r['test_loss'] for r in fold_results])
    avg_test_recon = np.mean([r['test_recon_loss'] for r in fold_results])
    avg_test_kl = np.mean([r['test_kl_loss'] for r in fold_results])
    
    print(f"\n{k_folds}-fold cross validation results:")
    print("=" * 40)
    print(f"Average validation loss: {avg_test_loss:.4f} ± {np.std([r['test_loss'] for r in fold_results]):.4f}")
    print(f"Average reconstruction loss: {avg_test_recon:.4f} ± {np.std([r['test_recon_loss'] for r in fold_results]):.4f}")
    print(f"Average KL divergence loss: {avg_test_kl:.4f} ± {np.std([r['test_kl_loss'] for r in fold_results]):.4f}")
    
    return fold_results, model, scaler, test_loader, dim


def save_model(model, scaler, args, model_path, scaler_path, input_dim):
    """Save model and preprocessor"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': input_dim,
            'hidden_dims': [512, 256, 128],
            'latent_dim': args.latent_dim,
            'dropout_rate': args.dropout
        },
        'args': vars(args)
    }, model_path)
    
    # Save scaler
    import pickle
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)


def load_model(model_path, scaler_path, device):
    """Load model and preprocessor"""
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['model_config']
    
    model = VAE(
        input_dim=model_config['input_dim'],
        hidden_dims=model_config['hidden_dims'],
        latent_dim=model_config['latent_dim'],
        dropout_rate=model_config['dropout_rate']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load scaler
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
        recon_data, mu, logvar = model(data_tensor)
    
    return recon_data.cpu().numpy(), mu.cpu().numpy(), logvar.cpu().numpy()

def load_and_predict(model_path, scaler_path, new_data, device, args=None):
    """
    Load model and predict new data
    """
    print("Loading model and preprocessor...")
    
    # Load model
    model, scaler = load_model(model_path, scaler_path, device)
    
    print(f"Model loaded, input dimension: {model.input_dim}")
    print(f"New data shape: {new_data.shape}")
    
    # Check data dimension
    if new_data.shape[1] != model.input_dim:
        raise ValueError(f"Data dimension mismatch! Model expects {model.input_dim} dimensions, but input data has {new_data.shape[1]} dimensions")
    
    # Predict
    print("Start predicting...")
    recon_data, mu, logvar = predict(model, scaler, new_data, device)
    
    # Unscale reconstructed data
    recon_data_unscaled = scaler.inverse_transform(recon_data)
    
    print("Prediction completed!")
    
    return {
        'original_data': new_data,
        'reconstructed_data': recon_data_unscaled,
        'latent_mu': mu,
        'latent_logvar': logvar,
        'model': model,
        'scaler': scaler
    }

def evaluate_predictions(prediction_results, save_path=None):
    """
    Evaluate prediction results
    """
    print("\n" + "="*60)
    print("Prediction results evaluation")
    print("="*60)
    
    original = prediction_results['original_data']
    reconstructed = prediction_results['reconstructed_data']
    latent_mu = prediction_results['latent_mu']
    latent_logvar = prediction_results['latent_logvar']
    
    # Calculate reconstruction metrics
    mse = mean_squared_error(original.flatten(), reconstructed.flatten())
    mae = mean_absolute_error(original.flatten(), reconstructed.flatten())
    r2 = r2_score(original.flatten(), reconstructed.flatten())
    correlation = np.corrcoef(original.flatten(), reconstructed.flatten())[0, 1]
    
    # Calculate reconstruction error
    reconstruction_error = np.mean(np.abs(original - reconstructed))
    
    # Calculate reconstruction accuracy
    threshold = 0.1 * np.std(original)
    reconstruction_accuracy = np.mean(np.abs(original - reconstructed) < threshold)
    
    # Calculate KL divergence
    kl_divergence = -0.5 * np.sum(1 + latent_logvar - latent_mu**2 - np.exp(latent_logvar), axis=1)
    avg_kl_divergence = np.mean(kl_divergence)
    
    # Calculate latent space quality
    latent_variance = np.var(latent_mu, axis=0)
    active_dimensions = np.sum(latent_variance > 0.01)
    dimension_utilization = active_dimensions / latent_mu.shape[1]
    
    # Print evaluation results
    print("\n Reconstruction quality evaluation:")
    print("-" * 30)
    print(f"Mean squared error (MSE): {mse:.6f}")
    print(f"Mean absolute error (MAE): {mae:.6f}")
    print(f"Coefficient of determination (R²): {r2:.6f}")
    print(f"Correlation coefficient: {correlation:.6f}")
    print(f"Reconstruction error: {reconstruction_error:.6f}")
    print(f"Reconstruction accuracy: {reconstruction_accuracy:.4f}")
    
    print("\n Latent space quality evaluation:")
    print("-" * 30)
    print(f"Average KL divergence: {avg_kl_divergence:.6f}")
    print(f"Dimension utilization: {dimension_utilization:.4f}")
    print(f"Active dimensions: {active_dimensions}")
    
    
    # Save evaluation results
    if save_path:
        evaluation_data = {
            'reconstruction_metrics': {
                'mse': float(mse),
                'mae': float(mae),
                'r2_score': float(r2),
                'correlation': float(correlation),
                'reconstruction_error': float(reconstruction_error),
                'reconstruction_accuracy': float(reconstruction_accuracy)
            },
            'latent_space_metrics': {
                'avg_kl_divergence': float(avg_kl_divergence),
                'dimension_utilization': float(dimension_utilization),
                'active_dimensions': int(active_dimensions)
            }
        }
        
        json_path = save_path.replace('.png', '.json') if save_path.endswith('.png') else save_path + '.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
    
    return {
        'reconstruction_metrics': {
            'mse': mse,
            'mae': mae,
            'r2_score': r2,
            'correlation': correlation,
            'reconstruction_error': reconstruction_error,
            'reconstruction_accuracy': reconstruction_accuracy
        },
        'latent_space_metrics': {
            'avg_kl_divergence': avg_kl_divergence,
            'dimension_utilization': dimension_utilization,
            'active_dimensions': active_dimensions
        }
    }

def visualize_predictions(prediction_results, save_path=None):
    """
    Visualize prediction results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('VAE prediction results visualization', fontsize=16, fontweight='bold')
    
    original = prediction_results['original_data']
    reconstructed = prediction_results['reconstructed_data']
    latent_mu = prediction_results['latent_mu']
    latent_logvar = prediction_results['latent_logvar']
    
    # 1. Reconstruction quality scatter plot
    ax1 = axes[0, 0]
    original_flat = original.flatten()
    reconstructed_flat = reconstructed.flatten()
    
    if len(original_flat) > 1000:
        indices = np.random.choice(len(original_flat), 1000, replace=False)
        original_sample = original_flat[indices]
        reconstructed_sample = reconstructed_flat[indices]
    else:
        original_sample = original_flat
        reconstructed_sample = reconstructed_flat
    
    ax1.scatter(original_sample, reconstructed_sample, alpha=0.5, s=1)
    ax1.plot([original_sample.min(), original_sample.max()], 
             [original_sample.min(), original_sample.max()], 'r--', lw=2)
    ax1.set_xlabel('Original data')
    ax1.set_ylabel('Reconstructed data')
    ax1.set_title('Reconstruction quality scatter plot')
    ax1.grid(True, alpha=0.3)
    
    # 2. Reconstruction error distribution
    ax2 = axes[0, 1]
    reconstruction_error = np.abs(original - reconstructed)
    ax2.hist(reconstruction_error.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Reconstruction error')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Reconstruction error distribution')
    ax2.grid(True, alpha=0.3)
    
    # 3. Latent space visualization (t-SNE)
    ax3 = axes[1, 0]
    if latent_mu.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=42)
        latent_2d = tsne.fit_transform(latent_mu[:1000])
    else:
        latent_2d = latent_mu[:1000]
    
    scatter = ax3.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                        c=np.arange(len(latent_2d)), cmap='viridis', alpha=0.6)
    ax3.set_xlabel('t-SNE 1')
    ax3.set_ylabel('t-SNE 2')
    ax3.set_title('Latent space visualization')
    plt.colorbar(scatter, ax=ax3, label='Sample index')
    
    # 4. KL divergence distribution
    ax4 = axes[1, 1]
    kl_div = -0.5 * np.sum(1 + latent_logvar - latent_mu**2 - np.exp(latent_logvar), axis=1)
    ax4.hist(kl_div, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    ax4.set_xlabel('KL divergence')
    ax4.set_ylabel('Frequency')
    ax4.set_title('KL divergence distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # plt.show()
    plt.close()



def main():
    parser = argparse.ArgumentParser(description='VAE training script')

    # Data path
    parser.add_argument('--ratio', type=str, default='0', help='Data split ratio')
    parser.add_argument('--jobid', type=str, default='train_example_un', help='Data set ID')
    
    # Model parameters
    parser.add_argument('--latent_dim', type=int, default=20, help='Latent space dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience')
    
    # Optimizer and loss function
    parser.add_argument('--optimizer_function', type=str, default='adam', 
                       choices=['adam', 'sgd', 'adamw'], help='Optimizer type')
    parser.add_argument('--loss_function', type=str, default='mse', 
                       choices=['mse', 'bce', 'mae', 'l1', 'smooth_l1', 'huber', 'focal', 
                               'contrastive', 'spectral', 'wasserstein', 'perceptual', 
                               'cosine', 'kl_div', 'regularization', 'infonce'], 
                       help='Reconstruction loss function type:\n'
                            'Basic loss: mse, bce, mae/l1, smooth_l1, huber\n'
                            'Advanced loss: focal, contrastive, spectral, wasserstein\n'
                            'Similarity loss: perceptual, cosine, kl_div, infonce(InfoNCE)\n'
                            'Regularization: regularization')
    
    # Other parameters
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--k_folds', type=int, default=0, help='k-fold number')
    parser.add_argument('--save_model', action='store_true', default=1, help='Save model')
    parser.add_argument('--model_path', type=str, default='vae_model.pth', help='Model save path')
    parser.add_argument('--scaler_path', type=str, default='scaler.pkl', help='Preprocessor save path')
    
    # Evaluation related parameters
    parser.add_argument('--evaluate_model', action='store_true', default=1, help='Evaluate model')
    parser.add_argument('--save_evaluation', action='store_true', default=1, help='Save evaluation results')
    parser.add_argument('--evaluation_path', type=str, default='evaluation_results.png', help='Evaluation results save path')
    parser.add_argument('--show_plots', action='store_true', default=1, help='Show plots')
    
    args = parser.parse_args()

    print("VAE model")
    print('ratio =', args.ratio)
    print('k_folds =', args.k_folds)
    print('latent_dim =', args.latent_dim)
    print('dropout =', args.dropout)
    print('epochs =', args.epochs)
    print('batch_size =', args.batch_size)
    print('learning_rate =', args.learning_rate)
    print('early_stopping_patience =', args.early_stopping_patience)
    print('optimizer_function =', args.optimizer_function)
    print('loss_function =', args.loss_function)
    print('random_seed =', args.random_seed)
    
    # Set random seed
    set_random_seed(args.random_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    
    # if args.use_kfold:
    if args.k_folds:
        # k-fold cross validation
        _, model, scaler, test_loader, dim = kfold_cross_validation(args, args.k_folds)
    else:
        # Regular training-validation-test
        print("\nStart regular training process")
        print("=" * 50)
        
        dim, train_loader, val_loader, test_loader, scaler = prepare_data(
            args.random_seed, args.batch_size, args.ratio, args.jobid, args.k_folds
        )
        
        # Create model
        model = VAE(
            input_dim=dim,
            hidden_dims=[512, 256, 128],
            latent_dim=args.latent_dim,
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
            args.epochs, args.early_stopping_patience, "VAE_Model", args.loss_function
        )
        
        # Test model
        test_loss, test_recon, test_kl = test_model(model, test_loader, device, args.loss_function)
        
    # Save model
    if args.save_model:
        save_model(model, scaler, args, args.model_path, args.scaler_path, dim)
    
    
    # Model evaluation
    if args.evaluate_model:
        print("\n" + "="*60)
        print("Start model evaluation")
        print("="*60)
        
        # Create evaluator
        evaluator = ModelEvaluator(device)
        
        # Comprehensive evaluation test model
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
                'reconstruction_metrics': {
                    'mse': float(metrics['reconstruction']['mse']),
                    'mae': float(metrics['reconstruction']['mae']),
                    'r2_score': float(metrics['reconstruction']['r2_score']),
                    'correlation': float(metrics['reconstruction']['correlation']),
                    'reconstruction_error': float(metrics['reconstruction']['reconstruction_error']),
                    'reconstruction_accuracy': float(metrics['reconstruction']['reconstruction_accuracy'])
                },
                'latent_space_metrics': {
                    'avg_kl_divergence': float(metrics['latent_space']['avg_kl_divergence']),
                    'dimension_utilization': float(metrics['latent_space']['dimension_utilization']),
                    'active_dimensions': int(metrics['latent_space']['active_dimensions']),
                    'latent_continuity': float(metrics['latent_space']['latent_continuity']),
                    'normality_score': float(metrics['latent_space']['normality_score'])
                },
                'generation_metrics': {
                    'generation_mean': float(metrics['generation']['generation_mean']),
                    'generation_std': float(metrics['generation']['generation_std']),
                    'generation_range': float(metrics['generation']['generation_range'])
                }
            }
            
            json_path = args.evaluation_path.replace('.png', '.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # cmd: pythonVAE.py --ratio 0 
    # cmd: pythonVAE.py --ratio 8:1:1 --k_folds 5
    # Output: evaluation_results.json,evaluation_results.png, vae_model.pth, scaler.pkl
    main()
