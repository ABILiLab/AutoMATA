import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from train.deepcluster import load_model, predict, set_random_seed 
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_and_predict(model_path, scaler_path, new_data, device, args=None):
    """
    Load model and predict new data
    """
    print("Loading model and preprocessor...")
    
    # Load model
    model, scaler = load_model(model_path, scaler_path, device)
    
    print(f"Model loaded, input dimension: {model.input_dim}")
    print(f"Cluster number: {model.n_clusters}")
    print(f"New data shape: {new_data.shape}")
    
    # Check data dimension
    if new_data.shape[1] != model.input_dim:
        raise ValueError(f"Data dimension mismatch! Model expects {model.input_dim} dimensions, but input data has {new_data.shape[1]} dimensions")
    
    # Predict
    print("Predicting...")
    recon_data, latent, cluster_assignments = predict(model, scaler, new_data, device)
    
    # Unstandardize reconstructed data
    recon_data_unscaled = scaler.inverse_transform(recon_data)
    
    print("Prediction completed!")
    
    return {
        'original_data': new_data,
        'reconstructed_data': recon_data_unscaled,
        'latent_representations': latent,
        'cluster_assignments': cluster_assignments,
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
    latent = prediction_results['latent_representations']
    cluster_assignments = prediction_results['cluster_assignments']
    
    # Calculate reconstruction indicators
    mse = mean_squared_error(original.flatten(), reconstructed.flatten())
    mae = mean_absolute_error(original.flatten(), reconstructed.flatten())
    r2 = r2_score(original.flatten(), reconstructed.flatten())
    correlation = np.corrcoef(original.flatten(), reconstructed.flatten())[0, 1]
    
    # Calculate reconstruction error
    reconstruction_error = np.mean(np.abs(original - reconstructed))
    
    # Calculate reconstruction accuracy
    threshold = 0.1 * np.std(original)
    reconstruction_accuracy = np.mean(np.abs(original - reconstructed) < threshold)
    
    # Calculate clustering indicators
    n_clusters = len(np.unique(cluster_assignments))
    cluster_sizes = np.bincount(cluster_assignments)
    
    # Calculate intra-cluster distance
    intra_cluster_distances = []
    for i in range(n_clusters):
        cluster_mask = (cluster_assignments == i)
        if cluster_mask.sum() > 1:
            cluster_points = latent[cluster_mask]
            distances = np.linalg.norm(cluster_points - cluster_points.mean(axis=0), axis=1)
            intra_cluster_distances.extend(distances)
    
    avg_intra_cluster_distance = np.mean(intra_cluster_distances) if intra_cluster_distances else 0
    
    # Calculate inter-cluster distance
    inter_cluster_distances = []
    cluster_centers = []
    for i in range(n_clusters):
        cluster_mask = (cluster_assignments == i)
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
    if len(np.unique(cluster_assignments)) > 1:
        silhouette_avg = silhouette_score(latent, cluster_assignments)
    else:
        silhouette_avg = 0
    
    # Calculate cluster balance
    cluster_balance = 1 - np.std(cluster_sizes) / np.mean(cluster_sizes) if np.mean(cluster_sizes) > 0 else 0
    
    # Print evaluation results
    print("\n Reconstruction quality evaluation:")
    print("-" * 30)
    print(f"Mean squared error (MSE): {mse:.6f}")
    print(f"Mean absolute error (MAE): {mae:.6f}")
    print(f"R²: {r2:.6f}")
    print(f"Correlation coefficient: {correlation:.6f}")
    print(f"Reconstruction error: {reconstruction_error:.6f}")
    print(f"Reconstruction accuracy: {reconstruction_accuracy:.4f}")
    
    print("\n Clustering quality evaluation:")
    print("-" * 30)
    print(f"Cluster number: {n_clusters}")
    print(f"Silhouette coefficient: {silhouette_avg:.6f}")
    print(f"Cluster balance: {cluster_balance:.6f}")
    print(f"Average intra-cluster distance: {avg_intra_cluster_distance:.6f}")
    print(f"Average inter-cluster distance: {avg_inter_cluster_distance:.6f}")
    print(f"Cluster size: {cluster_sizes}")
    
    
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
            'clustering_metrics': {
                'n_clusters': int(n_clusters),
                'silhouette_score': float(silhouette_avg),
                'cluster_balance': float(cluster_balance),
                'avg_intra_cluster_distance': float(avg_intra_cluster_distance),
                'avg_inter_cluster_distance': float(avg_inter_cluster_distance)
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
        'clustering_metrics': {
            'n_clusters': n_clusters,
            'silhouette_score': silhouette_avg,
            'cluster_balance': cluster_balance,
            'avg_intra_cluster_distance': avg_intra_cluster_distance,
            'avg_inter_cluster_distance': avg_inter_cluster_distance
        }
    }

def visualize_predictions(prediction_results, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Deep clustering prediction results visualization', fontsize=16, fontweight='bold')
    
    original = prediction_results['original_data']
    reconstructed = prediction_results['reconstructed_data']
    latent = prediction_results['latent_representations']
    cluster_assignments = prediction_results['cluster_assignments']
    
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
    
    # 3. Clustering result visualization (t-SNE)
    ax3 = axes[1, 0]
    if latent.shape[1] > 2:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42)
        latent_2d = tsne.fit_transform(latent[:1000])
    else:
        latent_2d = latent[:1000]
    
    scatter = ax3.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                        c=cluster_assignments[:1000], cmap='tab10', alpha=0.6)
    ax3.set_xlabel('t-SNE 1')
    ax3.set_ylabel('t-SNE 2')
    ax3.set_title('Clustering result visualization')
    plt.colorbar(scatter, ax=ax3, label='Clustering labels')
    
    # 4. Clustering size distribution
    ax4 = axes[1, 1]
    cluster_sizes = np.bincount(cluster_assignments)
    ax4.bar(range(len(cluster_sizes)), cluster_sizes, alpha=0.7, color='lightcoral')
    ax4.set_xlabel('Clustering labels')
    ax4.set_ylabel('Sample number')
    ax4.set_title('Clustering size distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # plt.show()
    plt.close()

if __name__ == "__main__":
    """
    cmd:  python predict_deepcluster.py
    output: prediction_results_clusters.txt, prediction_results.png, prediction_results.json
    """
    parser = argparse.ArgumentParser(description='Deep clustering model prediction script')
    
    # model path
    parser.add_argument('--model_path', type=str, default='deepcluster_model.pth', help='model file path')
    parser.add_argument('--scaler_path', type=str, default='deepcluster_scaler.pkl', help='preprocessor file path')
    
    # data path
    parser.add_argument('--data_path', type=str, default='../../data/train_exmaple_un/train_exmaple_un_test.txt', help='new data file path (tab separated)')
    
    # evaluation options
    parser.add_argument('--evaluate', action='store_true', default=1, help='whether to evaluate the prediction')
    parser.add_argument('--visualize', action='store_true', default=0, help='whether to visualize the prediction')
    parser.add_argument('--save_results', action='store_true', default=1, help='whether to save the prediction')
    parser.add_argument('--output_path', type=str, default='prediction_results', help='output file path prefix')
    
    # other parameters
    parser.add_argument('--random_seed', type=int, default=42, help='random seed')
    
    args = parser.parse_args()
    
    # set random seed
    set_random_seed(args.random_seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # prepare data
    if args.data_path:
        print(f"Load data from the file: {args.data_path}")
        import pandas as pd
        data = pd.read_csv(args.data_path, sep="\t")
        data = data.dropna().values
    print(f"data shape: {data.shape}")
    
    # load model and predict
    try:
        prediction_results = load_and_predict(
            args.model_path, args.scaler_path, data, device, args
        )
    except Exception as e:
        print(f"Prediction failed: {e}")
        exit(0)
    
    # evaluate prediction results
    if args.evaluate:
        metrics = evaluate_predictions(
            prediction_results, 
            args.output_path if args.save_results else None
        )
    
    # visualize results
    if args.visualize:
        visualize_predictions(
            prediction_results,
            args.output_path + '.png' if args.save_results else None
        )
    
    # save prediction results
    if args.save_results:

        # save each sample's cluster labels (tab separated)
        clusters_txt_path = args.output_path + '_clusters.txt'
        with open(clusters_txt_path, 'w', encoding='utf-8') as f:
            f.write('index\tcluster\n')
            for idx, c in enumerate(prediction_results['cluster_assignments']):
                f.write(str(idx) + '\t' + str(int(c)) + '\n')



