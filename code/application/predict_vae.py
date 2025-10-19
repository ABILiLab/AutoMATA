import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from train.VAE import load_and_predict, evaluate_predictions, visualize_predictions, set_random_seed
import torch
import numpy as np
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    """
    cmd: python predict_vae.py
    output: prediction_results_*.npy, prediction_results.png, prediction_results.json
    """
    parser = argparse.ArgumentParser(description='VAE prediction script')
    
    # model path
    parser.add_argument('--model_path', type=str, default='vae_model.pth', help='model file path')
    parser.add_argument('--scaler_path', type=str, default='scaler.pkl', help='preprocessor file path')
    
    # data path
    parser.add_argument('--data_path', type=str, default='../../data/train_exmaple_un/train_exmaple_un_test.txt', help='new data file path (tab separated)')
    # parser.add_argument('--data_shape', type=str, default='50,100', help='data shape, format: "samples,features" (e.g.: "100,50")')
    
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
    
    # Prepare data
    if args.data_path:
        # Load data from file
        print(f"Load data from file: {args.data_path}")
        import pandas as pd
        data = pd.read_csv(args.data_path, sep="\t")
        data = data.dropna().values
    print(f"Data shape: {data.shape}")
    
    # Load model and predict
    try:
        prediction_results = load_and_predict(
            args.model_path, args.scaler_path, data, device, args
        )
    except Exception as e:
        print(f"Prediction failed: {e}")
        exit(0)
    
    # Evaluate prediction results
    if args.evaluate:
        metrics = evaluate_predictions(
            prediction_results, 
            args.output_path if args.save_results else None
        )
    
    # Visualize results
    if args.visualize:
        visualize_predictions(
            prediction_results,
            args.output_path + '.png' if args.save_results else None
        )
    
    # Save prediction results
    if args.save_results:
        # Save reconstructed data
        np.save(args.output_path + '_reconstructed.npy', prediction_results['reconstructed_data'])
        np.save(args.output_path + '_latent_mu.npy', prediction_results['latent_mu'])
        np.save(args.output_path + '_latent_logvar.npy', prediction_results['latent_logvar'])
    
    

