"""
Testing/Inference Script for Seismic Impedance Inversion

This script performs inference with trained models and visualizes results.

Command-line usage examples:
    # Test 1D CNN
    python test.py --model 1d --checkpoint ./models/1d_best.pth --num_samples 10
    
    # Test 2D CNN
    python test.py --model 2d --checkpoint ./models/2d_best.pth --num_samples 5
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm

from model import SeismicImpedanceCNN1D, SeismicImpedanceCNN2D
from data_loading import get_dataloader
from utils import (load_checkpoint, calculate_metrics, plot_1d_results, 
                   plot_2d_results, set_seed)


def test_1d(model, dataloader, device, save_dir, num_samples_to_plot=5):
    """
    Test 1D model and generate visualizations.
    
    Args:
        model: Trained 1D CNN model
        dataloader: Test dataloader
        device: Device to run inference on
        save_dir: Directory to save results
        num_samples_to_plot: Number of samples to visualize
    
    Returns:
        Average metrics dictionary
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    print("\nRunning inference...")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader)):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Store results
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
            # Plot first few samples
            if batch_idx == 0:
                batch_size = inputs.size(0)
                num_to_plot = min(num_samples_to_plot, batch_size)
                
                for i in range(num_to_plot):
                    # Extract data
                    seismic = inputs[i, 0].cpu().numpy()
                    initial_impedance = inputs[i, 1].cpu().numpy() if inputs.size(1) > 1 else None
                    true_impedance = targets[i, 0].cpu().numpy()
                    pred_impedance = outputs[i, 0].cpu().numpy()
                    
                    # Use zeros if no initial impedance
                    if initial_impedance is None:
                        initial_impedance = np.zeros_like(seismic)
                    
                    # Plot
                    save_path = os.path.join(save_dir, f'1d_sample_{i+1}.png')
                    plot_1d_results(seismic, initial_impedance, true_impedance, 
                                    pred_impedance, save_path)
    
    # Concatenate all results
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(all_predictions.flatten(), all_targets.flatten())
    
    return metrics


def test_2d(model, dataloader, device, save_dir, num_samples_to_plot=5):
    """
    Test 2D model and generate visualizations.
    
    Args:
        model: Trained 2D CNN model
        dataloader: Test dataloader
        device: Device to run inference on
        save_dir: Directory to save results
        num_samples_to_plot: Number of samples to visualize
    
    Returns:
        Average metrics dictionary (evaluated at well locations)
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_masks = []
    
    print("\nRunning inference...")
    with torch.no_grad():
        for batch_idx, (inputs, targets, masks) in enumerate(tqdm(dataloader)):
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Store results
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_masks.append(masks.cpu().numpy())
            
            # Plot first few samples
            if batch_idx == 0:
                batch_size = inputs.size(0)
                num_to_plot = min(num_samples_to_plot, batch_size)
                
                for i in range(num_to_plot):
                    # Extract data
                    seismic = inputs[i, 0].cpu().numpy()
                    initial_impedance = inputs[i, 1].cpu().numpy()
                    true_impedance = targets[i, 0].cpu().numpy()
                    pred_impedance = outputs[i, 0].cpu().numpy()
                    mask = masks[i, 0].cpu().numpy()
                    
                    # Plot
                    save_path = os.path.join(save_dir, f'2d_sample_{i+1}.png')
                    plot_2d_results(seismic, initial_impedance, true_impedance,
                                    pred_impedance, mask, save_path)
    
    # Concatenate all results
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    
    # Calculate metrics (at well locations only)
    print("\nCalculating metrics (at well locations)...")
    metrics = calculate_metrics(
        all_predictions.flatten(), 
        all_targets.flatten(),
        mask=all_masks.flatten()
    )
    
    # Also calculate metrics on full field
    print("\nCalculating metrics (full field)...")
    metrics_full = calculate_metrics(
        all_predictions.flatten(), 
        all_targets.flatten()
    )
    
    # Combine metrics
    combined_metrics = {
        'well_locations': metrics,
        'full_field': metrics_full
    }
    
    return combined_metrics


def test(args):
    """
    Main testing function.
    
    Args:
        args: Command-line arguments
    """
    # Set random seed
    set_seed(args.seed)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Create model
    if args.model == '1d':
        model = SeismicImpedanceCNN1D(
            in_channels=2 if args.use_initial_impedance else 1,
            num_filters=args.num_filters,
            num_blocks=args.num_blocks
        )
        test_fn = test_1d
    else:  # 2d
        model = SeismicImpedanceCNN2D(
            in_channels=2,
            num_filters=args.num_filters,
            num_blocks=args.num_blocks
        )
        test_fn = test_2d
    
    model = model.to(device)
    print(f"\nModel: {args.model.upper()} CNN")
    
    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    load_checkpoint(model, None, args.checkpoint, device)
    
    # Create test dataloader
    print("\nCreating test dataset...")
    
    # Common dataset parameters
    common_kwargs = {
        'wavelet_freq': args.wavelet_freq,
        'velocity': args.velocity,
        'noise_level': args.noise_level,
        'smoothing_sigma': args.smoothing_sigma
    }
    
    # Model-specific parameters
    if args.model == '1d':
        dataset_kwargs = {
            **common_kwargs,
            'trace_length': args.trace_length,
            'dz': args.dz,
            'use_initial_impedance': args.use_initial_impedance
        }
    else:  # 2d
        dataset_kwargs = {
            **common_kwargs,
            'profile_height': args.profile_height,
            'profile_width': args.profile_width,
            'dz': args.dz,
            'dx': args.dx,
            'num_wells': args.num_wells
        }
    
    test_loader = get_dataloader(
        dataset_type=args.model,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        shuffle=False,
        num_workers=args.num_workers,
        **dataset_kwargs
    )
    
    print(f"Test samples: {args.num_samples}")
    
    # Run inference
    metrics = test_fn(model, test_loader, device, args.results_dir, 
                      num_samples_to_plot=args.num_plots)
    
    # Print metrics
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    
    if args.model == '2d':
        print("\nMetrics at well locations:")
        for key, value in metrics['well_locations'].items():
            print(f"  {key}: {value:.6f}")
        
        print("\nMetrics on full field:")
        for key, value in metrics['full_field'].items():
            print(f"  {key}: {value:.6f}")
    else:
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")
    
    print(f"\nResults saved to: {args.results_dir}")


def main():
    parser = argparse.ArgumentParser(description='Test Seismic Impedance Inversion Models')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='1d', choices=['1d', '2d'],
                        help='Model type: 1d or 2d')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--num_filters', type=int, default=16,
                        help='Number of filters in conv layers')
    parser.add_argument('--num_blocks', type=int, default=4,
                        help='Number of convolutional blocks')
    
    # Testing configuration
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of test samples')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num_plots', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Data configuration
    parser.add_argument('--trace_length', type=int, default=300,
                        help='Length of 1D traces')
    parser.add_argument('--profile_height', type=int, default=200,
                        help='Height of 2D profiles')
    parser.add_argument('--profile_width', type=int, default=200,
                        help='Width of 2D profiles')
    parser.add_argument('--num_wells', type=int, default=10,
                        help='Number of wells per 2D profile')
    parser.add_argument('--use_initial_impedance', action='store_true', default=True,
                        help='Use initial impedance as input (Scheme B)')
    parser.add_argument('--dz', type=float, default=4.0,
                        help='Depth sampling interval in meters (default: 4.0m)')
    parser.add_argument('--dx', type=float, default=25.0,
                        help='Lateral sampling interval in meters (default: 25.0m)')
    parser.add_argument('--velocity', type=float, default=3000.0,
                        help='Average P-wave velocity in m/s (default: 3000 m/s)')
    parser.add_argument('--wavelet_freq', type=float, default=25.0,
                        help='Dominant frequency of Ricker wavelet (Hz)')
    parser.add_argument('--noise_level', type=float, default=0.02,
                        help='Noise level for synthetic seismic')
    parser.add_argument('--smoothing_sigma', type=float, default=20.0,
                        help='Gaussian smoothing sigma for initial impedance (in samples)')
    
    # System configuration
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    test(args)


if __name__ == "__main__":
    main()
