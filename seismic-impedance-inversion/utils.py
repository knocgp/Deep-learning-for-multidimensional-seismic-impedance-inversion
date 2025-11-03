"""
Utility Functions for Seismic Impedance Inversion

This module provides helper functions for:
1. Model saving/loading
2. Metrics calculation
3. Visualization
4. Logging
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
import json


def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    loss: float,
                    save_path: str,
                    is_best: bool = False):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        save_path: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = save_path.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_path)
        print(f"Saved best model to {best_path}")


def load_checkpoint(model: torch.nn.Module,
                    optimizer: Optional[torch.optim.Optimizer],
                    checkpoint_path: str,
                    device: str = 'cpu') -> Tuple[int, float]:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer (optional)
        checkpoint_path: Path to checkpoint
        device: Device to load model to
    
    Returns:
        epoch: Epoch number
        loss: Loss value
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Loaded checkpoint from epoch {epoch} with loss {loss:.6f}")
    
    return epoch, loss


def calculate_metrics(predictions: np.ndarray, 
                      targets: np.ndarray,
                      mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate evaluation metrics.
    
    Args:
        predictions: Predicted impedance values
        targets: True impedance values
        mask: Optional mask for partial evaluation (2D case)
    
    Returns:
        Dictionary of metrics
    """
    if mask is not None:
        # Apply mask for 2D weak supervision
        predictions = predictions[mask > 0.5]
        targets = targets[mask > 0.5]
    
    # Mean Squared Error
    mse = np.mean((predictions - targets) ** 2)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(predictions - targets))
    
    # Relative Error
    relative_error = np.mean(np.abs(predictions - targets) / (np.abs(targets) + 1e-10))
    
    # R-squared
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))
    
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'relative_error': float(relative_error),
        'r2': float(r2)
    }
    
    return metrics


def plot_1d_results(seismic: np.ndarray,
                    initial_impedance: np.ndarray,
                    true_impedance: np.ndarray,
                    predicted_impedance: np.ndarray,
                    save_path: Optional[str] = None):
    """
    Plot 1D inversion results.
    
    Args:
        seismic: Input seismic trace
        initial_impedance: Initial (smoothed) impedance
        true_impedance: Ground truth impedance
        predicted_impedance: Predicted impedance
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 6))
    
    # Plot seismic
    axes[0].plot(seismic)
    axes[0].set_title('Input Seismic')
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # Plot initial impedance
    axes[1].plot(initial_impedance)
    axes[1].set_title('Initial Impedance')
    axes[1].set_xlabel('Sample')
    axes[1].set_ylabel('Impedance (m/s·g/cm³)')
    axes[1].grid(True, alpha=0.3)
    
    # Plot comparison
    axes[2].plot(true_impedance, 'b-', label='True', linewidth=2)
    axes[2].plot(predicted_impedance, 'r--', label='Predicted', linewidth=2)
    axes[2].set_title('Impedance Comparison')
    axes[2].set_xlabel('Sample')
    axes[2].set_ylabel('Impedance (m/s·g/cm³)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Plot error
    error = np.abs(true_impedance - predicted_impedance)
    axes[3].plot(error, 'g-', linewidth=2)
    axes[3].set_title('Absolute Error')
    axes[3].set_xlabel('Sample')
    axes[3].set_ylabel('Error (m/s·g/cm³)')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_2d_results(seismic: np.ndarray,
                    initial_impedance: np.ndarray,
                    true_impedance: np.ndarray,
                    predicted_impedance: np.ndarray,
                    mask: Optional[np.ndarray] = None,
                    save_path: Optional[str] = None):
    """
    Plot 2D inversion results.
    
    Args:
        seismic: Input seismic profile (2D)
        initial_impedance: Initial impedance profile
        true_impedance: Ground truth impedance profile
        predicted_impedance: Predicted impedance profile
        mask: Binary mask for well locations (optional)
        save_path: Path to save figure (optional)
    """
    num_plots = 5 if mask is not None else 4
    fig, axes = plt.subplots(1, num_plots, figsize=(4*num_plots, 6))
    
    # Common parameters for imshow
    aspect = 'auto'
    cmap_seismic = 'seismic'
    cmap_impedance = 'viridis'
    
    # Plot seismic
    im0 = axes[0].imshow(seismic, aspect=aspect, cmap=cmap_seismic)
    axes[0].set_title('Input Seismic')
    axes[0].set_xlabel('Lateral Position')
    axes[0].set_ylabel('Depth Sample')
    plt.colorbar(im0, ax=axes[0])
    
    # Plot initial impedance
    im1 = axes[1].imshow(initial_impedance, aspect=aspect, cmap=cmap_impedance)
    axes[1].set_title('Initial Impedance')
    axes[1].set_xlabel('Lateral Position')
    axes[1].set_ylabel('Depth Sample')
    plt.colorbar(im1, ax=axes[1])
    
    # Plot true impedance
    im2 = axes[2].imshow(true_impedance, aspect=aspect, cmap=cmap_impedance)
    axes[2].set_title('True Impedance')
    axes[2].set_xlabel('Lateral Position')
    axes[2].set_ylabel('Depth Sample')
    plt.colorbar(im2, ax=axes[2])
    
    # Plot predicted impedance
    im3 = axes[3].imshow(predicted_impedance, aspect=aspect, cmap=cmap_impedance)
    axes[3].set_title('Predicted Impedance')
    axes[3].set_xlabel('Lateral Position')
    axes[3].set_ylabel('Depth Sample')
    plt.colorbar(im3, ax=axes[3])
    
    # Plot mask if provided
    if mask is not None:
        im4 = axes[4].imshow(mask, aspect=aspect, cmap='gray')
        axes[4].set_title('Well Locations (Mask)')
        axes[4].set_xlabel('Lateral Position')
        axes[4].set_ylabel('Depth Sample')
        plt.colorbar(im4, ax=axes[4])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(train_losses: list,
                         val_losses: list,
                         save_path: Optional[str] = None):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_config(config: dict, save_path: str):
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save JSON file
    """
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Saved configuration to {save_path}")


def load_config(config_path: str) -> dict:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to JSON file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # For deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions")
    print("=" * 50)
    
    # Test metrics calculation
    pred = np.random.randn(100) * 1000 + 3000
    target = pred + np.random.randn(100) * 100
    
    metrics = calculate_metrics(pred, target)
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nUtility functions are working correctly!")
