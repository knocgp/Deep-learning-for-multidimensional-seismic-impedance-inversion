"""
Training Script for Seismic Impedance Inversion

This script implements training for both 1D and 2D CNN models.
- 1D CNN: Fully supervised with MSE loss
- 2D CNN: Weak supervision with adaptive/masked MSE loss

Command-line usage examples:
    # Train 1D CNN
    python train.py --model 1d --epochs 300 --batch_size 16 --lr 0.001
    
    # Train 2D CNN with weak supervision
    python train.py --model 2d --epochs 300 --batch_size 8 --lr 0.001
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from model import SeismicImpedanceCNN1D, SeismicImpedanceCNN2D, count_parameters
from data_loading import get_dataloader
from utils import (save_checkpoint, calculate_metrics, plot_training_curves, 
                   save_config, AverageMeter, set_seed)


class AdaptiveMSELoss(nn.Module):
    """
    Adaptive MSE Loss for weak supervision in 2D case.
    
    Loss is computed only at well locations indicated by the mask.
    L = (1 / sum(mask)) * sum(mask * (pred - target)^2)
    
    Args:
        reduction: 'mean' or 'sum'
    """
    def __init__(self, reduction='mean'):
        super(AdaptiveMSELoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, predictions, targets, mask):
        """
        Forward pass.
        
        Args:
            predictions: Predicted impedance (B, 1, H, W)
            targets: True impedance (B, 1, H, W)
            mask: Binary mask (B, 1, H, W) - 1 at well locations, 0 elsewhere
        
        Returns:
            Masked MSE loss
        """
        # Compute squared error
        squared_error = (predictions - targets) ** 2
        
        # Apply mask
        masked_error = squared_error * mask
        
        # Compute loss
        if self.reduction == 'mean':
            # Average over masked positions
            loss = masked_error.sum() / (mask.sum() + 1e-10)
        else:
            loss = masked_error.sum()
        
        return loss


def train_epoch_1d(model, dataloader, criterion, optimizer, device):
    """
    Train one epoch for 1D model.
    
    Args:
        model: 1D CNN model
        dataloader: Training dataloader
        criterion: Loss function (MSE)
        optimizer: Optimizer
        device: Device to train on
    
    Returns:
        Average training loss
    """
    model.train()
    loss_meter = AverageMeter()
    
    pbar = tqdm(dataloader, desc='Training')
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        loss_meter.update(loss.item(), inputs.size(0))
        pbar.set_postfix({'loss': f'{loss_meter.avg:.6f}'})
    
    return loss_meter.avg


def train_epoch_2d(model, dataloader, criterion, optimizer, device):
    """
    Train one epoch for 2D model with weak supervision.
    
    Args:
        model: 2D CNN model
        dataloader: Training dataloader
        criterion: Adaptive MSE loss
        optimizer: Optimizer
        device: Device to train on
    
    Returns:
        Average training loss
    """
    model.train()
    loss_meter = AverageMeter()
    
    pbar = tqdm(dataloader, desc='Training')
    for inputs, targets, masks in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        loss_meter.update(loss.item(), inputs.size(0))
        pbar.set_postfix({'loss': f'{loss_meter.avg:.6f}'})
    
    return loss_meter.avg


def validate_1d(model, dataloader, criterion, device):
    """
    Validate 1D model.
    
    Args:
        model: 1D CNN model
        dataloader: Validation dataloader
        criterion: Loss function (MSE)
        device: Device to validate on
    
    Returns:
        Average validation loss
    """
    model.eval()
    loss_meter = AverageMeter()
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc='Validation'):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Update metrics
            loss_meter.update(loss.item(), inputs.size(0))
    
    return loss_meter.avg


def validate_2d(model, dataloader, criterion, device):
    """
    Validate 2D model.
    
    Args:
        model: 2D CNN model
        dataloader: Validation dataloader
        criterion: Adaptive MSE loss
        device: Device to validate on
    
    Returns:
        Average validation loss
    """
    model.eval()
    loss_meter = AverageMeter()
    
    with torch.no_grad():
        for inputs, targets, masks in tqdm(dataloader, desc='Validation'):
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets, masks)
            
            # Update metrics
            loss_meter.update(loss.item(), inputs.size(0))
    
    return loss_meter.avg


def train(args):
    """
    Main training function.
    
    Args:
        args: Command-line arguments
    """
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    save_config(config, os.path.join(args.log_dir, 'config.json'))
    
    # Create model
    if args.model == '1d':
        model = SeismicImpedanceCNN1D(
            in_channels=2 if args.use_initial_impedance else 1,
            num_filters=args.num_filters,
            num_blocks=args.num_blocks
        )
        criterion = nn.MSELoss()
        train_epoch_fn = train_epoch_1d
        validate_fn = validate_1d
    else:  # 2d
        model = SeismicImpedanceCNN2D(
            in_channels=2,
            num_filters=args.num_filters,
            num_blocks=args.num_blocks
        )
        criterion = AdaptiveMSELoss()
        train_epoch_fn = train_epoch_2d
        validate_fn = validate_2d
    
    model = model.to(device)
    print(f"\nModel: {args.model.upper()} CNN")
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Create optimizer with adaptive learning rate
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler (adaptive decreasing)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Create dataloaders
    print("\nCreating datasets...")
    
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
    
    train_loader = get_dataloader(
        dataset_type=args.model,
        batch_size=args.batch_size,
        num_samples=args.num_train_samples,
        shuffle=True,
        num_workers=args.num_workers,
        **dataset_kwargs
    )
    
    val_loader = get_dataloader(
        dataset_type=args.model,
        batch_size=args.batch_size,
        num_samples=args.num_val_samples,
        shuffle=False,
        num_workers=args.num_workers,
        **dataset_kwargs
    )
    
    print(f"Training samples: {args.num_train_samples}")
    print(f"Validation samples: {args.num_val_samples}")
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_loss = train_epoch_fn(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate_fn(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print summary
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        if epoch % args.save_interval == 0 or is_best:
            checkpoint_path = os.path.join(
                args.save_dir, 
                f'{args.model}_epoch_{epoch}.pth'
            )
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path, is_best)
    
    # Plot training curves
    print("\nPlotting training curves...")
    plot_path = os.path.join(args.log_dir, 'training_curves.png')
    plot_training_curves(train_losses, val_losses, save_path=plot_path)
    
    # Save final model
    final_path = os.path.join(args.save_dir, f'{args.model}_final.pth')
    save_checkpoint(model, optimizer, args.epochs, val_losses[-1], final_path)
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Models saved to: {args.save_dir}")
    print(f"Logs saved to: {args.log_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train Seismic Impedance Inversion Models')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='1d', choices=['1d', '2d'],
                        help='Model type: 1d or 2d')
    parser.add_argument('--num_filters', type=int, default=16,
                        help='Number of filters in conv layers')
    parser.add_argument('--num_blocks', type=int, default=4,
                        help='Number of convolutional blocks')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Data configuration
    parser.add_argument('--num_train_samples', type=int, default=1000,
                        help='Number of training samples')
    parser.add_argument('--num_val_samples', type=int, default=200,
                        help='Number of validation samples')
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
    parser.add_argument('--save_dir', type=str, default='./models',
                        help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory to save logs')
    parser.add_argument('--save_interval', type=int, default=50,
                        help='Save model every N epochs')
    
    args = parser.parse_args()
    
    train(args)


if __name__ == "__main__":
    main()
