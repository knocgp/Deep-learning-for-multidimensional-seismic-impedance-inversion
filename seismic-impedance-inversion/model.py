"""
Deep Learning Models for Seismic Impedance Inversion

This module implements 1D and 2D CNN architectures for seismic impedance prediction
as described in "Deep learning for multidimensional seismic impedance inversion" 
(Wu et al., 2021, Geophysics).

Architecture details:
- ResBlock with skip connections
- ReLU activation
- 1D CNN: ~10,001 parameters
- 2D CNN: Similar architecture with 3x3 2D convolutions
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """
    Residual block with skip connection.
    
    Args:
        channels (int): Number of input/output channels
        kernel_size (int or tuple): Kernel size for convolution
        dim (str): '1d' or '2d' for 1D or 2D convolution
    """
    def __init__(self, channels=16, kernel_size=3, dim='1d'):
        super(ResBlock, self).__init__()
        self.dim = dim
        padding = kernel_size // 2 if isinstance(kernel_size, int) else tuple(k // 2 for k in kernel_size)
        
        if dim == '1d':
            self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
            self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        else:  # 2d
            self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """Forward pass with residual connection."""
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += identity  # Skip connection
        out = self.relu(out)
        return out


class SeismicImpedanceCNN1D(nn.Module):
    """
    1D CNN for seismic impedance inversion.
    
    Architecture:
    - Input: 1D seismic trace (Scheme A) or 2-channel (seismic + initial impedance, Scheme B)
    - First conv: 16 filters, kernel size 7
    - 4 blocks: each with regular conv + ResBlock
    - Output conv: 1x1 to produce impedance
    
    Args:
        in_channels (int): Number of input channels (1 for Scheme A, 2 for Scheme B)
        num_filters (int): Number of filters in conv layers (default: 16)
        num_blocks (int): Number of convolutional blocks (default: 4)
    """
    def __init__(self, in_channels=2, num_filters=16, num_blocks=4):
        super(SeismicImpedanceCNN1D, self).__init__()
        
        # First convolutional layer: kernel size 7
        self.conv_first = nn.Conv1d(in_channels, num_filters, kernel_size=7, padding=3)
        self.relu = nn.ReLU(inplace=True)
        
        # Build blocks: each block contains a regular conv + ResBlock
        blocks = []
        for _ in range(num_blocks):
            # Regular convolutional layer
            blocks.append(nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1))
            blocks.append(nn.ReLU(inplace=True))
            # ResBlock
            blocks.append(ResBlock(channels=num_filters, kernel_size=3, dim='1d'))
        
        self.blocks = nn.Sequential(*blocks)
        
        # Final output layer: 1x1 conv to produce single-channel impedance
        self.conv_out = nn.Conv1d(num_filters, 1, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, length)
                - channels=1: seismic only (Scheme A)
                - channels=2: seismic + initial impedance (Scheme B)
        
        Returns:
            torch.Tensor: Predicted impedance of shape (batch, 1, length)
        """
        x = self.relu(self.conv_first(x))
        x = self.blocks(x)
        x = self.conv_out(x)
        return x


class SeismicImpedanceCNN2D(nn.Module):
    """
    2D CNN for seismic impedance inversion.
    
    Same architecture as 1D CNN but with 2D convolutions (3x3 kernels).
    Supports weak supervision with adaptive loss (training on well locations only).
    
    Architecture:
    - Input: 2D seismic profile + 2D initial impedance profile (2 channels)
    - First conv: 16 filters, kernel size 3x3
    - 4 blocks: each with regular conv + ResBlock
    - Output conv: 1x1 to produce impedance profile
    
    Args:
        in_channels (int): Number of input channels (default: 2)
        num_filters (int): Number of filters in conv layers (default: 16)
        num_blocks (int): Number of convolutional blocks (default: 4)
    """
    def __init__(self, in_channels=2, num_filters=16, num_blocks=4):
        super(SeismicImpedanceCNN2D, self).__init__()
        
        # First convolutional layer: 3x3 kernel
        self.conv_first = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # Build blocks: each block contains a regular conv + ResBlock
        blocks = []
        for _ in range(num_blocks):
            # Regular convolutional layer (3x3)
            blocks.append(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1))
            blocks.append(nn.ReLU(inplace=True))
            # ResBlock (3x3)
            blocks.append(ResBlock(channels=num_filters, kernel_size=3, dim='2d'))
        
        self.blocks = nn.Sequential(*blocks)
        
        # Final output layer: 1x1 conv to produce single-channel impedance
        self.conv_out = nn.Conv2d(num_filters, 1, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, 2, height, width)
                - channel 0: seismic profile
                - channel 1: initial impedance profile
        
        Returns:
            torch.Tensor: Predicted impedance profile of shape (batch, 1, height, width)
        """
        x = self.relu(self.conv_first(x))
        x = self.blocks(x)
        x = self.conv_out(x)
        return x


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test 1D CNN
    print("=" * 50)
    print("Testing 1D CNN")
    print("=" * 50)
    
    model_1d = SeismicImpedanceCNN1D(in_channels=2, num_filters=16, num_blocks=4)
    print(f"Total parameters: {count_parameters(model_1d):,}")
    
    # Test with random input
    batch_size = 4
    seq_length = 300
    x_1d = torch.randn(batch_size, 2, seq_length)
    y_1d = model_1d(x_1d)
    print(f"Input shape: {x_1d.shape}")
    print(f"Output shape: {y_1d.shape}")
    print()
    
    # Test 2D CNN
    print("=" * 50)
    print("Testing 2D CNN")
    print("=" * 50)
    
    model_2d = SeismicImpedanceCNN2D(in_channels=2, num_filters=16, num_blocks=4)
    print(f"Total parameters: {count_parameters(model_2d):,}")
    
    # Test with random input
    height, width = 128, 128
    x_2d = torch.randn(batch_size, 2, height, width)
    y_2d = model_2d(x_2d)
    print(f"Input shape: {x_2d.shape}")
    print(f"Output shape: {y_2d.shape}")
