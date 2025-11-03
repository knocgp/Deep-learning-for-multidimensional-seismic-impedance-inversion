"""
Data Loading and Preprocessing for Seismic Impedance Inversion

This module provides utilities for:
1. Generating synthetic seismic data and impedance models
2. Creating training samples for 1D and 2D CNNs
3. Preprocessing (smoothing, interpolation)
4. Random path extraction for 2D training
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.interpolate import griddata
from typing import Tuple, List, Optional


def generate_ricker_wavelet(frequency: float, dt: float, length: float) -> np.ndarray:
    """
    Generate Ricker wavelet for seismic convolution.
    
    Args:
        frequency: Dominant frequency in Hz
        dt: Time sampling interval in seconds
        length: Total length of wavelet in seconds
    
    Returns:
        Ricker wavelet array
    """
    t = np.arange(-length/2, length/2, dt)
    w = (1.0 - 2.0 * (np.pi * frequency * t) ** 2) * np.exp(-(np.pi * frequency * t) ** 2)
    return w


def impedance_to_reflectivity(impedance: np.ndarray) -> np.ndarray:
    """
    Convert impedance to reflectivity coefficients.
    
    Args:
        impedance: Acoustic impedance array
    
    Returns:
        Reflectivity coefficients
    """
    # Reflectivity = (Z2 - Z1) / (Z2 + Z1)
    z1 = impedance[:-1]
    z2 = impedance[1:]
    reflectivity = (z2 - z1) / (z2 + z1 + 1e-10)
    # Pad to maintain same length
    reflectivity = np.concatenate([reflectivity, [0]])
    return reflectivity


def generate_synthetic_seismic(impedance: np.ndarray, 
                                wavelet_freq: float = 25.0,
                                dt: float = 0.002,
                                noise_level: float = 0.0) -> np.ndarray:
    """
    Generate synthetic seismic trace from impedance model.
    
    Args:
        impedance: Impedance model (1D or multi-dimensional)
        wavelet_freq: Dominant frequency of Ricker wavelet (Hz)
        dt: Time sampling interval (seconds)
        noise_level: Standard deviation of additive Gaussian noise
    
    Returns:
        Synthetic seismic trace
    """
    # Generate wavelet
    wavelet = generate_ricker_wavelet(wavelet_freq, dt, length=0.128)
    
    # Convert impedance to reflectivity
    if impedance.ndim == 1:
        reflectivity = impedance_to_reflectivity(impedance)
        seismic = np.convolve(reflectivity, wavelet, mode='same')
    else:
        # For multi-dimensional, process along first axis
        seismic = np.zeros_like(impedance)
        for idx in np.ndindex(impedance.shape[1:]):
            trace_idx = (slice(None),) + idx
            reflectivity = impedance_to_reflectivity(impedance[trace_idx])
            seismic[trace_idx] = np.convolve(reflectivity, wavelet, mode='same')
    
    # Add noise
    if noise_level > 0:
        seismic += np.random.randn(*seismic.shape) * noise_level
    
    return seismic


def create_initial_impedance(impedance: np.ndarray, 
                              smoothing_sigma: float = 20.0,
                              method: str = 'gaussian') -> np.ndarray:
    """
    Create initial (low-frequency) impedance model by smoothing.
    
    Args:
        impedance: True impedance model
        smoothing_sigma: Gaussian filter standard deviation (in samples)
        method: Smoothing method ('gaussian')
    
    Returns:
        Smoothed initial impedance model
    """
    if method == 'gaussian':
        if impedance.ndim == 1:
            initial = gaussian_filter1d(impedance, sigma=smoothing_sigma, mode='nearest')
        else:
            initial = gaussian_filter(impedance, sigma=smoothing_sigma, mode='nearest')
        return initial
    else:
        raise ValueError(f"Unknown smoothing method: {method}")


class SeismicImpedanceDataset1D(Dataset):
    """
    Dataset for 1D seismic impedance inversion.
    
    Generates synthetic training data with random impedance models.
    
    Args:
        num_samples: Number of training samples
        trace_length: Length of each trace in samples
        min_impedance: Minimum impedance value
        max_impedance: Maximum impedance value
        num_layers: Number of random layers in impedance model
        use_initial_impedance: Whether to include initial impedance as input (Scheme B)
        wavelet_freq: Dominant frequency of Ricker wavelet (Hz)
        noise_level: Noise level for synthetic seismic
        smoothing_sigma: Smoothing parameter for initial impedance
    """
    def __init__(self, 
                 num_samples: int = 1000,
                 trace_length: int = 300,
                 min_impedance: float = 2000.0,
                 max_impedance: float = 5000.0,
                 num_layers: int = 20,
                 use_initial_impedance: bool = True,
                 wavelet_freq: float = 25.0,
                 noise_level: float = 0.02,
                 smoothing_sigma: float = 20.0):
        
        self.num_samples = num_samples
        self.trace_length = trace_length
        self.min_impedance = min_impedance
        self.max_impedance = max_impedance
        self.num_layers = num_layers
        self.use_initial_impedance = use_initial_impedance
        self.wavelet_freq = wavelet_freq
        self.noise_level = noise_level
        self.smoothing_sigma = smoothing_sigma
        
        # Pre-generate all samples for consistency during training
        self.seismic_traces = []
        self.impedance_traces = []
        self.initial_impedance_traces = []
        
        self._generate_dataset()
    
    def _generate_random_impedance(self) -> np.ndarray:
        """Generate random layered impedance model."""
        # Create random layer boundaries
        layer_positions = sorted(np.random.randint(0, self.trace_length, self.num_layers))
        layer_positions = [0] + layer_positions + [self.trace_length]
        
        # Create impedance values for each layer
        impedance = np.zeros(self.trace_length)
        for i in range(len(layer_positions) - 1):
            start, end = layer_positions[i], layer_positions[i + 1]
            layer_impedance = np.random.uniform(self.min_impedance, self.max_impedance)
            impedance[start:end] = layer_impedance
        
        # Smooth slightly to avoid sharp discontinuities
        impedance = gaussian_filter1d(impedance, sigma=1.0, mode='nearest')
        
        return impedance
    
    def _generate_dataset(self):
        """Pre-generate all training samples."""
        for _ in range(self.num_samples):
            # Generate random impedance
            impedance = self._generate_random_impedance()
            
            # Generate synthetic seismic
            seismic = generate_synthetic_seismic(impedance, 
                                                  wavelet_freq=self.wavelet_freq,
                                                  noise_level=self.noise_level)
            
            # Create initial impedance
            initial_impedance = create_initial_impedance(impedance, 
                                                          smoothing_sigma=self.smoothing_sigma)
            
            self.seismic_traces.append(seismic)
            self.impedance_traces.append(impedance)
            self.initial_impedance_traces.append(initial_impedance)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Get a training sample.
        
        Returns:
            input_tensor: (1 or 2, trace_length) - seismic [+ initial impedance]
            target_tensor: (1, trace_length) - true impedance
        """
        seismic = self.seismic_traces[idx]
        impedance = self.impedance_traces[idx]
        
        if self.use_initial_impedance:
            initial_impedance = self.initial_impedance_traces[idx]
            input_data = np.stack([seismic, initial_impedance], axis=0)
        else:
            input_data = seismic[np.newaxis, :]
        
        target = impedance[np.newaxis, :]
        
        return torch.FloatTensor(input_data), torch.FloatTensor(target)


class SeismicImpedanceDataset2D(Dataset):
    """
    Dataset for 2D seismic impedance inversion with weak supervision.
    
    Generates 2D synthetic profiles with random paths connecting well locations.
    
    Args:
        num_samples: Number of training samples (2D profiles)
        profile_height: Height of 2D profile
        profile_width: Width of 2D profile
        num_wells: Number of wells per profile
        min_impedance: Minimum impedance value
        max_impedance: Maximum impedance value
        num_layers: Number of layers in impedance model
        wavelet_freq: Dominant frequency of Ricker wavelet (Hz)
        noise_level: Noise level for synthetic seismic
        smoothing_sigma: Smoothing parameter for initial impedance
    """
    def __init__(self,
                 num_samples: int = 500,
                 profile_height: int = 200,
                 profile_width: int = 200,
                 num_wells: int = 10,
                 min_impedance: float = 2000.0,
                 max_impedance: float = 5000.0,
                 num_layers: int = 30,
                 wavelet_freq: float = 25.0,
                 noise_level: float = 0.02,
                 smoothing_sigma: float = 20.0):
        
        self.num_samples = num_samples
        self.profile_height = profile_height
        self.profile_width = profile_width
        self.num_wells = num_wells
        self.min_impedance = min_impedance
        self.max_impedance = max_impedance
        self.num_layers = num_layers
        self.wavelet_freq = wavelet_freq
        self.noise_level = noise_level
        self.smoothing_sigma = smoothing_sigma
        
        # Pre-generate dataset
        self.seismic_profiles = []
        self.impedance_profiles = []
        self.initial_impedance_profiles = []
        self.masks = []  # Binary masks indicating well locations
        
        self._generate_dataset()
    
    def _generate_random_impedance_2d(self) -> np.ndarray:
        """Generate random 2D layered impedance model."""
        # Create horizontal layers with lateral variations
        impedance = np.zeros((self.profile_height, self.profile_width))
        
        layer_positions = sorted(np.random.randint(0, self.profile_height, self.num_layers))
        layer_positions = [0] + layer_positions + [self.profile_height]
        
        for i in range(len(layer_positions) - 1):
            start, end = layer_positions[i], layer_positions[i + 1]
            # Create lateral variation
            base_impedance = np.random.uniform(self.min_impedance, self.max_impedance)
            lateral_variation = np.random.randn(self.profile_width) * 100
            for j in range(start, end):
                impedance[j, :] = base_impedance + lateral_variation
        
        # Smooth to create realistic variations
        impedance = gaussian_filter(impedance, sigma=2.0, mode='nearest')
        
        return impedance
    
    def _create_well_mask(self) -> Tuple[np.ndarray, List[int]]:
        """
        Create binary mask with well locations.
        
        Returns:
            mask: Binary mask (1 at well locations, 0 elsewhere)
            well_positions: List of well lateral positions
        """
        mask = np.zeros((self.profile_height, self.profile_width))
        
        # Random well positions (columns)
        well_positions = sorted(np.random.choice(self.profile_width, 
                                                  size=self.num_wells, 
                                                  replace=False))
        
        # Mark entire well traces
        for pos in well_positions:
            mask[:, pos] = 1.0
        
        return mask, well_positions
    
    def _generate_dataset(self):
        """Pre-generate all training samples."""
        for _ in range(self.num_samples):
            # Generate 2D impedance model
            impedance = self._generate_random_impedance_2d()
            
            # Generate synthetic seismic
            seismic = generate_synthetic_seismic(impedance,
                                                  wavelet_freq=self.wavelet_freq,
                                                  noise_level=self.noise_level)
            
            # Create initial impedance
            initial_impedance = create_initial_impedance(impedance,
                                                          smoothing_sigma=self.smoothing_sigma)
            
            # Create well mask
            mask, _ = self._create_well_mask()
            
            self.seismic_profiles.append(seismic)
            self.impedance_profiles.append(impedance)
            self.initial_impedance_profiles.append(initial_impedance)
            self.masks.append(mask)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Get a training sample.
        
        Returns:
            input_tensor: (2, height, width) - seismic + initial impedance
            target_tensor: (1, height, width) - true impedance
            mask_tensor: (1, height, width) - binary mask for well locations
        """
        seismic = self.seismic_profiles[idx]
        impedance = self.impedance_profiles[idx]
        initial_impedance = self.initial_impedance_profiles[idx]
        mask = self.masks[idx]
        
        input_data = np.stack([seismic, initial_impedance], axis=0)
        target = impedance[np.newaxis, :]
        mask_tensor = mask[np.newaxis, :]
        
        return (torch.FloatTensor(input_data), 
                torch.FloatTensor(target), 
                torch.FloatTensor(mask_tensor))


def get_dataloader(dataset_type: str = '1d',
                   batch_size: int = 16,
                   num_samples: int = 1000,
                   shuffle: bool = True,
                   num_workers: int = 0,
                   **kwargs) -> DataLoader:
    """
    Create DataLoader for training.
    
    Args:
        dataset_type: '1d' or '2d'
        batch_size: Batch size
        num_samples: Number of samples in dataset
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        **kwargs: Additional arguments for Dataset constructor
    
    Returns:
        DataLoader instance
    """
    if dataset_type == '1d':
        dataset = SeismicImpedanceDataset1D(num_samples=num_samples, **kwargs)
    elif dataset_type == '2d':
        dataset = SeismicImpedanceDataset2D(num_samples=num_samples, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    loader = DataLoader(dataset, 
                        batch_size=batch_size, 
                        shuffle=shuffle, 
                        num_workers=num_workers,
                        pin_memory=True)
    
    return loader


if __name__ == "__main__":
    # Test 1D dataset
    print("Testing 1D Dataset")
    print("=" * 50)
    
    dataset_1d = SeismicImpedanceDataset1D(num_samples=10, trace_length=300)
    loader_1d = DataLoader(dataset_1d, batch_size=2)
    
    for inputs, targets in loader_1d:
        print(f"Input shape: {inputs.shape}")
        print(f"Target shape: {targets.shape}")
        break
    
    print()
    
    # Test 2D dataset
    print("Testing 2D Dataset")
    print("=" * 50)
    
    dataset_2d = SeismicImpedanceDataset2D(num_samples=10, profile_height=128, profile_width=128)
    loader_2d = DataLoader(dataset_2d, batch_size=2)
    
    for inputs, targets, masks in loader_2d:
        print(f"Input shape: {inputs.shape}")
        print(f"Target shape: {targets.shape}")
        print(f"Mask shape: {masks.shape}")
        print(f"Well coverage: {masks.mean().item() * 100:.2f}%")
        break
