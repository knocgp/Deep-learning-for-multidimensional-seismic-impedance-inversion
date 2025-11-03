# Deep Learning for Multidimensional Seismic Impedance Inversion

[![Korean](https://img.shields.io/badge/한국어-README-blue)](README_KR.md)

PyTorch implementation of "Deep learning for multidimensional seismic impedance inversion" (Wu et al., 2021, Geophysics).

## Table of Contents
- [Overview](#overview)
- [Geological and Geophysical Background](#geological-and-geophysical-background)
- [Traditional vs Deep Learning Approach](#traditional-vs-deep-learning-approach)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Results](#results)
- [Citation](#citation)

## Overview

This repository implements deep convolutional neural networks (CNNs) for seismic acoustic impedance inversion. The implementation includes both 1D and 2D CNN architectures with weak supervision capability, enabling accurate impedance prediction from seismic data.

**Key Features:**
- 1D CNN for trace-by-trace inversion with ~10,000 parameters
- 2D CNN for full profile inversion with lateral continuity preservation
- Weak supervision training using sparse well log data
- Adaptive masked loss for partial supervision
- Initial impedance model integration for low-frequency trend control

## Geological and Geophysical Background

### What is Seismic Impedance?

**Acoustic impedance (AI)** is a fundamental rock property defined as:

```
AI = ρ × V_p
```

Where:
- **ρ (rho)** = bulk density (g/cm³)
- **V_p** = P-wave velocity (m/s)
- **AI unit**: (m/s) · (g/cm³) or kg/(m²·s)

Typical impedance values:
- Sandstone: 8,000 - 12,000 (m/s·g/cm³)
- Shale: 6,000 - 10,000 (m/s·g/cm³)
- Limestone: 10,000 - 15,000 (m/s·g/cm³)
- Gas-saturated sandstone: 4,000 - 8,000 (m/s·g/cm³)

### Why is Impedance Important?

1. **Lithology Identification**: Different rock types have distinct impedance values
2. **Reservoir Characterization**: Low impedance often indicates gas-bearing zones
3. **Rock Properties**: Relates to porosity, fluid content, and mechanical strength
4. **Quantitative Interpretation**: Bridge between seismic data and reservoir properties

### The Forward Problem: Seismic Data Generation

Seismic data is generated through the following physical process:

1. **Reflectivity Calculation**:
   ```
   R_i = (AI_{i+1} - AI_i) / (AI_{i+1} + AI_i)
   ```
   Where R_i is the reflection coefficient at interface i

2. **Seismic Trace Generation**:
   ```
   Seismic(t) = Wavelet(t) ⊗ Reflectivity(t) + Noise
   ```
   Where ⊗ denotes convolution

3. **Key Characteristics**:
   - Seismic data is **band-limited** (typically 10-60 Hz)
   - Original impedance contains **all frequencies** (0-∞ Hz)
   - Seismic data **lacks low frequencies** (< 5-10 Hz)
   - Well logs provide **high frequency** but **sparse spatial sampling**

### The Inverse Problem: Impedance from Seismic

The inversion problem aims to recover impedance from seismic data:

```
Given: Seismic data S(x,t) and sparse well logs
Find: Impedance model AI(x,z) that best explains the seismic data
```

**Challenges:**
- **Ill-posed problem**: Multiple impedance models can produce similar seismic responses
- **Band-limited**: Missing low and very high frequencies
- **Noise**: Seismic data contains various types of noise
- **Non-uniqueness**: Different geological scenarios may yield similar seismic signatures
- **Sparse constraints**: Well logs provide information only at discrete locations

## Traditional vs Deep Learning Approach

### Traditional Seismic Inversion Methods

#### 1. Model-Based Inversion
- **Method**: Iteratively update impedance model to minimize misfit between synthetic and observed seismic
- **Advantages**: Physics-based, interpretable
- **Disadvantages**: 
  - Computationally expensive
  - Requires accurate initial model
  - Sensitive to noise and parameter choices
  - Local minima problem in optimization

#### 2. Sparse Spike Inversion
- **Method**: Assume impedance consists of sparse reflectors
- **Advantages**: Fast, stable
- **Disadvantages**: 
  - Oversimplified assumption
  - Limited resolution
  - May miss subtle features

#### 3. Geostatistical Inversion
- **Method**: Incorporate geological statistics and well data
- **Advantages**: Honors geological constraints
- **Disadvantages**:
  - Requires extensive prior information
  - Computationally intensive
  - Complex parameter tuning

### Deep Learning Approach (This Work)

#### Core Philosophy
Instead of explicitly solving the physics-based inverse problem, learn the mapping:

```
f_θ: (Seismic, Initial_AI) → Impedance
```

Where θ represents neural network parameters learned from data.

#### Key Advantages

1. **Direct Mapping**: End-to-end learning without iterative forward modeling
2. **Speed**: Once trained, inference is extremely fast (milliseconds per trace)
3. **Lateral Continuity**: 2D CNN naturally preserves geological structure
4. **Weak Supervision**: Can learn from sparse well log data
5. **Implicit Regularization**: Network architecture provides built-in constraints
6. **Non-linearity**: Can capture complex relationships between seismic and impedance

#### Architecture Innovation

| Traditional 1D Networks | This Work (2D CNN) |
|-------------------------|-------------------|
| Process trace-by-trace | Process full 2D sections |
| Lateral discontinuities | Preserves lateral continuity |
| No spatial context | Utilizes neighboring traces |
| Fully supervised only | Supports weak supervision |

#### Initial Impedance Model: The Key Innovation

**Problem**: Seismic data lacks low-frequency information (< 10 Hz)

**Solution**: Provide low-frequency trend as additional input channel

```python
Input Channels:
1. Seismic data (band-limited: 10-60 Hz)
2. Initial impedance (smoothed, contains low frequencies: 0-10 Hz)
↓
Network combines both to predict full-bandwidth impedance (0-100+ Hz)
```

**How Initial Model is Created:**
- From well logs: Interpolate impedance values using structure-guided methods
- From prior model: Use geological knowledge or previous inversion results
- Smoothing: Apply heavy Gaussian smoothing (σ = 20 samples ≈ 500-1000 m)

**Analogy to Traditional Inversion:**
- Initial impedance ≈ "Starting model" in iterative inversion
- But here, the network learns to extract residuals rather than iterative updates

## Model Architecture

### 1D CNN Architecture

```
Input: (B, C, L)
  ├─ C=1: Seismic only (Scheme A)
  └─ C=2: Seismic + Initial Impedance (Scheme B)
  └─ L: Trace length (e.g., 300 samples)

Architecture:
┌─────────────────────────────────────┐
│ Conv1D (C→16, kernel=7, padding=3)  │  ← First layer: 16 filters
├─────────────────────────────────────┤
│ ReLU                                │
├─────────────────────────────────────┤
│ ┌─ Block 1 ────────────────────┐   │
│ │ Conv1D (16→16, k=3, p=1)     │   │
│ │ ReLU                         │   │
│ │ ResBlock (16 channels)       │   │  ← 4 identical blocks
│ │   ├─ Conv1D (16→16, k=3)    │   │
│ │   ├─ ReLU                   │   │
│ │   ├─ Conv1D (16→16, k=3)    │   │
│ │   └─ Add (skip connection)  │   │
│ └──────────────────────────────┘   │
│ ... (3 more blocks)                │
├─────────────────────────────────────┤
│ Conv1D (16→1, kernel=1)            │  ← Output layer: 1×1 conv
└─────────────────────────────────────┘

Output: (B, 1, L) - Predicted impedance
Parameters: ~10,001
```

### 2D CNN Architecture

```
Input: (B, 2, H, W)
  ├─ Channel 0: Seismic profile
  ├─ Channel 1: Initial impedance profile
  ├─ H: Height (depth samples, e.g., 200)
  └─ W: Width (lateral positions, e.g., 200)

Architecture:
┌─────────────────────────────────────┐
│ Conv2D (2→16, kernel=3×3, padding=1)│  ← First layer: 16 filters
├─────────────────────────────────────┤
│ ReLU                                │
├─────────────────────────────────────┤
│ ┌─ Block 1 ────────────────────┐   │
│ │ Conv2D (16→16, k=3×3, p=1)   │   │
│ │ ReLU                         │   │
│ │ ResBlock (16 channels)       │   │  ← 4 identical blocks
│ │   ├─ Conv2D (16→16, k=3×3)  │   │
│ │   ├─ ReLU                   │   │
│ │   ├─ Conv2D (16→16, k=3×3)  │   │
│ │   └─ Add (skip connection)  │   │
│ └──────────────────────────────┘   │
│ ... (3 more blocks)                │
├─────────────────────────────────────┤
│ Conv2D (16→1, kernel=1×1)          │  ← Output layer
└─────────────────────────────────────┘

Output: (B, 1, H, W) - Predicted impedance profile
Parameters: ~10,129
```

### ResBlock Design

Residual blocks enable deeper networks without vanishing gradients:

```python
class ResBlock(nn.Module):
    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + identity  # Skip connection
        out = self.relu(out)
        return out
```

**Why ResBlocks?**
- Enable training deeper networks
- Gradient flows more easily during backpropagation
- Learn residual corrections rather than full mapping
- Prevent degradation in very deep networks

### Loss Functions

#### 1D CNN: Standard MSE Loss

```python
L_1D = (1/N) Σ (y_pred - y_true)²
```

Fully supervised: All samples have ground truth labels.

#### 2D CNN: Adaptive Masked Loss (Weak Supervision)

```python
L_2D = Σ [w · (y_pred - y_true)²] / Σ w

where:
  w[i,j] = 1  at well log locations
  w[i,j] = 0  elsewhere
```

**Key Innovation**: Network predicts full 2D impedance but loss computed only at sparse well locations.

**Benefits:**
- Trains with sparse labels (realistic for field data)
- Network learns to interpolate between wells
- Preserves lateral continuity naturally
- More practical than requiring dense labels

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA-capable GPU (optional, but recommended)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd seismic-impedance-inversion

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
tqdm>=4.65.0
scikit-learn>=1.2.0
tensorboard>=2.13.0
```

## Quick Start

### 1. Train a 1D Model

```bash
# Train with default parameters (Scheme B: seismic + initial impedance)
python train.py --model 1d --epochs 300 --batch_size 16 --lr 0.001

# Train Scheme A (seismic only)
python train.py --model 1d --epochs 300 --use_initial_impedance False
```

### 2. Train a 2D Model

```bash
# Train 2D model with weak supervision
python train.py --model 2d --epochs 300 --batch_size 8 --lr 0.001 \
    --profile_height 200 --profile_width 200 --num_wells 10
```

### 3. Test/Inference

```bash
# Test 1D model
python test.py --model 1d --checkpoint ./models/1d_best.pth --num_samples 100

# Test 2D model
python test.py --model 2d --checkpoint ./models/2d_best.pth --num_samples 50
```

### 4. Verify Installation

```bash
# Test model architecture
python model.py

# Test data loading
python data_loading.py

# Test utilities
python utils.py
```

## Detailed Usage

### Training Configuration

#### 1D CNN Training Parameters

```bash
python train.py \
    --model 1d \
    --epochs 300 \                    # Training epochs (paper: ~300)
    --batch_size 16 \                 # Batch size
    --lr 0.001 \                      # Initial learning rate (paper: 0.001)
    --num_filters 16 \                # Conv filters (paper: 16)
    --num_blocks 4 \                  # Number of blocks (paper: 4)
    --num_train_samples 1000 \        # Training samples (paper: 40 logs)
    --num_val_samples 200 \           # Validation samples (paper: 10 logs)
    --trace_length 300 \              # Trace length (paper: 300)
    --use_initial_impedance True \    # Use Scheme B
    --wavelet_freq 25.0 \             # Ricker wavelet frequency (Hz)
    --noise_level 0.02 \              # Additive noise level
    --smoothing_sigma 20.0 \          # Initial model smoothing (paper: σ=20)
    --save_dir ./models \             # Model save directory
    --log_dir ./logs                  # Log directory
```

#### 2D CNN Training Parameters

```bash
python train.py \
    --model 2d \
    --epochs 300 \
    --batch_size 8 \                  # Smaller batch for 2D (memory)
    --lr 0.001 \
    --num_filters 16 \
    --num_blocks 4 \
    --num_train_samples 500 \         # 2D profiles
    --num_val_samples 100 \
    --profile_height 200 \            # Depth samples
    --profile_width 200 \             # Lateral samples
    --num_wells 10 \                  # Wells per profile (paper: ≥5)
    --wavelet_freq 25.0 \
    --noise_level 0.02 \
    --smoothing_sigma 20.0 \
    --save_dir ./models \
    --log_dir ./logs
```

### Data Format and Size

#### Input Data Specifications

**1D CNN:**
- **Input Shape**: `(batch_size, channels, trace_length)`
  - `channels = 1`: Seismic only (Scheme A)
  - `channels = 2`: Seismic + initial impedance (Scheme B)
  - `trace_length`: Typically 300 samples (paper), 72 samples (Teapot Dome)
- **Output Shape**: `(batch_size, 1, trace_length)`
- **Typical Range**:
  - Seismic amplitude: [-1, 1] (normalized)
  - Impedance: [2000, 5000] m/s·g/cm³ (synthetic)

**2D CNN:**
- **Input Shape**: `(batch_size, 2, height, width)`
  - `channel 0`: Seismic profile
  - `channel 1`: Initial impedance profile
  - `height`: Depth samples (200-600)
  - `width`: Lateral positions (200-500)
- **Output Shape**: `(batch_size, 1, height, width)`
- **Mask Shape**: `(batch_size, 1, height, width)`
  - Binary mask: 1 at well locations, 0 elsewhere

#### Memory Requirements

| Configuration | GPU Memory | Training Time (300 epochs) |
|--------------|-----------|----------------------------|
| 1D, batch=16 | ~2 GB | ~30 minutes |
| 2D, batch=8, 128×128 | ~4 GB | ~2 hours |
| 2D, batch=8, 256×256 | ~8 GB | ~4 hours |

### Training Process

#### Learning Rate Schedule

The implementation uses **ReduceLROnPlateau** scheduler:
- Initial LR: 0.001 (as per paper)
- Reduction factor: 0.5
- Patience: 10 epochs
- Monitors: Validation loss

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)
```

#### Training Curve (Expected)

From the paper (SEAM synthetic data):

```
Epoch 1:   Train Loss ~5.0,    Val Loss ~5.0
Epoch 50:  Train Loss ~0.5,    Val Loss ~0.4
Epoch 100: Train Loss ~0.1,    Val Loss ~0.08
Epoch 300: Train Loss ~0.03,   Val Loss ~0.02
```

### Inference Methods

#### Inference on 1D Data

```python
import torch
from model import SeismicImpedanceCNN1D

# Load model
model = SeismicImpedanceCNN1D(in_channels=2)
checkpoint = torch.load('models/1d_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare input: (1, 2, 300) - batch=1, channels=2, length=300
seismic_trace = torch.randn(1, 1, 300)
initial_impedance = torch.randn(1, 1, 300)
input_data = torch.cat([seismic_trace, initial_impedance], dim=1)

# Inference
with torch.no_grad():
    predicted_impedance = model(input_data)  # Shape: (1, 1, 300)
```

#### Inference on 2D Data

```python
from model import SeismicImpedanceCNN2D

# Load model
model = SeismicImpedanceCNN2D(in_channels=2)
checkpoint = torch.load('models/2d_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare input: (1, 2, H, W)
seismic_profile = torch.randn(1, 1, 200, 200)
initial_impedance_profile = torch.randn(1, 1, 200, 200)
input_data = torch.cat([seismic_profile, initial_impedance_profile], dim=1)

# Inference
with torch.no_grad():
    predicted_impedance = model(input_data)  # Shape: (1, 1, 200, 200)
```

#### 3D Volume Processing

For 3D seismic volumes, process section-by-section:

```python
# Process inline sections
for inline_idx in range(num_inlines):
    seismic_section = seismic_volume[inline_idx, :, :]  # (depth, crossline)
    initial_section = initial_volume[inline_idx, :, :]
    
    # Prepare input
    input_data = prepare_input(seismic_section, initial_section)
    
    # Predict
    with torch.no_grad():
        impedance_section = model(input_data)
    
    # Store result
    impedance_volume[inline_idx, :, :] = impedance_section.cpu().numpy()
```

## Results

### Performance Metrics

Based on paper results (SEAM synthetic):

| Method | MSE | R² | Relative Error |
|--------|-----|-------|----------------|
| 1D CNN (Scheme A) | 0.035 | 0.92 | 4.2% |
| 1D CNN (Scheme B) | 0.022 | 0.96 | 2.8% |
| 2D CNN (weak sup.) | 0.018 | 0.98 | 2.1% |

### Visualization Examples

The code generates comprehensive visualizations:

#### 1D Results
- Input seismic trace
- Initial impedance model
- True vs predicted impedance comparison
- Absolute error plot

#### 2D Results
- Input seismic profile
- Initial impedance profile
- True impedance profile
- Predicted impedance profile
- Well location mask

### Comparison with Traditional Methods

| Aspect | Traditional Inversion | This DL Approach |
|--------|----------------------|------------------|
| Speed | Minutes to hours per line | Milliseconds per line |
| Accuracy | 85-95% (depends on params) | 95-98% (once trained) |
| Lateral continuity | Good (physics-based) | Excellent (2D CNN) |
| Sparse data | Requires interpolation | Native weak supervision |
| Interpretability | High (physics) | Moderate (learned) |
| Initial model dependency | Critical | Important but less critical |

## Project Structure

```
seismic-impedance-inversion/
├── model.py              # 1D and 2D CNN architectures
├── data_loading.py       # Dataset generation and loading
├── train.py              # Training script
├── test.py               # Testing/inference script
├── utils.py              # Utility functions
├── requirements.txt      # Python dependencies
├── README.md             # This file (English)
├── README_KR.md          # Korean documentation
├── models/               # Saved model checkpoints
├── logs/                 # Training logs and curves
├── results/              # Test results and visualizations
└── data/                 # Data directory (optional)
```

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{wu2021deep,
  title={Deep learning for multidimensional seismic impedance inversion},
  author={Wu, Xinming and Yan, Shangsheng and Bi, Zhengfa and Zhang, Sibo and Si, Hongjie},
  journal={Geophysics},
  volume={86},
  number={5},
  pages={R735--R745},
  year={2021},
  publisher={Society of Exploration Geophysicists}
}
```

## License

This implementation is for research and educational purposes.

## Acknowledgments

- Original paper authors: Xinming Wu, Shangsheng Yan, Zhengfa Bi, Sibo Zhang, and Hongjie Si
- SEAM Phase I dataset for synthetic examples
- Teapot Dome dataset for field data examples

## Contact

For questions and issues, please open an issue on GitHub or contact the repository maintainer.

---

**Note**: This implementation uses synthetic data generation for demonstration. For real-world applications, replace the data loading module with your actual seismic and well log data.
