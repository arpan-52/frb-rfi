# FRB-RFI Detector

**SAM-based Fast Radio Burst Detection in the Presence of Radio Frequency Interference**

This project implements a deep learning pipeline for detecting Fast Radio Bursts (FRBs) in radio astronomy data contaminated with Radio Frequency Interference (RFI). It uses Meta's Segment Anything Model (SAM) for multi-class segmentation.

## Overview

### The Problem
Fast Radio Bursts are transient radio pulses that are often obscured by various types of Radio Frequency Interference. Traditional detection methods struggle to distinguish FRBs from RFI, especially when they overlap in time-frequency space.

### Our Solution
We use a multi-class segmentation approach with SAM to simultaneously detect and classify:
- **Class 0**: Background/Noise
- **Class 1**: RFI (various types)
- **Class 2**: FRB signals

### Pipeline

```
Gaussian Noise → Add RFI Patterns → Inject Dispersed FRBs → SAM Detection
```

## Features

### FRB Simulation
- **Realistic dispersion**: Implements proper cold plasma dispersion with DM range 100-200 pc/cm³
- **Frequency sweep**: FRBs arrive at high frequencies first, sweep to low frequencies
- **Edge cases**: FRBs can appear anywhere, including partial visibility at edges
- **Observational parameters**:
  - Frequency: 550-750 MHz (200 MHz bandwidth)
  - Time resolution: 1.3 ms
  - Array size: 1024×1024 (optimized for SAM)

### RFI Simulation
Based on [SAM-RFI](https://github.com/preshanth/SAM-RFI) implementation:
- **Persistent RFI**: Frequency-domain and time-domain (narrowband, broadband)
- **Intermittent RFI**: Periodic signals with duty cycles
- **Multiple shapes**: Gaussian and square wave patterns
- **Realistic parameters**: Randomly sampled amplitudes, bandwidths, periods

### Model Architecture
- **Encoder**: Pre-trained SAM ViT (vit_b/vit_l/vit_h)
- **Decoder**: Custom multi-class segmentation head
- **Loss**: Combined Cross-Entropy + Dice Loss
- **Training**: Mixed precision, learning rate scheduling

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- PyTorch 2.0+

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/arpan-52/frb-rfi.git
cd frb-rfi
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Install Segment Anything**:
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

5. **Download SAM checkpoint** (optional but recommended):
```bash
# Download ViT-B checkpoint (~375 MB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# Or ViT-L checkpoint (~1.2 GB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

# Or ViT-H checkpoint (~2.4 GB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Quick Start

### Generate Example Data

```python
from frb_rfi_detector.data_generator import FRBRFIDataGenerator

# Initialize generator
generator = FRBRFIDataGenerator()

# Generate a single sample
spectrum, mask, metadata = generator.generate_sample()

print(f"Spectrum shape: {spectrum.shape}")  # (1024, 1024)
print(f"Has FRB: {metadata['has_frb']}")
print(f"FRB DM: {metadata.get('frb_dm', 'N/A')}")
```

### Visualize Data

```python
from frb_rfi_detector.visualizations.plotting import plot_examples

# Generate and plot examples
plot_examples(generator, n_examples=4, save_dir='examples')
```

### Train Model

```bash
# Basic training with on-the-fly data generation
python frb_rfi_detector/train.py \
    --model_type vit_b \
    --checkpoint sam_vit_b_01ec64.pth \
    --train_size 5000 \
    --val_size 500 \
    --batch_size 4 \
    --epochs 50 \
    --lr 1e-4 \
    --output_dir outputs

# Advanced training with custom parameters
python frb_rfi_detector/train.py \
    --model_type vit_l \
    --checkpoint sam_vit_l_0b3195.pth \
    --train_size 10000 \
    --val_size 1000 \
    --batch_size 8 \
    --epochs 100 \
    --lr 5e-5 \
    --dm_min 100 \
    --dm_max 200 \
    --snr_min 5 \
    --snr_max 15 \
    --dice_weight 0.5 \
    --output_dir outputs/run_vit_l
```

### Inference

```python
import torch
from frb_rfi_detector.models import SAMFRBDetector

# Load trained model
model = SAMFRBDetector(model_type='vit_b', num_classes=3)
checkpoint = torch.load('outputs/checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate test data
spectrum, ground_truth, _ = generator.generate_sample()

# Prepare input
from frb_rfi_detector.dataset import FRBRFIDataset
spectrum_norm = FRBRFIDataset.normalize_spectrum(spectrum)
image = torch.from_numpy(spectrum_norm).float().unsqueeze(0).repeat(1, 3, 1, 1)

# Predict
with torch.no_grad():
    prediction, probabilities = model.predict(image)

# Visualize
from frb_rfi_detector.visualizations.plotting import plot_segmentation_comparison
plot_segmentation_comparison(
    spectrum, ground_truth, prediction[0].numpy(),
    title='FRB Detection Result',
    save_path='prediction.png'
)
```

## Project Structure

```
frb-rfi/
├── frb_rfi_detector/
│   ├── frb_simulator.py          # FRB dispersion simulation
│   ├── rfi_simulator.py          # RFI pattern generation
│   ├── data_generator.py         # Combined data pipeline
│   ├── dataset.py                # PyTorch dataset classes
│   ├── train.py                  # Training script
│   ├── models/
│   │   ├── __init__.py
│   │   └── sam_frb_detector.py   # SAM-based model
│   ├── utils/
│   │   ├── __init__.py
│   │   └── metrics.py            # Evaluation metrics
│   └── visualizations/
│       └── plotting.py           # Visualization utilities
├── notebooks/                     # Jupyter notebooks
├── outputs/                       # Training outputs
├── requirements.txt
├── setup.py
└── README.md
```

## Physics Background

### FRB Dispersion

FRBs experience dispersion as they travel through ionized plasma. The dispersion delay is:

```
Δt = 4.15 ms × DM × [(ν₁/GHz)⁻² - (ν₂/GHz)⁻²]
```

Where:
- `DM` = Dispersion Measure (pc/cm³)
- `ν₁` = Reference frequency (GHz)
- `ν₂` = Observed frequency (GHz)

In our simulation:
- DM range: 100-200 pc/cm³
- Frequency range: 550-750 MHz (0.55-0.75 GHz)
- Maximum sweep time (DM=200): ~1268 ms

### RFI Types

The simulator generates realistic RFI based on actual observations:

1. **Persistent Narrowband RFI**: Fixed frequency transmitters (cell towers, satellites)
2. **Persistent Broadband RFI**: Continuous wideband interference
3. **Intermittent RFI**: Periodic signals (radar, pulsed transmitters)
4. **Transient RFI**: Short-duration bursts (can mimic FRBs!)

## Metrics

The model is evaluated using:

- **Overall Metrics**:
  - Accuracy
  - Mean IoU (Intersection over Union)

- **Per-Class Metrics**:
  - IoU (Dice coefficient)
  - Precision, Recall, F1 Score

Focus metric: **FRB F1 Score** - measures ability to correctly detect FRBs without false positives.

## Training Tips

1. **Start small**: Use `vit_b` model for initial experiments
2. **Batch size**: Reduce if running out of GPU memory (1024×1024 images are large!)
3. **Data generation**: On-the-fly generation saves disk space but increases CPU usage
4. **Mixed precision**: Enabled by default, reduces memory by ~40%
5. **Checkpointing**: Best model saved based on validation mIoU

## Citation

If you use this code, please cite:

```bibtex
@software{frb_rfi_detector,
  title={FRB-RFI Detector: SAM-based Fast Radio Burst Detection},
  author={Your Name},
  year={2025},
  url={https://github.com/arpan-52/frb-rfi}
}
```

## Acknowledgments

- **SAM-RFI**: RFI simulation based on [preshanth/SAM-RFI](https://github.com/preshanth/SAM-RFI)
- **Segment Anything**: Model from [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Contact

For questions or issues, please open a GitHub issue or contact [your email].

---

**Note**: This is a research project. For production FRB detection, additional validation with real observational data is required.
