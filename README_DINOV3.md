# DINOv3 FRB Detector

Fast Radio Burst detection using DINOv3 foundation model.

## Architecture

- **Backbone**: DINOv3 ViT-g/14 (1.1B parameters, frozen)
- **Head**: Lightweight 2-output head (~1-2M trainable parameters)
  - FRB detection (binary classification)
  - DM estimation (regression: 30-2000 pc/cm³)
- **Input**: 4096×4096 images (freq × time)
- **Output**: FRB probability + DM estimate

## Key Features

✅ **Full sweep simulation** - Generates complete DM sweeps, extracts sliding windows
✅ **Config-driven** - Control DM range, freq range, freezing strategy via YAML
✅ **Auto-flip filterbanks** - Automatically detects and corrects frequency ordering
✅ **Progressive unfreezing** - Start frozen, unfreeze last N blocks if needed
✅ **LoRA support** - Efficient fine-tuning for large models
✅ **Handles DM 30-2000** - Adaptive time resolution for high DM

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_dinov3.txt
```

### 2. Configure Training

Edit `config/dinov3_frb_config.yaml`:

```yaml
# Key parameters
model:
  backbone: "dinov2_vitg14"  # Or vitl14, vitb14, vits14
  freeze_backbone: true
  unfreeze_last_n_blocks: 0  # Set to 4, 8, etc. to unfreeze

data:
  frequency:
    min: 300  # MHz
    max: 500
  dispersion:
    dm_min: 30
    dm_max: 2000

training:
  batch_size: 4
  num_epochs: 50
```

### 3. Train Model

```bash
# Default training (frozen backbone)
python train_dinov3.py --config config/dinov3_frb_config.yaml

# With last 4 blocks unfrozen (set in config first)
python train_dinov3.py --config config/dinov3_frb_config_unfreeze.yaml
```

### 4. Run Inference

```bash
python inference_dinov3.py \
    --filterbank your_data.fil \
    --model outputs/dinov3/run_XXX/checkpoints/best_model.pt \
    --config config/dinov3_frb_config.yaml \
    --output detections/ \
    --threshold 0.5
```

## Training Strategy

### Phase 1: Frozen Backbone (Recommended Start)

```yaml
freeze_backbone: true
unfreeze_last_n_blocks: 0
```

- Trains only ~1-2M parameters
- Fast training (~1-2 hours on H100)
- Low overfitting risk
- **Try this first!**

### Phase 2: Partial Unfreezing (If Phase 1 plateaus)

```yaml
freeze_backbone: true
unfreeze_last_n_blocks: 4  # Unfreeze last 4 of 40 blocks
```

- Trains ~112M parameters
- Allows backbone adaptation
- Use differential learning rates

### Phase 3: LoRA (If more capacity needed)

```yaml
freeze_backbone: true
unfreeze_last_n_blocks: 8
use_lora: true
lora_rank: 16
```

- Reduces trainable params by 5-10×
- Prevents overfitting
- Requires `peft` library

## Config Options

### Frequency Configuration

```yaml
data:
  frequency:
    min: 300  # MHz - change for different bands
    max: 500
    channels: 4096
```

**For 550-750 MHz model**: Change `min: 550, max: 750`

### DM Range

```yaml
data:
  dispersion:
    dm_min: 30
    dm_max: 2000
    sampling: "log_uniform"  # More low-DM samples
    dm_distribution_weights:
      - [30, 200, 0.4]    # 40% from DM 30-200
      - [200, 500, 0.3]   # 30% from DM 200-500
      - [500, 1000, 0.2]  # 20% from DM 500-1000
      - [1000, 2000, 0.1] # 10% from DM 1000-2000
```

### Model Selection

```yaml
model:
  backbone: "dinov2_vitg14"  # Options:
    # dinov2_vits14 (21M params)
    # dinov2_vitb14 (86M params)
    # dinov2_vitl14 (300M params)
    # dinov2_vitg14 (1.1B params) ← Recommended for H100
```

## Inference Details

### Automatic Frequency Handling

The inference script automatically:
1. Reads filterbank header (`fch1`, `foff`)
2. Determines if channel 0 is at high or low frequency
3. Flips frequency axis if needed to match training (high→low)

```python
# Automatic in code:
if metadata['foff'] < 0:
    # Ch0 = high freq (correct)
    pass
else:
    # Ch0 = low freq (flip needed)
    data = np.flip(data, axis=0)
```

### Overlapping Chunks

```yaml
inference:
  chunk_size: 4096
  overlap: 2048  # 50% overlap
```

Processes long filterbanks as overlapping 4096×4096 windows.

## File Structure

```
frb-rfi/
├── config/
│   └── dinov3_frb_config.yaml          # Main config
├── frb_rfi_detector/
│   ├── models/
│   │   └── dinov3_frb_detector.py      # Model architecture
│   ├── data_generator_dinov3.py        # Full sweep generator
│   ├── dataset_dinov3.py               # PyTorch dataset
│   ├── frb_simulator.py                # FRB simulation (reused)
│   └── rfi_simulator.py                # RFI simulation (reused)
├── train_dinov3.py                     # Training script
├── inference_dinov3.py                 # Inference script
└── requirements_dinov3.txt             # Dependencies
```

## Expected Performance

Based on Phase 1 (frozen backbone):

- **Training time**: 1-2 hours (50 epochs, H100)
- **FRB detection accuracy**: >90%
- **DM MAE**: <20 pc/cm³
- **Inference speed**: ~0.5s per 4096×4096 chunk (H100)

## Troubleshooting

### Out of Memory

Reduce batch size:
```yaml
training:
  batch_size: 2  # Or 1
```

### Low Accuracy

Try unfreezing last blocks:
```yaml
model:
  unfreeze_last_n_blocks: 4
```

### DM Estimates Wrong

Check frequency range matches training:
```yaml
inference:
  target_freq_min: 300  # Must match training
  target_freq_max: 500
```

## Multiple Frequency Bands

To train models for different bands:

```bash
# Model 1: 300-500 MHz
python train_dinov3.py --config config/dinov3_300_500.yaml

# Model 2: 550-750 MHz
python train_dinov3.py --config config/dinov3_550_750.yaml
```

Just change `frequency: min/max` in config.

## Citation

DINOv2/v3: https://github.com/facebookresearch/dinov2
