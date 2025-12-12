"""
PyTorch Dataset for DINOv3 FRB detection.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Optional, Tuple
import yaml

from .data_generator_dinov3 import FullSweepFRBGenerator


class FRBDataset(Dataset):
    """
    PyTorch dataset for FRB detection with on-the-fly generation.
    """

    def __init__(
        self,
        config: Dict,
        num_samples: int = 10000,
        seed: Optional[int] = None,
    ):
        """
        Initialize dataset.

        Args:
            config: Configuration dict with data parameters
            num_samples: Number of samples per epoch
            seed: Random seed
        """
        self.config = config
        self.num_samples = num_samples
        self.seed = seed

        # Extract config
        data_config = config['data']
        freq_config = data_config['frequency']
        time_config = data_config['time']
        dm_config = data_config['dispersion']
        frb_config = data_config['frb']
        rfi_config = data_config['rfi']

        # Initialize generator
        self.generator = FullSweepFRBGenerator(
            freq_min=freq_config['min'],
            freq_max=freq_config['max'],
            freq_channels=freq_config['channels'],
            time_resolution=time_config['resolution'],
            window_time_bins=time_config['bins'],
            dm_range=(dm_config['dm_min'], dm_config['dm_max']),
            dm_sampling=dm_config['sampling'],
            dm_distribution_weights=dm_config.get('dm_distribution_weights'),
            frb_snr_range=tuple(frb_config['snr_range']),
            frb_width_range=tuple(frb_config['width_range']),
            rfi_probability=rfi_config['probability'],
            rfi_patterns=rfi_config['patterns'],
            window_sampling=frb_config.get('window_sampling', 'random'),
            partial_sweep_probability=frb_config.get('partial_sweep_probability', 0.7),
            seed=seed,
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get one sample.

        Returns:
            image: [3, H, W] RGB tensor (grayscale replicated)
            frb_label: [1] tensor (0 or 1)
            dm_value: [1] tensor (DM in pc/cm³)
        """
        # Generate sample
        image, frb_label, dm_value = self.generator.generate_sample()

        # Normalize image to [0, 1]
        image_norm = self._normalize(image)

        # Convert to RGB (replicate grayscale to 3 channels)
        image_rgb = np.stack([image_norm, image_norm, image_norm], axis=0)

        # Convert to tensors
        image_tensor = torch.from_numpy(image_rgb).float()
        frb_label_tensor = torch.tensor([frb_label], dtype=torch.float32)
        dm_value_tensor = torch.tensor([dm_value], dtype=torch.float32)

        return image_tensor, frb_label_tensor, dm_value_tensor

    def _normalize(self, image: np.ndarray, percentile: float = 99.5) -> np.ndarray:
        """
        Normalize image to [0, 1] range using percentile clipping.

        Args:
            image: Input image
            percentile: Percentile for clipping

        Returns:
            Normalized image
        """
        vmin = np.percentile(image, 100 - percentile)
        vmax = np.percentile(image, percentile)

        image_norm = (image - vmin) / (vmax - vmin + 1e-8)
        image_norm = np.clip(image_norm, 0, 1)

        return image_norm.astype(np.float32)


def create_dataloaders(
    config: Dict,
    samples_per_epoch: int = 10000,
    val_split: float = 0.15,
    batch_size: int = 4,
    num_workers: int = 8,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        config: Configuration dict
        samples_per_epoch: Total samples per epoch
        val_split: Validation split ratio
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        train_loader, val_loader
    """
    # Calculate splits
    val_samples = int(samples_per_epoch * val_split)
    train_samples = samples_per_epoch - val_samples

    # Create datasets
    train_dataset = FRBDataset(
        config=config,
        num_samples=train_samples,
        seed=42,
    )

    val_dataset = FRBDataset(
        config=config,
        num_samples=val_samples,
        seed=43,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    print("Testing FRBDataset...")

    # Load config
    with open('config/dinov3_frb_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create dataset
    dataset = FRBDataset(config=config, num_samples=100)

    print(f"Dataset size: {len(dataset)}")

    # Get sample
    image, frb_label, dm_value = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"FRB label: {frb_label.item()}")
    print(f"DM value: {dm_value.item():.1f}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        config=config,
        samples_per_epoch=100,
        val_split=0.2,
        batch_size=2,
        num_workers=0,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Test batch loading
    images, frb_labels, dm_values = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Images: {images.shape}")
    print(f"  FRB labels: {frb_labels.shape}")
    print(f"  DM values: {dm_values.shape}")

    print("\n✅ Dataset test passed!")
