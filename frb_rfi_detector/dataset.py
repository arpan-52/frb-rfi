"""
PyTorch Dataset for FRB-RFI Detection

This module provides PyTorch dataset classes for training SAM on FRB detection
in the presence of RFI.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, List
import os

from .data_generator import FRBRFIDataGenerator


class FRBRFIDataset(Dataset):
    """
    PyTorch Dataset for FRB-RFI detection.

    Can either:
    1. Load pre-generated data from disk
    2. Generate data on-the-fly during training

    Parameters:
        data_path (Optional[str]): Path to .npz file with pre-generated data
        n_samples (int): Number of samples (if generating on-the-fly)
        generator (Optional[FRBRFIDataGenerator]): Data generator instance
        transform (Optional): Transformations to apply
        generate_on_fly (bool): Generate data during training (default: False)
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        n_samples: int = 1000,
        generator: Optional[FRBRFIDataGenerator] = None,
        transform: Optional = None,
        generate_on_fly: bool = False,
        dm_range: Tuple[float, float] = (100, 200),
        frb_snr_range: Tuple[float, float] = (5, 15),
        frb_probability: float = 0.8,
        rfi_probability: float = 0.9,
    ):
        self.transform = transform
        self.generate_on_fly = generate_on_fly
        self.dm_range = dm_range
        self.frb_snr_range = frb_snr_range
        self.frb_probability = frb_probability
        self.rfi_probability = rfi_probability

        if generate_on_fly:
            # On-the-fly generation
            self.n_samples = n_samples
            self.generator = generator if generator is not None else FRBRFIDataGenerator()
            self.spectra = None
            self.masks = None
            print(f"Dataset initialized for on-the-fly generation ({n_samples} samples per epoch)")

        elif data_path and os.path.exists(data_path):
            # Load from disk
            data = np.load(data_path)
            self.spectra = data['spectra']
            self.masks = data['masks']
            self.n_samples = len(self.spectra)
            self.generator = None
            print(f"Dataset loaded from {data_path}")
            print(f"  Loaded {self.n_samples} samples")
            print(f"  Spectra shape: {self.spectra.shape}")
            print(f"  Masks shape: {self.masks.shape}")

        else:
            raise ValueError("Either provide data_path or set generate_on_fly=True")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Image: (3, H, W) tensor (RGB format for SAM)
                - Mask: (H, W) tensor with class labels
        """
        if self.generate_on_fly:
            # Generate sample on-the-fly
            include_frb = np.random.random() < self.frb_probability
            include_rfi = np.random.random() < self.rfi_probability

            spectrum, mask, _ = self.generator.generate_sample(
                dm_range=self.dm_range,
                frb_snr_range=self.frb_snr_range,
                include_frb=include_frb,
                include_rfi=include_rfi,
            )
        else:
            # Load from pre-generated data
            spectrum = self.spectra[idx]
            mask = self.masks[idx]

        # Normalize spectrum to [0, 1] range
        spectrum_norm = self.normalize_spectrum(spectrum)

        # Resize to 1024×1024 if needed (SAM requires square input)
        if spectrum_norm.shape != (1024, 1024):
            import cv2
            spectrum_norm = cv2.resize(spectrum_norm, (1024, 1024), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask.astype(np.float32), (1024, 1024), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

        # Convert to RGB format (SAM expects 3-channel images)
        # Repeat grayscale across 3 channels
        image = np.stack([spectrum_norm, spectrum_norm, spectrum_norm], axis=0)

        # Convert to torch tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

    @staticmethod
    def normalize_spectrum(spectrum: np.ndarray, percentile: float = 99.5) -> np.ndarray:
        """
        Normalize spectrum to [0, 1] range using percentile clipping.

        Parameters:
            spectrum (np.ndarray): Input spectrum
            percentile (float): Percentile for clipping (default: 99.5)

        Returns:
            np.ndarray: Normalized spectrum
        """
        # Clip outliers
        vmin = np.percentile(spectrum, 100 - percentile)
        vmax = np.percentile(spectrum, percentile)

        # Normalize to [0, 1]
        spectrum_norm = (spectrum - vmin) / (vmax - vmin + 1e-8)
        spectrum_norm = np.clip(spectrum_norm, 0, 1)

        return spectrum_norm


class FRBRFIDataModule:
    """
    Data module for managing train/val/test datasets and dataloaders.

    Parameters:
        train_size (int): Number of training samples
        val_size (int): Number of validation samples
        test_size (int): Number of test samples
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of workers for dataloaders
        generate_on_fly (bool): Generate data on-the-fly
        data_dir (Optional[str]): Directory with pre-generated data
    """

    def __init__(
        self,
        train_size: int = 5000,
        val_size: int = 500,
        test_size: int = 500,
        batch_size: int = 8,
        num_workers: int = 4,
        generate_on_fly: bool = True,
        data_dir: Optional[str] = None,
        dm_range: Tuple[float, float] = (100, 200),
        frb_snr_range: Tuple[float, float] = (5, 15),
    ):
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.generate_on_fly = generate_on_fly
        self.data_dir = data_dir
        self.dm_range = dm_range
        self.frb_snr_range = frb_snr_range

        # Initialize generator
        self.generator = FRBRFIDataGenerator()

        # Create datasets
        self.setup()

    def setup(self):
        """Setup train, validation, and test datasets."""
        if self.generate_on_fly:
            # On-the-fly generation
            self.train_dataset = FRBRFIDataset(
                n_samples=self.train_size,
                generator=self.generator,
                generate_on_fly=True,
                dm_range=self.dm_range,
                frb_snr_range=self.frb_snr_range,
                frb_probability=0.8,
                rfi_probability=0.9,
            )

            self.val_dataset = FRBRFIDataset(
                n_samples=self.val_size,
                generator=self.generator,
                generate_on_fly=True,
                dm_range=self.dm_range,
                frb_snr_range=self.frb_snr_range,
                frb_probability=0.8,
                rfi_probability=0.9,
            )

            self.test_dataset = FRBRFIDataset(
                n_samples=self.test_size,
                generator=self.generator,
                generate_on_fly=True,
                dm_range=self.dm_range,
                frb_snr_range=self.frb_snr_range,
                frb_probability=0.8,
                rfi_probability=0.9,
            )

        else:
            # Load from disk
            if not self.data_dir:
                raise ValueError("data_dir must be specified when not generating on-the-fly")

            self.train_dataset = FRBRFIDataset(
                data_path=os.path.join(self.data_dir, 'train.npz')
            )
            self.val_dataset = FRBRFIDataset(
                data_path=os.path.join(self.data_dir, 'val.npz')
            )
            self.test_dataset = FRBRFIDataset(
                data_path=os.path.join(self.data_dir, 'test.npz')
            )

    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


if __name__ == "__main__":
    print("Testing FRB-RFI Dataset...")

    # Test on-the-fly generation
    dataset = FRBRFIDataset(
        n_samples=10,
        generate_on_fly=True,
    )

    print(f"\nDataset length: {len(dataset)}")

    # Get a sample
    image, mask = dataset[0]
    print(f"Image shape: {image.shape}")  # Should be (3, 1024, 1024)
    print(f"Mask shape: {mask.shape}")    # Should be (1024, 1024)
    print(f"Image dtype: {image.dtype}")
    print(f"Mask dtype: {mask.dtype}")
    print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"Mask unique values: {torch.unique(mask)}")

    # Test dataloader
    print("\nTesting DataLoader...")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for batch_images, batch_masks in dataloader:
        print(f"Batch images shape: {batch_images.shape}")
        print(f"Batch masks shape: {batch_masks.shape}")
        break
