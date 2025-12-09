"""
Combined FRB + RFI Data Generator

This module combines FRB and RFI simulators to generate realistic
training data with multi-class labels for SAM-based detection.

Pipeline: Gaussian Noise → Add RFI → Inject FRB → Generate Multi-class Masks

Classes:
    0: Background/Noise
    1: RFI
    2: FRB
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
import warnings

from .frb_simulator import FRBSimulator
from .rfi_simulator import RFISimulator


class FRBRFIDataGenerator:
    """
    Generates synthetic radio astronomy data with FRBs and RFI.

    This generator creates realistic observational scenarios where FRBs
    may be obscured or coexist with various types of RFI.

    Parameters:
        freq_min (float): Minimum frequency in MHz
        freq_max (float): Maximum frequency in MHz
        n_freq_channels (int): Number of frequency channels (1024 for SAM)
        n_time_bins (int): Number of time bins (1024 for SAM)
        time_resolution (float): Time resolution in ms
        noise_mean (float): Mean of background Gaussian noise
        noise_std (float): Standard deviation of background noise
    """

    def __init__(
        self,
        freq_min: float = 550.0,
        freq_max: float = 750.0,
        n_freq_channels: int = 1024,
        n_time_bins: int = 1024,
        time_resolution: float = 1.3,
        noise_mean: float = 0.0,
        noise_std: float = 1.0,
    ):
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.n_freq_channels = n_freq_channels
        self.n_time_bins = n_time_bins
        self.time_resolution = time_resolution
        self.noise_mean = noise_mean
        self.noise_std = noise_std

        # Initialize simulators
        self.frb_sim = FRBSimulator(
            freq_min=freq_min,
            freq_max=freq_max,
            n_freq_channels=n_freq_channels,
            n_time_bins=n_time_bins,
            time_resolution=time_resolution,
        )

        self.rfi_sim = RFISimulator(
            freq_min=freq_min,
            freq_max=freq_max,
            n_freq_channels=n_freq_channels,
            n_time_bins=n_time_bins,
            time_resolution=time_resolution,
        )

    def generate_noise(self) -> np.ndarray:
        """
        Generate Gaussian background noise.

        Returns:
            np.ndarray: Noise array of shape (n_freq_channels, n_time_bins)
        """
        noise = np.random.normal(
            self.noise_mean,
            self.noise_std,
            size=(self.n_freq_channels, self.n_time_bins)
        )
        return noise

    def generate_sample(
        self,
        dm_range: Tuple[float, float] = (100, 200),
        frb_snr_range: Tuple[float, float] = (5, 15),
        include_frb: bool = True,
        include_rfi: bool = True,
        rfi_params: Optional[Dict] = None,
        frb_params: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Generate a single training sample with noise, RFI, and FRB.

        Pipeline:
        1. Generate Gaussian noise
        2. Add RFI patterns
        3. Inject dispersed FRB
        4. Generate multi-class labels (Background=0, RFI=1, FRB=2)

        Parameters:
            dm_range (Tuple[float, float]): DM range for FRB in pc/cm³
            frb_snr_range (Tuple[float, float]): SNR range for FRB
            include_frb (bool): Whether to include FRB
            include_rfi (bool): Whether to include RFI
            rfi_params (Optional[Dict]): Custom RFI generation parameters
            frb_params (Optional[Dict]): Custom FRB generation parameters

        Returns:
            Tuple[np.ndarray, np.ndarray, Dict]:
                - Dynamic spectrum (freq, time) with noise + RFI + FRB
                - Multi-class mask (0=background, 1=RFI, 2=FRB)
                - Metadata dictionary
        """
        # Step 1: Generate baseline noise
        dynamic_spectrum = self.generate_noise()

        # Initialize masks
        rfi_mask = np.zeros((self.n_freq_channels, self.n_time_bins), dtype=np.uint8)
        frb_mask = np.zeros((self.n_freq_channels, self.n_time_bins), dtype=np.uint8)

        # Metadata
        metadata = {
            'has_frb': include_frb,
            'has_rfi': include_rfi,
            'noise_mean': self.noise_mean,
            'noise_std': self.noise_std,
        }

        # Step 2: Add RFI
        if include_rfi:
            if rfi_params is None:
                # Default random RFI parameters
                rfi_params = {
                    'pers_freq_gauss': np.random.randint(1, 4),
                    'pers_time_gauss': np.random.randint(0, 3),
                    'pers_freq_square': np.random.randint(0, 3),
                    'pers_time_square': np.random.randint(0, 2),
                    'inter_freq_gauss': np.random.randint(0, 3),
                    'inter_freq_square': np.random.randint(0, 2),
                }

            rfi_spectrum, rfi_table = self.rfi_sim.generate_rfi_waterfall(**rfi_params)

            # Add RFI to dynamic spectrum
            dynamic_spectrum += rfi_spectrum

            # Generate RFI mask
            rfi_mask = self.rfi_sim.generate_rfi_mask(rfi_spectrum, threshold=0.1)

            metadata['rfi_table'] = rfi_table
            metadata['rfi_params'] = rfi_params

        # Step 3: Inject FRB
        if include_frb:
            # Random DM and SNR
            dm = np.random.uniform(*dm_range)
            snr = np.random.uniform(*frb_snr_range)

            if frb_params is None:
                frb_params = {}

            # Generate FRB with random parameters
            frb_spectrum, frb_metadata = self.frb_sim.generate_frb_with_snr(
                dm=dm,
                snr=snr,
                noise_std=self.noise_std,
                width=np.random.uniform(3, 8),  # Pulse width 3-8 ms
                **frb_params
            )

            # Add FRB to dynamic spectrum
            dynamic_spectrum += frb_spectrum

            # Generate FRB mask
            frb_mask = self.frb_sim.generate_frb_mask(frb_metadata)

            metadata['frb_metadata'] = frb_metadata
            metadata['frb_dm'] = dm
            metadata['frb_snr'] = snr

        # Step 4: Create multi-class mask
        # Priority: FRB (2) > RFI (1) > Background (0)
        # If FRB and RFI overlap, FRB takes precedence
        multiclass_mask = np.zeros((self.n_freq_channels, self.n_time_bins), dtype=np.uint8)
        multiclass_mask[rfi_mask == 1] = 1  # RFI regions
        multiclass_mask[frb_mask == 1] = 2  # FRB regions (overrides RFI)

        metadata['mask_stats'] = {
            'background_fraction': (multiclass_mask == 0).sum() / multiclass_mask.size,
            'rfi_fraction': (multiclass_mask == 1).sum() / multiclass_mask.size,
            'frb_fraction': (multiclass_mask == 2).sum() / multiclass_mask.size,
        }

        return dynamic_spectrum, multiclass_mask, metadata

    def generate_batch(
        self,
        batch_size: int = 16,
        frb_probability: float = 0.8,
        rfi_probability: float = 0.9,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Generate a batch of training samples.

        Parameters:
            batch_size (int): Number of samples to generate
            frb_probability (float): Probability of including FRB (0-1)
            rfi_probability (float): Probability of including RFI (0-1)
            **kwargs: Additional arguments passed to generate_sample()

        Returns:
            Tuple[np.ndarray, np.ndarray, List[Dict]]:
                - Batch of dynamic spectra (batch_size, freq, time)
                - Batch of masks (batch_size, freq, time)
                - List of metadata dictionaries
        """
        spectra = []
        masks = []
        metadatas = []

        for _ in range(batch_size):
            include_frb = np.random.random() < frb_probability
            include_rfi = np.random.random() < rfi_probability

            spectrum, mask, metadata = self.generate_sample(
                include_frb=include_frb,
                include_rfi=include_rfi,
                **kwargs
            )

            spectra.append(spectrum)
            masks.append(mask)
            metadatas.append(metadata)

        return np.array(spectra), np.array(masks), metadatas

    def generate_dataset(
        self,
        n_samples: int = 1000,
        save_path: Optional[str] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Generate a complete dataset.

        Parameters:
            n_samples (int): Number of samples to generate
            save_path (Optional[str]): Path to save dataset (as .npz file)
            **kwargs: Additional arguments passed to generate_batch()

        Returns:
            Tuple[np.ndarray, np.ndarray, List[Dict]]:
                - All dynamic spectra
                - All masks
                - All metadata
        """
        print(f"Generating dataset with {n_samples} samples...")

        all_spectra = []
        all_masks = []
        all_metadata = []

        batch_size = 16
        n_batches = (n_samples + batch_size - 1) // batch_size

        for i in range(n_batches):
            current_batch_size = min(batch_size, n_samples - i * batch_size)

            spectra, masks, metadata = self.generate_batch(
                batch_size=current_batch_size,
                **kwargs
            )

            all_spectra.append(spectra)
            all_masks.append(masks)
            all_metadata.extend(metadata)

            if (i + 1) % 10 == 0:
                print(f"  Generated {(i + 1) * batch_size}/{n_samples} samples...")

        all_spectra = np.vstack(all_spectra)
        all_masks = np.vstack(all_masks)

        print(f"Dataset generation complete!")
        print(f"  Spectra shape: {all_spectra.shape}")
        print(f"  Masks shape: {all_masks.shape}")

        # Save if path provided
        if save_path:
            np.savez_compressed(
                save_path,
                spectra=all_spectra,
                masks=all_masks,
            )
            print(f"  Saved to {save_path}")

        return all_spectra, all_masks, all_metadata


if __name__ == "__main__":
    # Example usage
    print("Initializing FRB-RFI Data Generator...")
    generator = FRBRFIDataGenerator()

    # Generate a single sample
    print("\nGenerating single sample...")
    spectrum, mask, metadata = generator.generate_sample()

    print(f"Spectrum shape: {spectrum.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Has FRB: {metadata['has_frb']}")
    print(f"Has RFI: {metadata['has_rfi']}")
    print(f"Mask statistics: {metadata['mask_stats']}")

    # Generate a small batch
    print("\nGenerating batch of 8 samples...")
    spectra, masks, metadatas = generator.generate_batch(batch_size=8)
    print(f"Batch spectra shape: {spectra.shape}")
    print(f"Batch masks shape: {masks.shape}")

    # Count samples with FRB
    n_with_frb = sum(m['has_frb'] for m in metadatas)
    print(f"Samples with FRB: {n_with_frb}/8")
