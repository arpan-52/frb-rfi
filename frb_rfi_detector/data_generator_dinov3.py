"""
Data generator for DINOv3 FRB detector.

Generates full DM sweeps and extracts sliding windows for training.
Reuses RFI simulation from SAM-RFI code.
"""

import numpy as np
from typing import Tuple, Dict, Optional
import warnings

# Reuse existing simulators
from .frb_simulator import FRBSimulator
from .rfi_simulator import RFISimulator


class FullSweepFRBGenerator:
    """
    Generate FRBs with full dispersion sweeps, then extract windows.

    This generator:
    1. Simulates FULL DM sweep (may be >> 4096 bins for high DM)
    2. Adds RFI patterns
    3. Extracts 4096×4096 windows (random, center, etc.)
    4. Returns: image, FRB presence label, DM value
    """

    def __init__(
        self,
        freq_min: float = 300.0,
        freq_max: float = 500.0,
        freq_channels: int = 4096,
        time_resolution: float = 1.3,
        window_time_bins: int = 4096,
        dm_range: Tuple[float, float] = (30, 2000),
        dm_sampling: str = 'log_uniform',
        dm_distribution_weights: Optional[list] = None,
        frb_snr_range: Tuple[float, float] = (5, 50),
        frb_width_range: Tuple[float, float] = (1, 10),
        rfi_probability: float = 0.8,
        rfi_patterns: Optional[list] = None,
        window_sampling: str = 'random',
        partial_sweep_probability: float = 0.7,
        seed: Optional[int] = None,
    ):
        """
        Initialize generator.

        Args:
            freq_min, freq_max: Frequency range in MHz
            freq_channels: Number of frequency channels
            time_resolution: Time resolution in ms
            window_time_bins: Size of extracted window (4096)
            dm_range: (min_dm, max_dm) in pc/cm³
            dm_sampling: 'uniform' or 'log_uniform'
            dm_distribution_weights: List of [dm_min, dm_max, weight] for sampling
            frb_snr_range: SNR range for FRBs
            frb_width_range: FRB width range in ms
            rfi_probability: Probability of adding RFI
            rfi_patterns: List of RFI pattern types
            window_sampling: How to extract window ('random', 'center', 'start', 'end')
            partial_sweep_probability: Probability of extracting partial sweep
            seed: Random seed
        """
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.freq_channels = freq_channels
        self.time_resolution = time_resolution
        self.window_time_bins = window_time_bins
        self.dm_min, self.dm_max = dm_range
        self.dm_sampling = dm_sampling
        self.dm_distribution_weights = dm_distribution_weights
        self.frb_snr_range = frb_snr_range
        self.frb_width_range = frb_width_range
        self.rfi_probability = rfi_probability
        self.rfi_patterns = rfi_patterns or [
            'persistent_freq_gaussian',
            'persistent_freq_square',
            'persistent_time_gaussian',
            'persistent_time_square',
            'intermittent_gaussian',
            'intermittent_square',
        ]
        self.window_sampling = window_sampling
        self.partial_sweep_probability = partial_sweep_probability

        if seed is not None:
            np.random.seed(seed)

        # Initialize FRB simulator (will be used for dispersion calculation)
        self.frb_simulator = FRBSimulator(
            freq_min=freq_min,
            freq_max=freq_max,
            n_freq_bins=freq_channels,
            n_time_bins=window_time_bins,  # Will override per sample
            time_resolution=time_resolution,
        )

        # Initialize RFI simulator
        self.rfi_simulator = RFISimulator(
            freq_min=freq_min,
            freq_max=freq_max,
            n_freq_bins=freq_channels,
            n_time_bins=window_time_bins,  # Will override per sample
            time_resolution=time_resolution,
        )

    def sample_dm(self) -> float:
        """Sample DM value according to configured distribution."""
        if self.dm_distribution_weights:
            # Weighted sampling
            ranges = []
            weights = []
            for dm_min, dm_max, weight in self.dm_distribution_weights:
                ranges.append((dm_min, dm_max))
                weights.append(weight)

            weights = np.array(weights)
            weights /= weights.sum()

            # Select range
            range_idx = np.random.choice(len(ranges), p=weights)
            dm_min, dm_max = ranges[range_idx]

            # Sample within range
            if self.dm_sampling == 'log_uniform':
                dm = np.exp(np.random.uniform(np.log(dm_min), np.log(dm_max)))
            else:
                dm = np.random.uniform(dm_min, dm_max)
        else:
            # Simple sampling
            if self.dm_sampling == 'log_uniform':
                dm = np.exp(np.random.uniform(np.log(self.dm_min), np.log(self.dm_max)))
            else:
                dm = np.random.uniform(self.dm_min, self.dm_max)

        return dm

    def calculate_sweep_bins(self, dm: float) -> int:
        """
        Calculate number of time bins needed for full DM sweep.

        Uses correct dispersion formula:
        Δt = 4.15 × DM × [(ν_low)^-2 - (ν_high)^-2] ms
        """
        freq_high_ghz = self.freq_max / 1000.0
        freq_low_ghz = self.freq_min / 1000.0

        sweep_time_ms = 4.15 * dm * ((freq_low_ghz ** -2) - (freq_high_ghz ** -2))
        sweep_bins = int(np.ceil(sweep_time_ms / self.time_resolution))

        return max(sweep_bins, self.window_time_bins)

    def generate_sample(self) -> Tuple[np.ndarray, int, float]:
        """
        Generate one training sample.

        Returns:
            image: [freq_channels, window_time_bins] array
            frb_present: 0 or 1
            dm_value: DM in pc/cm³ (0 if no FRB)
        """
        # Decide if this sample has FRB
        has_frb = np.random.rand() < 0.5  # 50% have FRBs

        if has_frb:
            dm = self.sample_dm()
            sweep_bins = self.calculate_sweep_bins(dm)
        else:
            dm = 0.0
            sweep_bins = self.window_time_bins

        # Create full spectrum
        full_spectrum = np.random.randn(self.freq_channels, sweep_bins).astype(np.float32)

        # Add RFI
        if np.random.rand() < self.rfi_probability:
            # Temporarily override RFI simulator time bins
            self.rfi_simulator.n_time_bins = sweep_bins
            rfi_mask = self.rfi_simulator.generate_rfi(
                n_patterns=np.random.randint(1, 5)
            )
            full_spectrum += rfi_mask

        # Add FRB if present
        if has_frb:
            # Temporarily override FRB simulator time bins
            self.frb_simulator.n_time_bins = sweep_bins

            # Sample FRB parameters
            snr = np.random.uniform(*self.frb_snr_range)
            width_ms = np.random.uniform(*self.frb_width_range)

            # Generate FRB
            frb_spectrum = self.frb_simulator.generate_frb(
                dm=dm,
                snr=snr,
                width_ms=width_ms,
            )

            full_spectrum += frb_spectrum

        # Extract window
        window = self._extract_window(full_spectrum, has_frb, sweep_bins)

        return window, int(has_frb), dm

    def _extract_window(
        self,
        full_spectrum: np.ndarray,
        has_frb: bool,
        sweep_bins: int
    ) -> np.ndarray:
        """
        Extract 4096×4096 window from full spectrum.

        Args:
            full_spectrum: [freq_channels, sweep_bins]
            has_frb: Whether FRB is present
            sweep_bins: Total time bins in full spectrum

        Returns:
            window: [freq_channels, window_time_bins]
        """
        if sweep_bins <= self.window_time_bins:
            # Sweep fits in window - pad to exact size
            pad_width = self.window_time_bins - sweep_bins
            window = np.pad(
                full_spectrum,
                ((0, 0), (0, pad_width)),
                mode='constant',
                constant_values=0
            )
        else:
            # Sweep longer than window - extract subset
            if not has_frb:
                # No FRB - extract random window
                start = np.random.randint(0, sweep_bins - self.window_time_bins + 1)
            else:
                # Has FRB - decide if we want full or partial sweep
                if np.random.rand() < self.partial_sweep_probability:
                    # Extract partial sweep
                    if self.window_sampling == 'random':
                        start = np.random.randint(0, sweep_bins - self.window_time_bins + 1)
                    elif self.window_sampling == 'center':
                        start = (sweep_bins - self.window_time_bins) // 2
                    elif self.window_sampling == 'start':
                        start = 0
                    elif self.window_sampling == 'end':
                        start = sweep_bins - self.window_time_bins
                    else:
                        start = np.random.randint(0, sweep_bins - self.window_time_bins + 1)
                else:
                    # Try to get full sweep if possible (shouldn't happen often)
                    start = 0

            window = full_spectrum[:, start:start + self.window_time_bins]

        return window

    def generate_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Generate a batch of samples.

        Returns:
            Dict with keys:
                'images': [batch_size, freq_channels, time_bins]
                'frb_labels': [batch_size] - 0 or 1
                'dm_values': [batch_size] - DM values
        """
        images = []
        frb_labels = []
        dm_values = []

        for _ in range(batch_size):
            image, frb_label, dm = self.generate_sample()
            images.append(image)
            frb_labels.append(frb_label)
            dm_values.append(dm)

        return {
            'images': np.stack(images, axis=0),
            'frb_labels': np.array(frb_labels, dtype=np.float32),
            'dm_values': np.array(dm_values, dtype=np.float32),
        }


if __name__ == "__main__":
    # Test generator
    print("Testing FullSweepFRBGenerator...")

    generator = FullSweepFRBGenerator(
        freq_min=300,
        freq_max=500,
        freq_channels=4096,
        time_resolution=1.3,
        window_time_bins=4096,
        dm_range=(30, 2000),
        dm_sampling='log_uniform',
    )

    # Test DM sweep calculation
    for dm in [30, 100, 200, 500, 1000, 2000]:
        sweep_bins = generator.calculate_sweep_bins(dm)
        sweep_time_ms = sweep_bins * 1.3
        print(f"DM={dm:4d} → {sweep_bins:6d} bins ({sweep_time_ms:8.1f} ms)")

    # Generate sample
    print("\nGenerating test sample...")
    image, frb_label, dm = generator.generate_sample()
    print(f"Image shape: {image.shape}")
    print(f"FRB present: {frb_label}")
    print(f"DM value: {dm:.1f}")

    # Generate batch
    print("\nGenerating test batch...")
    batch = generator.generate_batch(batch_size=4)
    print(f"Images shape: {batch['images'].shape}")
    print(f"FRB labels: {batch['frb_labels']}")
    print(f"DM values: {batch['dm_values']}")

    print("\n✅ Generator test passed!")
