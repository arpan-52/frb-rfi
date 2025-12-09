"""
FRB (Fast Radio Burst) Dispersion Simulator

This module simulates FRB signals with proper dispersion delay across frequency channels.
The dispersion follows the cold plasma dispersion relation.
"""

import numpy as np
from typing import Tuple, Optional


class FRBSimulator:
    """
    Simulates FRB signals with realistic dispersion across frequency channels.

    Parameters:
        freq_min (float): Minimum frequency in MHz (default: 550 MHz)
        freq_max (float): Maximum frequency in MHz (default: 750 MHz)
        n_freq_channels (int): Number of frequency channels (default: 1024)
        n_time_bins (int): Number of time bins (default: 1024)
        time_resolution (float): Time resolution in milliseconds (default: 1.3 ms)
    """

    def __init__(
        self,
        freq_min: float = 550.0,  # MHz
        freq_max: float = 750.0,  # MHz
        n_freq_channels: int = 1024,
        n_time_bins: int = 1024,
        time_resolution: float = 1.3,  # ms
    ):
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.n_freq_channels = n_freq_channels
        self.n_time_bins = n_time_bins
        self.time_resolution = time_resolution

        # Create frequency array (MHz) - highest to lowest
        self.frequencies = np.linspace(freq_max, freq_min, n_freq_channels)

        # Create time array (ms)
        self.times = np.arange(n_time_bins) * time_resolution

        # Total time window in ms
        self.total_time = n_time_bins * time_resolution

    def compute_dispersion_delay(self, dm: float, freq: float, ref_freq: Optional[float] = None) -> float:
        """
        Compute the dispersion delay relative to a reference frequency.

        The dispersion delay formula:
        Δt = 4.15 ms × DM × [(ν_ref/GHz)^-2 - (ν/GHz)^-2]

        Parameters:
            dm (float): Dispersion Measure in pc/cm³
            freq (float): Frequency in MHz
            ref_freq (float): Reference frequency in MHz (default: highest frequency)

        Returns:
            float: Dispersion delay in milliseconds
        """
        if ref_freq is None:
            ref_freq = self.freq_max

        # Convert MHz to GHz
        freq_ghz = freq / 1000.0
        ref_freq_ghz = ref_freq / 1000.0

        # Dispersion delay formula
        delay = 4.15 * dm * ((ref_freq_ghz ** -2) - (freq_ghz ** -2))

        return delay

    def gaussian_pulse(self, t: np.ndarray, arrival_time: float, width: float, amplitude: float) -> np.ndarray:
        """
        Generate a Gaussian pulse profile.

        Parameters:
            t (np.ndarray): Time array
            arrival_time (float): Pulse arrival time in ms
            width (float): Pulse width (sigma) in ms
            amplitude (float): Pulse amplitude

        Returns:
            np.ndarray: Gaussian pulse profile
        """
        return amplitude * np.exp(-((t - arrival_time) ** 2) / (2 * width ** 2))

    def generate_frb(
        self,
        dm: float,
        arrival_time: float,
        width: float = 5.0,  # ms
        amplitude: float = 10.0,
        pulse_shape: str = 'gaussian',
        spectral_index: float = 0.0,
        scattering_timescale: float = 0.0,
    ) -> Tuple[np.ndarray, dict]:
        """
        Generate an FRB with dispersion sweep across the frequency band.

        The FRB arrives at the highest frequency first and sweeps down to lower frequencies.

        Parameters:
            dm (float): Dispersion Measure in pc/cm³ (100-200 recommended)
            arrival_time (float): Arrival time at highest frequency in ms
            width (float): Intrinsic pulse width in ms (default: 5.0)
            amplitude (float): Peak amplitude of the burst (default: 10.0)
            pulse_shape (str): Pulse shape ('gaussian', 'exponential', default: 'gaussian')
            spectral_index (float): Spectral index for frequency-dependent amplitude (default: 0.0)
            scattering_timescale (float): Exponential scattering timescale in ms (default: 0.0)

        Returns:
            Tuple[np.ndarray, dict]:
                - Dynamic spectrum (freq, time) array
                - Dictionary with FRB parameters and metadata
        """
        # Initialize dynamic spectrum
        dynamic_spectrum = np.zeros((self.n_freq_channels, self.n_time_bins))

        # Store actual arrival times for each frequency channel (for mask generation)
        arrival_times_per_freq = np.zeros(self.n_freq_channels)

        # Generate dispersed burst
        for i, freq in enumerate(self.frequencies):
            # Compute dispersion delay for this frequency
            delay = self.compute_dispersion_delay(dm, freq, ref_freq=self.freq_max)

            # Actual arrival time at this frequency
            actual_arrival = arrival_time + delay
            arrival_times_per_freq[i] = actual_arrival

            # Frequency-dependent amplitude (spectral index)
            freq_amplitude = amplitude * (freq / self.freq_max) ** spectral_index

            # Generate pulse at this frequency
            if pulse_shape == 'gaussian':
                pulse = self.gaussian_pulse(self.times, actual_arrival, width, freq_amplitude)
            elif pulse_shape == 'exponential':
                # Exponential rise, Gaussian decay
                pulse = np.zeros_like(self.times)
                mask = self.times >= actual_arrival
                pulse[mask] = freq_amplitude * np.exp(-(self.times[mask] - actual_arrival) / width)
            else:
                raise ValueError(f"Unknown pulse shape: {pulse_shape}")

            # Add scattering if specified
            if scattering_timescale > 0:
                # Convolve with exponential scattering tail
                scatter_kernel = np.exp(-self.times / scattering_timescale)
                scatter_kernel = scatter_kernel / scatter_kernel.sum()  # Normalize
                pulse = np.convolve(pulse, scatter_kernel, mode='same')

            # Add to dynamic spectrum
            dynamic_spectrum[i, :] = pulse

        # Compute actual dispersion sweep time
        max_delay = self.compute_dispersion_delay(dm, self.freq_min, ref_freq=self.freq_max)

        # Metadata
        metadata = {
            'dm': dm,
            'arrival_time_top': arrival_time,  # At highest frequency
            'arrival_time_bottom': arrival_time + max_delay,  # At lowest frequency
            'width': width,
            'amplitude': amplitude,
            'pulse_shape': pulse_shape,
            'spectral_index': spectral_index,
            'scattering_timescale': scattering_timescale,
            'dispersion_sweep_time': max_delay,
            'freq_min': self.freq_min,
            'freq_max': self.freq_max,
            'arrival_times_per_freq': arrival_times_per_freq,
        }

        return dynamic_spectrum, metadata

    def generate_frb_with_snr(
        self,
        dm: float,
        snr: float,
        noise_std: float = 1.0,
        arrival_time: Optional[float] = None,
        **kwargs
    ) -> Tuple[np.ndarray, dict]:
        """
        Generate an FRB with a specified Signal-to-Noise Ratio.

        Parameters:
            dm (float): Dispersion Measure in pc/cm³
            snr (float): Desired signal-to-noise ratio
            noise_std (float): Standard deviation of background noise (default: 1.0)
            arrival_time (Optional[float]): Arrival time in ms (random if None)
            **kwargs: Additional arguments passed to generate_frb()

        Returns:
            Tuple[np.ndarray, dict]: Dynamic spectrum and metadata
        """
        # Calculate required amplitude for desired SNR
        amplitude = snr * noise_std

        # Random arrival time if not specified (allow edge cases)
        if arrival_time is None:
            # Can arrive anywhere, including before start or after end
            # This creates natural edge cases
            arrival_time = np.random.uniform(-100, self.total_time + 100)

        # Generate FRB with calculated amplitude
        return self.generate_frb(dm=dm, arrival_time=arrival_time, amplitude=amplitude, **kwargs)

    def generate_frb_mask(
        self,
        metadata: dict,
        mask_width_factor: float = 3.0,
    ) -> np.ndarray:
        """
        Generate a binary mask for the FRB signal.

        Parameters:
            metadata (dict): Metadata from generate_frb()
            mask_width_factor (float): How many pulse widths to include in mask (default: 3.0)

        Returns:
            np.ndarray: Binary mask (1 where FRB is, 0 elsewhere)
        """
        mask = np.zeros((self.n_freq_channels, self.n_time_bins), dtype=np.uint8)

        arrival_times = metadata['arrival_times_per_freq']
        width = metadata['width']
        mask_width = width * mask_width_factor

        # For each frequency channel, mask the region where FRB pulse exists
        for i, arrival_time in enumerate(arrival_times):
            # Find time bins within mask_width of arrival time
            time_mask = np.abs(self.times - arrival_time) <= mask_width
            mask[i, time_mask] = 1

        return mask


if __name__ == "__main__":
    # Example usage
    sim = FRBSimulator()

    # Generate an FRB with DM=150
    frb, meta = sim.generate_frb(dm=150, arrival_time=500, width=5.0, amplitude=15.0)

    print("FRB Generated:")
    print(f"  DM: {meta['dm']} pc/cm³")
    print(f"  Dispersion sweep time: {meta['dispersion_sweep_time']:.2f} ms")
    print(f"  Arrival at top freq: {meta['arrival_time_top']:.2f} ms")
    print(f"  Arrival at bottom freq: {meta['arrival_time_bottom']:.2f} ms")
    print(f"  Shape: {frb.shape}")

    # Generate mask
    mask = sim.generate_frb_mask(meta)
    print(f"  Mask shape: {mask.shape}")
    print(f"  Mask coverage: {mask.sum() / mask.size * 100:.2f}%")
