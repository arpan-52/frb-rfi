"""
RFI (Radio Frequency Interference) Simulator

This module simulates various types of RFI patterns commonly observed in radio astronomy.
Based on the SAM-RFI implementation by preshanth.

Supports:
- Persistent RFI (frequency and time domain)
- Intermittent RFI (periodic signals)
- Gaussian and square wave patterns
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
import warnings

# Suppress pandas FutureWarning about DataFrame concatenation
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')


class RFISimulator:
    """
    Simulates realistic RFI patterns for radio astronomy observations.

    Parameters:
        freq_min (float): Minimum frequency in MHz (default: 550 MHz)
        freq_max (float): Maximum frequency in MHz (default: 750 MHz)
        n_freq_channels (int): Number of frequency channels (default: 1024)
        n_time_bins (int): Number of time bins (default: 1024)
        time_resolution (float): Time resolution in milliseconds (default: 1.3 ms)
    """

    def __init__(
        self,
        freq_min: float = 300.0,
        freq_max: float = 500.0,
        n_freq_channels: int = 1024,
        n_time_bins: int = 2048,
        time_resolution: float = 1.3,
    ):
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.n_freq_channels = n_freq_channels
        self.n_time_bins = n_time_bins
        self.time_resolution = time_resolution

        # Create frequency and time arrays
        self.frequencies = np.linspace(freq_max, freq_min, n_freq_channels)
        self.times = np.arange(n_time_bins) * time_resolution
        self.total_time = n_time_bins * time_resolution

        # Blank spectrograph for operations
        self.blank_spectrograph = np.zeros((n_freq_channels, n_time_bins))

        # RFI table to store all RFI parameters
        self.rfi_table = pd.DataFrame(
            columns=[
                'rfi_type', 'amplitude', 'center_freq', 'bandwidth',
                'center_time', 'timewidth', 'duty_cycle', 'time_period', 'time_offset'
            ]
        )

    def gaussian_function(self, x: np.ndarray, amplitude: float, center: float, width: float) -> np.ndarray:
        """
        Generate a Gaussian function.

        Parameters:
            x (np.ndarray): Input array (frequency or time)
            amplitude (float): Peak amplitude
            center (float): Center position
            width (float): Width (sigma)

        Returns:
            np.ndarray: Gaussian function values
        """
        return amplitude * np.exp(-((x - center) ** 2) / (2 * (width ** 2)))

    def square_function(self, x: np.ndarray, amplitude: float, center: float, width: float) -> np.ndarray:
        """
        Generate a square wave function.

        Parameters:
            x (np.ndarray): Input array (frequency or time)
            amplitude (float): Amplitude
            center (float): Center position
            width (float): Width

        Returns:
            np.ndarray: Square wave values
        """
        return amplitude * np.where(np.abs(x - center) <= width / 2, 1, 0)

    def add_gaussian_rfi_spectrograph(
        self,
        amplitude: float,
        center: float,
        width: float,
        rfi_axis: str = 'FREQ',
        table: bool = False
    ) -> np.ndarray:
        """
        Add Gaussian RFI to the spectrograph.

        Parameters:
            amplitude (float): RFI amplitude
            center (float): Center frequency (MHz) or time (ms)
            width (float): Bandwidth (MHz) or time width (ms)
            rfi_axis (str): 'FREQ' for frequency-persistent or 'TIME' for time-persistent
            table (bool): Whether to record in RFI table

        Returns:
            np.ndarray: Spectrograph with added RFI
        """
        if rfi_axis == 'FREQ':
            # Frequency-persistent RFI (vertical line in waterfall)
            rfi_signal = self.gaussian_function(self.frequencies, amplitude, center, width)
            rfi_spec = np.outer(rfi_signal, np.ones(self.n_time_bins))
            rfi_type = 'gauss_persistent_freq'

            if table:
                new_row = pd.DataFrame({
                    'rfi_type': [rfi_type],
                    'amplitude': [amplitude],
                    'center_freq': [center],
                    'bandwidth': [width],
                    'center_time': [np.nan],
                    'timewidth': [np.nan],
                    'duty_cycle': [np.nan],
                    'time_period': [np.nan],
                    'time_offset': [np.nan]
                })
                self.rfi_table = pd.concat([self.rfi_table, new_row], ignore_index=True)

        elif rfi_axis == 'TIME':
            # Time-persistent RFI (horizontal line in waterfall)
            rfi_signal = self.gaussian_function(self.times, amplitude, center, width)
            rfi_spec = np.outer(np.ones(self.n_freq_channels), rfi_signal)
            rfi_type = 'gauss_persistent_time'

            if table:
                new_row = pd.DataFrame({
                    'rfi_type': [rfi_type],
                    'amplitude': [amplitude],
                    'center_freq': [np.nan],
                    'bandwidth': [np.nan],
                    'center_time': [center],
                    'timewidth': [width],
                    'duty_cycle': [np.nan],
                    'time_period': [np.nan],
                    'time_offset': [np.nan]
                })
                self.rfi_table = pd.concat([self.rfi_table, new_row], ignore_index=True)
        else:
            raise ValueError("Invalid rfi_axis. Use 'FREQ' or 'TIME'.")

        return rfi_spec

    def add_square_rfi_spectrograph(
        self,
        amplitude: float,
        center: float,
        width: float,
        rfi_axis: str = 'FREQ',
        table: bool = False
    ) -> np.ndarray:
        """
        Add square wave RFI to the spectrograph.

        Parameters:
            amplitude (float): RFI amplitude
            center (float): Center frequency (MHz) or time (ms)
            width (float): Bandwidth (MHz) or time width (ms)
            rfi_axis (str): 'FREQ' for frequency-persistent or 'TIME' for time-persistent
            table (bool): Whether to record in RFI table

        Returns:
            np.ndarray: Spectrograph with added RFI
        """
        if rfi_axis == 'FREQ':
            rfi_signal = self.square_function(self.frequencies, amplitude, center, width)
            rfi_spec = np.outer(rfi_signal, np.ones(self.n_time_bins))
            rfi_type = 'square_persistent_freq'

            if table:
                new_row = pd.DataFrame({
                    'rfi_type': [rfi_type],
                    'amplitude': [amplitude],
                    'center_freq': [center],
                    'bandwidth': [width],
                    'center_time': [np.nan],
                    'timewidth': [np.nan],
                    'duty_cycle': [np.nan],
                    'time_period': [np.nan],
                    'time_offset': [np.nan]
                })
                self.rfi_table = pd.concat([self.rfi_table, new_row], ignore_index=True)

        elif rfi_axis == 'TIME':
            rfi_signal = self.square_function(self.times, amplitude, center, width)
            rfi_spec = np.outer(np.ones(self.n_freq_channels), rfi_signal)
            rfi_type = 'square_persistent_time'

            if table:
                new_row = pd.DataFrame({
                    'rfi_type': [rfi_type],
                    'amplitude': [amplitude],
                    'center_freq': [np.nan],
                    'bandwidth': [np.nan],
                    'center_time': [center],
                    'timewidth': [width],
                    'duty_cycle': [np.nan],
                    'time_period': [np.nan],
                    'time_offset': [np.nan]
                })
                self.rfi_table = pd.concat([self.rfi_table, new_row], ignore_index=True)
        else:
            raise ValueError("Invalid rfi_axis. Use 'FREQ' or 'TIME'.")

        return rfi_spec

    def intermittent_rfi(
        self,
        amplitude: float,
        center_freq: float,
        bandwidth: float,
        time_period: int,
        duty_cycle: float,
        time_offset: int = 0,
        func_type: str = 'GAUSS',
        table: bool = False
    ) -> np.ndarray:
        """
        Generate intermittent (periodic) RFI.

        Parameters:
            amplitude (float): RFI amplitude
            center_freq (float): Center frequency in MHz
            bandwidth (float): Bandwidth in MHz
            time_period (int): Period in time bins
            duty_cycle (float): Fraction of period when RFI is active (0-1)
            time_offset (int): Initial offset in time bins
            func_type (str): 'GAUSS' or 'SQUARE'
            table (bool): Whether to record in RFI table

        Returns:
            np.ndarray: Spectrograph with intermittent RFI
        """
        modified_spectrograph = self.blank_spectrograph.copy()

        # Create time mask for intermittent RFI
        time_mask = np.zeros(self.n_time_bins)
        period_indices = np.arange(time_offset, self.n_time_bins, time_period)

        for start_idx in period_indices:
            end_idx = min(start_idx + int(time_period * duty_cycle), self.n_time_bins)
            time_mask[int(start_idx):int(end_idx)] = 1

        # Generate RFI signal
        if func_type == 'GAUSS':
            rfi_signal = self.gaussian_function(self.frequencies, amplitude, center_freq, bandwidth)
            rfi_type = 'intermittent_gauss'
        elif func_type == 'SQUARE':
            rfi_signal = self.square_function(self.frequencies, amplitude, center_freq, bandwidth)
            rfi_type = 'intermittent_square'
        else:
            raise ValueError("Invalid func_type. Use 'GAUSS' or 'SQUARE'.")

        # Apply intermittent pattern
        for t in range(self.n_time_bins):
            if time_mask[t] == 1:
                modified_spectrograph[:, t] += rfi_signal

        # Update RFI table
        if table:
            new_row = pd.DataFrame({
                'rfi_type': [rfi_type],
                'amplitude': [amplitude],
                'center_freq': [center_freq],
                'bandwidth': [bandwidth],
                'center_time': [np.nan],
                'timewidth': [np.nan],
                'duty_cycle': [duty_cycle],
                'time_period': [time_period],
                'time_offset': [time_offset]
            })
            self.rfi_table = pd.concat([self.rfi_table, new_row], ignore_index=True)

        return modified_spectrograph

    def generate_rfi_waterfall(
        self,
        pers_freq_gauss: int = 2,
        pers_time_gauss: int = 1,
        pers_freq_square: int = 1,
        pers_time_square: int = 1,
        inter_freq_gauss: int = 2,
        inter_freq_square: int = 1,
        mean_rfi: float = 1.0,
        std_rfi: float = 3.0,
        edge_buffer: float = 50.0,  # MHz or ms
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Generate a complete RFI waterfall with multiple random RFI signals.

        Parameters:
            pers_freq_gauss (int): Number of persistent Gaussian RFI in frequency
            pers_time_gauss (int): Number of persistent Gaussian RFI in time
            pers_freq_square (int): Number of persistent square RFI in frequency
            pers_time_square (int): Number of persistent square RFI in time
            inter_freq_gauss (int): Number of intermittent Gaussian RFI
            inter_freq_square (int): Number of intermittent square RFI
            mean_rfi (float): Mean amplitude for RFI
            std_rfi (float): Standard deviation of RFI amplitude
            edge_buffer (float): Buffer from edges in MHz or ms

        Returns:
            Tuple[np.ndarray, pd.DataFrame]: RFI spectrograph and RFI table
        """
        # Reset RFI table
        self.rfi_table = pd.DataFrame(
            columns=[
                'rfi_type', 'amplitude', 'center_freq', 'bandwidth',
                'center_time', 'timewidth', 'duty_cycle', 'time_period', 'time_offset'
            ]
        )

        total_rfi = self.blank_spectrograph.copy()

        # Persistent Gaussian RFI in frequency
        for _ in range(pers_freq_gauss):
            rfi = self.add_gaussian_rfi_spectrograph(
                amplitude=np.abs(np.random.normal(mean_rfi, std_rfi)),
                center=np.random.uniform(self.freq_min + edge_buffer, self.freq_max - edge_buffer),
                width=np.abs(np.random.normal(10, 3)),
                rfi_axis='FREQ',
                table=True
            )
            total_rfi += rfi

        # Persistent Gaussian RFI in time
        for _ in range(pers_time_gauss):
            rfi = self.add_gaussian_rfi_spectrograph(
                amplitude=np.abs(np.random.normal(mean_rfi, std_rfi)),
                center=np.random.uniform(edge_buffer, self.total_time - edge_buffer),
                width=np.abs(np.random.normal(50, 10)),
                rfi_axis='TIME',
                table=True
            )
            total_rfi += rfi

        # Persistent Square RFI in frequency
        for _ in range(pers_freq_square):
            rfi = self.add_square_rfi_spectrograph(
                amplitude=np.abs(np.random.uniform(mean_rfi, std_rfi)),
                center=np.random.uniform(self.freq_min + edge_buffer, self.freq_max - edge_buffer),
                width=np.random.uniform(5, 20),
                rfi_axis='FREQ',
                table=True
            )
            total_rfi += rfi

        # Persistent Square RFI in time
        for _ in range(pers_time_square):
            rfi = self.add_square_rfi_spectrograph(
                amplitude=np.abs(np.random.uniform(mean_rfi, std_rfi)),
                center=np.random.uniform(edge_buffer, self.total_time - edge_buffer),
                width=np.random.uniform(30, 100),
                rfi_axis='TIME',
                table=True
            )
            total_rfi += rfi

        # Intermittent Gaussian RFI
        for _ in range(inter_freq_gauss):
            rfi = self.intermittent_rfi(
                amplitude=np.abs(np.random.normal(mean_rfi, std_rfi)),
                center_freq=np.random.uniform(self.freq_min + edge_buffer, self.freq_max - edge_buffer),
                bandwidth=np.abs(np.random.normal(15, 5)),
                time_period=np.random.randint(50, 300),
                duty_cycle=np.random.uniform(0.2, 0.8),
                time_offset=np.random.randint(0, 100),
                func_type='GAUSS',
                table=True
            )
            total_rfi += rfi

        # Intermittent Square RFI
        for _ in range(inter_freq_square):
            rfi = self.intermittent_rfi(
                amplitude=np.abs(np.random.normal(mean_rfi, std_rfi)),
                center_freq=np.random.uniform(self.freq_min + edge_buffer, self.freq_max - edge_buffer),
                bandwidth=np.random.uniform(5, 25),
                time_period=np.random.randint(50, 300),
                duty_cycle=np.random.uniform(0.2, 0.8),
                time_offset=np.random.randint(0, 100),
                func_type='SQUARE',
                table=True
            )
            total_rfi += rfi

        return total_rfi, self.rfi_table.copy()

    def generate_rfi_mask(self, rfi_spectrograph: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """
        Generate a binary mask for RFI locations.

        Parameters:
            rfi_spectrograph (np.ndarray): RFI spectrograph
            threshold (float): Threshold for masking (default: 0.1)

        Returns:
            np.ndarray: Binary mask (1 where RFI is, 0 elsewhere)
        """
        mask = (rfi_spectrograph > threshold).astype(np.uint8)
        return mask


if __name__ == "__main__":
    # Example usage
    rfi_sim = RFISimulator()

    # Generate RFI waterfall
    rfi, rfi_table = rfi_sim.generate_rfi_waterfall(
        pers_freq_gauss=3,
        pers_time_gauss=2,
        inter_freq_gauss=2
    )

    print("RFI Generated:")
    print(f"  Shape: {rfi.shape}")
    print(f"  Number of RFI types: {len(rfi_table)}")
    print(f"  Max amplitude: {rfi.max():.2f}")
    print(f"\nRFI Table:")
    print(rfi_table)

    # Generate mask
    mask = rfi_sim.generate_rfi_mask(rfi, threshold=0.1)
    print(f"\nRFI Mask coverage: {mask.sum() / mask.size * 100:.2f}%")
