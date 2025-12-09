"""
FRB-RFI Detector

SAM-based Fast Radio Burst detection in the presence of Radio Frequency Interference.
"""

__version__ = "0.1.0"

from .frb_simulator import FRBSimulator
from .rfi_simulator import RFISimulator
from .data_generator import FRBRFIDataGenerator
from .dataset import FRBRFIDataset, FRBRFIDataModule

__all__ = [
    'FRBSimulator',
    'RFISimulator',
    'FRBRFIDataGenerator',
    'FRBRFIDataset',
    'FRBRFIDataModule',
]
