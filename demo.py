"""
Quick demo script for FRB-RFI detector

This script demonstrates:
1. Data generation (FRB + RFI simulation)
2. Visualization
3. Model initialization
"""

import sys
import os

# Add frb_rfi_detector to path
sys.path.insert(0, os.path.dirname(__file__))

from frb_rfi_detector.frb_simulator import FRBSimulator
from frb_rfi_detector.rfi_simulator import RFISimulator
from frb_rfi_detector.data_generator import FRBRFIDataGenerator
from frb_rfi_detector.models import SAMFRBDetector

import numpy as np
import matplotlib.pyplot as plt


def demo_frb_simulation():
    """Demonstrate FRB simulation."""
    print("\n" + "="*70)
    print("DEMO 1: FRB Dispersion Simulation")
    print("="*70)

    sim = FRBSimulator()

    # Generate FRB with different DMs
    dms = [100, 150, 200]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, dm in enumerate(dms):
        frb, meta = sim.generate_frb(
            dm=dm,
            arrival_time=500,
            width=5.0,
            amplitude=15.0
        )

        ax = axes[i]
        im = ax.imshow(frb, aspect='auto', origin='lower', cmap='hot')
        ax.set_xlabel('Time (bins)')
        ax.set_ylabel('Frequency (bins)')
        ax.set_title(f'DM = {dm} pc/cm³\nSweep = {meta["dispersion_sweep_time"]:.1f} ms')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('demo_frb_dispersion.png', dpi=150)
    print(f"✓ Saved FRB dispersion demo to demo_frb_dispersion.png")


def demo_rfi_simulation():
    """Demonstrate RFI simulation."""
    print("\n" + "="*70)
    print("DEMO 2: RFI Pattern Simulation")
    print("="*70)

    rfi_sim = RFISimulator()

    # Generate different RFI types
    rfi, rfi_table = rfi_sim.generate_rfi_waterfall(
        pers_freq_gauss=2,
        pers_time_gauss=1,
        pers_freq_square=1,
        inter_freq_gauss=1,
    )

    plt.figure(figsize=(12, 8))
    plt.imshow(rfi, aspect='auto', origin='lower', cmap='viridis')
    plt.xlabel('Time (bins)')
    plt.ylabel('Frequency (bins)')
    plt.title(f'RFI Simulation ({len(rfi_table)} RFI sources)')
    plt.colorbar(label='Intensity')
    plt.tight_layout()
    plt.savefig('demo_rfi_patterns.png', dpi=150)
    print(f"✓ Saved RFI pattern demo to demo_rfi_patterns.png")

    print(f"\nRFI Table:")
    print(rfi_table.to_string())


def demo_combined_simulation():
    """Demonstrate combined FRB + RFI simulation."""
    print("\n" + "="*70)
    print("DEMO 3: Combined FRB + RFI Simulation")
    print("="*70)

    generator = FRBRFIDataGenerator()

    # Generate samples with different configurations
    configs = [
        {'include_frb': True, 'include_rfi': False},
        {'include_frb': False, 'include_rfi': True},
        {'include_frb': True, 'include_rfi': True},
    ]

    titles = ['FRB Only', 'RFI Only', 'FRB + RFI']

    fig, axes = plt.subplots(3, 2, figsize=(15, 18))

    for i, (config, title) in enumerate(zip(configs, titles)):
        spectrum, mask, metadata = generator.generate_sample(**config)

        # Color map for masks
        colors = np.array([
            [0, 0, 0],      # Background
            [1, 0, 0],      # RFI
            [0, 1, 1],      # FRB
        ])
        mask_rgb = colors[mask]

        # Plot spectrum
        ax = axes[i, 0]
        im = ax.imshow(spectrum, aspect='auto', origin='lower', cmap='viridis')
        ax.set_xlabel('Time (bins)')
        ax.set_ylabel('Frequency (bins)')
        ax.set_title(f'{title} - Dynamic Spectrum')
        plt.colorbar(im, ax=ax)

        # Plot mask
        ax = axes[i, 1]
        ax.imshow(mask_rgb, aspect='auto', origin='lower')
        ax.set_xlabel('Time (bins)')
        ax.set_ylabel('Frequency (bins)')
        ax.set_title(f'{title} - Ground Truth')

        # Print metadata
        if metadata['has_frb']:
            print(f"\n{title}:")
            print(f"  FRB DM: {metadata['frb_dm']:.1f} pc/cm³")
            print(f"  FRB SNR: {metadata['frb_snr']:.1f}")

    plt.tight_layout()
    plt.savefig('demo_combined.png', dpi=150)
    print(f"\n✓ Saved combined demo to demo_combined.png")


def demo_model():
    """Demonstrate model initialization."""
    print("\n" + "="*70)
    print("DEMO 4: Model Initialization")
    print("="*70)

    # Initialize model (without SAM checkpoint)
    model = SAMFRBDetector(model_type='vit_b', num_classes=3)

    print(f"✓ Model initialized successfully")
    print(f"  SAM available: {model.sam_available}")
    print(f"  Number of classes: {model.num_classes}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")


def demo_dataset():
    """Demonstrate dataset creation."""
    print("\n" + "="*70)
    print("DEMO 5: Dataset and DataLoader")
    print("="*70)

    from frb_rfi_detector.dataset import FRBRFIDataset
    from torch.utils.data import DataLoader

    # Create dataset
    dataset = FRBRFIDataset(
        n_samples=10,
        generate_on_fly=True,
    )

    print(f"✓ Dataset created with {len(dataset)} samples")

    # Get a sample
    image, mask = dataset[0]
    print(f"  Image shape: {image.shape}")  # (3, 1024, 1024)
    print(f"  Mask shape: {mask.shape}")    # (1024, 1024)
    print(f"  Unique mask values: {mask.unique().tolist()}")

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    print(f"✓ DataLoader created with batch_size=2")

    # Get a batch
    for batch_images, batch_masks in dataloader:
        print(f"  Batch images shape: {batch_images.shape}")
        print(f"  Batch masks shape: {batch_masks.shape}")
        break


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("FRB-RFI DETECTOR DEMO")
    print("="*70)
    print("\nThis demo will generate example data and visualizations.")
    print("All outputs will be saved to the current directory.")

    # Run demos
    demo_frb_simulation()
    demo_rfi_simulation()
    demo_combined_simulation()
    demo_model()
    demo_dataset()

    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - demo_frb_dispersion.png")
    print("  - demo_rfi_patterns.png")
    print("  - demo_combined.png")
    print("\nNext steps:")
    print("  1. Review the generated images")
    print("  2. Try training with: python frb_rfi_detector/train.py")
    print("  3. Check the README for more examples")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
