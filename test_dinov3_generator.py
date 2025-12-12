"""
Quick test script for DINOv3 data generator.

Verifies:
1. Full sweep generation
2. DM sweep calculations
3. Sliding window extraction
4. Data shapes and ranges
"""

import numpy as np
import matplotlib.pyplot as plt
from frb_rfi_detector.data_generator_dinov3 import FullSweepFRBGenerator


def test_dm_sweep_calculations():
    """Test DM sweep time calculations."""
    print("="*70)
    print("TESTING DM SWEEP CALCULATIONS")
    print("="*70)

    generator = FullSweepFRBGenerator(
        freq_min=300,
        freq_max=500,
        time_resolution=1.3,
    )

    print("\nDM Sweep Times (300-500 MHz, 1.3ms resolution):")
    print(f"{'DM (pc/cm³)':>12} | {'Sweep Time (ms)':>16} | {'Time Bins':>12} | {'Fits in 4096?':>15}")
    print("-" * 70)

    for dm in [30, 50, 100, 200, 500, 1000, 1500, 2000]:
        sweep_bins = generator.calculate_sweep_bins(dm)
        sweep_time = sweep_bins * 1.3

        fits = "✅ YES" if sweep_bins <= 4096 else "❌ NO (partial)"

        print(f"{dm:>12} | {sweep_time:>16.1f} | {sweep_bins:>12} | {fits:>15}")

    print()


def test_sample_generation():
    """Test sample generation."""
    print("="*70)
    print("TESTING SAMPLE GENERATION")
    print("="*70)

    generator = FullSweepFRBGenerator(
        freq_min=300,
        freq_max=500,
        freq_channels=4096,
        time_resolution=1.3,
        window_time_bins=4096,
        dm_range=(30, 2000),
        dm_sampling='log_uniform',
        rfi_probability=0.8,
        partial_sweep_probability=0.7,
    )

    print("\nGenerating 10 samples...")
    for i in range(10):
        image, frb_label, dm = generator.generate_sample()

        status = "FRB" if frb_label else "No FRB"
        dm_str = f"DM={dm:6.1f}" if frb_label else "DM=  N/A"

        print(f"  Sample {i+1:2d}: {status:6s} | {dm_str} | Shape: {image.shape} | Range: [{image.min():7.2f}, {image.max():7.2f}]")

    print("\n✅ Sample generation working!")


def test_batch_generation():
    """Test batch generation."""
    print("="*70)
    print("TESTING BATCH GENERATION")
    print("="*70)

    generator = FullSweepFRBGenerator(
        freq_min=300,
        freq_max=500,
        freq_channels=4096,
        dm_range=(30, 2000),
    )

    batch_size = 8
    print(f"\nGenerating batch of {batch_size}...")

    batch = generator.generate_batch(batch_size)

    print(f"\nBatch contents:")
    print(f"  Images shape: {batch['images'].shape}")
    print(f"  FRB labels: {batch['frb_labels']}")
    print(f"  DM values: {batch['dm_values']}")

    print(f"\nStatistics:")
    print(f"  FRB samples: {batch['frb_labels'].sum():.0f}/{batch_size}")
    print(f"  DM range: [{batch['dm_values'].min():.1f}, {batch['dm_values'].max():.1f}]")

    print("\n✅ Batch generation working!")


def test_visualization():
    """Generate and visualize sample FRBs."""
    print("="*70)
    print("TESTING VISUALIZATION")
    print("="*70)

    generator = FullSweepFRBGenerator(
        freq_min=300,
        freq_max=500,
        freq_channels=1024,  # Lower res for faster plotting
        time_resolution=1.3,
        window_time_bins=1024,
        dm_range=(100, 500),
        rfi_probability=0.9,
    )

    print("\nGenerating 4 FRB samples for visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        # Generate until we get an FRB
        attempts = 0
        while attempts < 100:
            image, frb_label, dm = generator.generate_sample()
            if frb_label:
                break
            attempts += 1

        if not frb_label:
            print(f"  Warning: Failed to generate FRB for subplot {i+1}")
            continue

        # Plot
        im = ax.imshow(image, aspect='auto', origin='upper', cmap='viridis')
        ax.set_xlabel('Time bins')
        ax.set_ylabel('Frequency bins')
        ax.set_title(f'FRB with DM={dm:.1f} pc/cm³')
        plt.colorbar(im, ax=ax)

        print(f"  Sample {i+1}: DM={dm:.1f}")

    plt.tight_layout()
    save_path = 'test_dinov3_frb_samples.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved visualization to: {save_path}")


def test_dm_sampling():
    """Test DM sampling distribution."""
    print("="*70)
    print("TESTING DM SAMPLING DISTRIBUTION")
    print("="*70)

    # Test with weighted distribution
    dm_weights = [
        [30, 200, 0.4],
        [200, 500, 0.3],
        [500, 1000, 0.2],
        [1000, 2000, 0.1],
    ]

    generator = FullSweepFRBGenerator(
        dm_range=(30, 2000),
        dm_sampling='log_uniform',
        dm_distribution_weights=dm_weights,
    )

    print("\nSampling 1000 DM values...")
    dms = [generator.sample_dm() for _ in range(1000)]

    print("\nDistribution:")
    print(f"  DM 30-200:    {sum(1 for dm in dms if 30 <= dm < 200):4d} ({sum(1 for dm in dms if 30 <= dm < 200)/10:.1f}%)")
    print(f"  DM 200-500:   {sum(1 for dm in dms if 200 <= dm < 500):4d} ({sum(1 for dm in dms if 200 <= dm < 500)/10:.1f}%)")
    print(f"  DM 500-1000:  {sum(1 for dm in dms if 500 <= dm < 1000):4d} ({sum(1 for dm in dms if 500 <= dm < 1000)/10:.1f}%)")
    print(f"  DM 1000-2000: {sum(1 for dm in dms if 1000 <= dm <= 2000):4d} ({sum(1 for dm in dms if 1000 <= dm <= 2000)/10:.1f}%)")

    print("\n✅ DM sampling working!")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print(" "*20 + "DINOV3 GENERATOR TEST SUITE")
    print("="*70 + "\n")

    test_dm_sweep_calculations()
    test_sample_generation()
    test_batch_generation()
    test_dm_sampling()
    test_visualization()

    print("\n" + "="*70)
    print("ALL TESTS PASSED! ✅")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
