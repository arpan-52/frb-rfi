"""
Test script to verify FRB-RFI simulator is working correctly.

This script tests:
1. FRB dispersion (correct direction, DM 100-200)
2. RFI generation
3. Combined data generation
4. Dataset creation with proper resizing

Run this BEFORE training to ensure everything is correct!
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add frb_rfi_detector to path
sys.path.insert(0, os.path.dirname(__file__))

from frb_rfi_detector.frb_simulator import FRBSimulator
from frb_rfi_detector.rfi_simulator import RFISimulator
from frb_rfi_detector.data_generator import FRBRFIDataGenerator
from frb_rfi_detector.dataset import FRBRFIDataset

print("="*70)
print("FRB-RFI SIMULATOR TEST")
print("="*70)

# Test 1: FRB Dispersion Direction
print("\n" + "="*70)
print("TEST 1: FRB Dispersion (300-500 MHz, DM 100-200)")
print("="*70)

sim = FRBSimulator()
print(f"Simulator settings:")
print(f"  Freq range: {sim.freq_min} - {sim.freq_max} MHz")
print(f"  Freq channels: {sim.n_freq_channels}")
print(f"  Time bins: {sim.n_time_bins}")
print(f"  Time resolution: {sim.time_resolution} ms")
print(f"  Total time window: {sim.total_time:.1f} ms")

# Test different DMs
test_dms = [100, 150, 200]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, dm in enumerate(test_dms):
    frb, meta = sim.generate_frb(
        dm=dm,
        arrival_time=500,  # ms
        width=5.0,
        amplitude=15.0
    )

    print(f"\nDM = {dm} pc/cm³:")
    print(f"  Dispersion sweep: {meta['dispersion_sweep_time']:.1f} ms")
    print(f"  Arrival at 500 MHz: {meta['arrival_time_top']:.1f} ms")
    print(f"  Arrival at 300 MHz: {meta['arrival_time_bottom']:.1f} ms")
    print(f"  Shape: {frb.shape}")

    # Verify direction: high freq arrives FIRST
    if meta['arrival_time_top'] < meta['arrival_time_bottom']:
        print(f"  ✓ CORRECT: High freq arrives before low freq")
    else:
        print(f"  ✗ ERROR: Dispersion direction is WRONG!")

    ax = axes[i]
    im = ax.imshow(frb, aspect='auto', origin='upper', cmap='hot',
                   extent=[0, sim.n_time_bins, sim.freq_min, sim.freq_max])
    ax.set_xlabel('Time (bins)')
    ax.set_ylabel('Frequency (MHz)')
    ax.set_title(f'DM = {dm} pc/cm³\nSweep = {meta["dispersion_sweep_time"]:.1f} ms')
    plt.colorbar(im, ax=ax, label='Intensity')

    # Draw line showing expected slope
    t_top = meta['arrival_time_top'] / sim.time_resolution
    t_bottom = meta['arrival_time_bottom'] / sim.time_resolution
    ax.plot([t_top, t_bottom], [sim.freq_max, sim.freq_min],
            'c--', linewidth=2, label='Expected sweep')
    ax.legend()

plt.tight_layout()
plt.savefig('test_frb_dispersion.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: test_frb_dispersion.png")

# Test 2: RFI Generation
print("\n" + "="*70)
print("TEST 2: RFI Generation")
print("="*70)

rfi_sim = RFISimulator()
rfi, rfi_table = rfi_sim.generate_rfi_waterfall(
    pers_freq_gauss=2,
    pers_time_gauss=1,
    inter_freq_gauss=1,
)

print(f"RFI shape: {rfi.shape}")
print(f"Number of RFI sources: {len(rfi_table)}")
print(f"RFI types: {rfi_table['rfi_type'].value_counts().to_dict()}")

plt.figure(figsize=(12, 8))
plt.imshow(rfi, aspect='auto', origin='upper', cmap='viridis',
           extent=[0, rfi_sim.n_time_bins, rfi_sim.freq_min, rfi_sim.freq_max])
plt.xlabel('Time (bins)')
plt.ylabel('Frequency (MHz)')
plt.title(f'RFI ({len(rfi_table)} sources)')
plt.colorbar(label='Intensity')
plt.tight_layout()
plt.savefig('test_rfi.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: test_rfi.png")

# Test 3: Combined Generation
print("\n" + "="*70)
print("TEST 3: Combined FRB + RFI + Noise")
print("="*70)

generator = FRBRFIDataGenerator()

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

scenarios = [
    ('FRB Only', {'include_frb': True, 'include_rfi': False}),
    ('RFI Only', {'include_frb': False, 'include_rfi': True}),
    ('FRB + RFI', {'include_frb': True, 'include_rfi': True}),
]

for col, (title, config) in enumerate(scenarios):
    spectrum, mask, metadata = generator.generate_sample(**config)

    print(f"\n{title}:")
    print(f"  Spectrum shape: {spectrum.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Has FRB: {metadata['has_frb']}")
    print(f"  Has RFI: {metadata['has_rfi']}")

    if metadata['has_frb']:
        print(f"  FRB DM: {metadata['frb_dm']:.1f} pc/cm³")
        print(f"  FRB SNR: {metadata['frb_snr']:.1f}")

    # Verify mask classes
    unique_classes = np.unique(mask)
    print(f"  Mask classes: {unique_classes}")

    # Plot spectrum
    ax = axes[0, col]
    im = ax.imshow(spectrum, aspect='auto', origin='upper', cmap='viridis',
                   extent=[0, spectrum.shape[1], 300, 500])
    ax.set_xlabel('Time (bins)')
    ax.set_ylabel('Frequency (MHz)')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Intensity')

    # Plot mask
    ax = axes[1, col]
    colors = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]])
    mask_rgb = colors[mask]
    ax.imshow(mask_rgb, aspect='auto', origin='upper',
              extent=[0, mask.shape[1], 300, 500])
    ax.set_xlabel('Time (bins)')
    ax.set_ylabel('Frequency (MHz)')
    ax.set_title('Ground Truth Mask')

plt.tight_layout()
plt.savefig('test_combined.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: test_combined.png")

# Test 4: Dataset with Resizing
print("\n" + "="*70)
print("TEST 4: Dataset Creation (1024×2048 → 1024×1024)")
print("="*70)

dataset = FRBRFIDataset(
    n_samples=5,
    generate_on_fly=True,
)

print(f"Dataset size: {len(dataset)}")

# Get a sample
image, mask = dataset[0]
print(f"Image shape: {image.shape}")  # Should be (3, 1024, 1024)
print(f"Mask shape: {mask.shape}")    # Should be (1024, 1024)

if image.shape == (3, 1024, 1024) and mask.shape == (1024, 1024):
    print("✓ PASS: Dataset produces correct 1024×1024 output for SAM")
else:
    print("✗ FAIL: Dataset shape is wrong!")

# Verify mask classes
unique = mask.unique().tolist()
print(f"Unique mask values: {unique}")

# Test 5: Verify DM Range
print("\n" + "="*70)
print("TEST 5: DM Range Verification")
print("="*70)

n_samples = 20
dm_values = []

for i in range(n_samples):
    _, _, meta = generator.generate_sample(include_frb=True)
    if meta['has_frb']:
        dm_values.append(meta['frb_dm'])

print(f"Generated {len(dm_values)} FRBs")
print(f"DM range: {min(dm_values):.1f} - {max(dm_values):.1f} pc/cm³")
print(f"Mean DM: {np.mean(dm_values):.1f} pc/cm³")

if min(dm_values) >= 100 and max(dm_values) <= 200:
    print("✓ PASS: All DMs within 100-200 range")
else:
    print("✗ FAIL: DMs outside expected range!")

# Summary
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)
print("Generated files:")
print("  - test_frb_dispersion.png  (check FRB curves go top-left → bottom-right)")
print("  - test_rfi.png             (check RFI patterns)")
print("  - test_combined.png        (check combined data)")
print("\nVisual checks:")
print("  1. FRB curves should go from TOP-LEFT to BOTTOM-RIGHT (\\)")
print("  2. Higher frequencies (500 MHz) should light up FIRST (left side)")
print("  3. Lower frequencies (300 MHz) should light up LAST (right side)")
print("  4. DM=100 should have less slope than DM=200")
print("\nIf all looks good, you're ready to train!")
print("="*70)
