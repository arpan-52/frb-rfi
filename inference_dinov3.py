"""
Inference script for DINOv3 FRB detector.

Handles filterbank files with automatic frequency axis detection and flipping.
Processes data in overlapping 4096×4096 chunks.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import json
import yaml
import warnings

from frb_rfi_detector.models.dinov3_frb_detector import DINOv3FRBDetector


def read_filterbank_with_header(filename: str) -> Tuple[np.ndarray, Dict]:
    """
    Read filterbank file and extract header information.

    Automatically determines frequency ordering from header.

    Args:
        filename: Path to filterbank file

    Returns:
        data: (freq, time) array
        metadata: Dict with header information including frequency ordering
    """
    try:
        # Try sigpyproc
        from sigpyproc.Readers import FilReader

        fil = FilReader(filename)

        # Read data
        data = fil.readBlock(0, fil.header.nsamples)  # (time, freq)
        data = data.T  # Convert to (freq, time)

        # Get metadata
        metadata = {
            'nchans': fil.header.nchans,
            'fch1': fil.header.fch1,  # MHz - frequency of channel 0
            'foff': fil.header.foff,  # MHz - channel bandwidth (negative if decreasing)
            'tsamp': fil.header.tsamp * 1000,  # Convert to ms
            'tstart': fil.header.tstart,
            'source_name': getattr(fil.header, 'source_name', 'Unknown'),
            'nsamples': fil.header.nsamples,
        }

        # Determine frequency ordering
        # foff < 0 means frequencies decrease with channel number (high to low)
        # foff > 0 means frequencies increase with channel number (low to high)
        if metadata['foff'] < 0:
            metadata['freq_order'] = 'descending'  # Ch0=high, ChN=low
        else:
            metadata['freq_order'] = 'ascending'   # Ch0=low, ChN=high

        # Calculate frequency range
        fmin = metadata['fch1'] + metadata['foff'] * (metadata['nchans'] - 1)
        fmax = metadata['fch1']
        if fmin > fmax:
            fmin, fmax = fmax, fmin

        metadata['fmin'] = fmin
        metadata['fmax'] = fmax

        print(f"Loaded filterbank: {filename}")
        print(f"  Shape: {data.shape} (freq, time)")
        print(f"  Frequency: Ch0={metadata['fch1']:.2f} MHz, foff={metadata['foff']:.4f} MHz")
        print(f"  Frequency order: {metadata['freq_order']}")
        print(f"  Frequency range: {metadata['fmin']:.2f} - {metadata['fmax']:.2f} MHz")
        print(f"  Time samples: {metadata['nsamples']}, resolution: {metadata['tsamp']:.3f} ms")

        return data, metadata

    except ImportError:
        print("sigpyproc not installed. Trying 'your' library...")

        try:
            from your import Your

            yr = Your(filename)

            # Read data
            data = yr.get_data(0, yr.your_header.nspectra)  # (time, freq)
            data = data.T  # Convert to (freq, time)

            metadata = {
                'nchans': yr.your_header.nchans,
                'fch1': yr.your_header.fch1,
                'foff': yr.your_header.foff,
                'tsamp': yr.your_header.tsamp * 1000,
                'tstart': yr.your_header.tstart,
                'source_name': getattr(yr.your_header, 'source_name', 'Unknown'),
                'nsamples': yr.your_header.nspectra,
            }

            # Determine frequency ordering
            if metadata['foff'] < 0:
                metadata['freq_order'] = 'descending'
            else:
                metadata['freq_order'] = 'ascending'

            # Frequency range
            fmin = metadata['fch1'] + metadata['foff'] * (metadata['nchans'] - 1)
            fmax = metadata['fch1']
            if fmin > fmax:
                fmin, fmax = fmax, fmin

            metadata['fmin'] = fmin
            metadata['fmax'] = fmax

            print(f"Loaded filterbank: {filename}")
            print(f"  Shape: {data.shape} (freq, time)")
            print(f"  Frequency order: {metadata['freq_order']}")
            print(f"  Frequency range: {metadata['fmin']:.2f} - {metadata['fmax']:.2f} MHz")

            return data, metadata

        except ImportError:
            raise ImportError("Please install sigpyproc or your: pip install sigpyproc-python OR pip install your")


def extract_and_normalize_frequency_range(
    data: np.ndarray,
    metadata: Dict,
    target_fmin: float,
    target_fmax: float,
    target_channels: int = 4096,
) -> np.ndarray:
    """
    Extract frequency range and resample to target channels.

    Automatically handles frequency axis orientation based on header.

    Args:
        data: Input data (freq, time)
        metadata: Filterbank metadata with frequency ordering info
        target_fmin: Target minimum frequency (MHz)
        target_fmax: Target maximum frequency (MHz)
        target_channels: Target number of frequency channels

    Returns:
        Extracted and resampled data (target_channels, time)
    """
    fch1 = metadata['fch1']
    foff = metadata['foff']
    nchans = metadata['nchans']

    # Create frequency array
    freqs = fch1 + np.arange(nchans) * foff

    print(f"\nExtracting {target_fmin}-{target_fmax} MHz:")
    print(f"  Original: {freqs.min():.2f} - {freqs.max():.2f} MHz ({nchans} channels)")

    # Find channels in target range
    mask = (freqs >= target_fmin) & (freqs <= target_fmax)
    indices = np.where(mask)[0]

    if len(indices) == 0:
        raise ValueError(f"No channels found in range {target_fmin}-{target_fmax} MHz!")

    # Extract
    extracted = data[indices, :]
    extracted_freqs = freqs[indices]

    print(f"  Extracted: {len(indices)} channels")
    print(f"  Actual range: {extracted_freqs.min():.2f} - {extracted_freqs.max():.2f} MHz")

    # Check if we need to flip to ensure high->low ordering (model expects this)
    first_freq = extracted_freqs[0]
    last_freq = extracted_freqs[-1]

    if first_freq < last_freq:
        # Currently low->high, need to flip to high->low
        print(f"  Flipping frequency axis (low→high to high→low)")
        extracted = np.flip(extracted, axis=0)
        extracted_freqs = np.flip(extracted_freqs)
    else:
        print(f"  Frequency order correct (high→low)")

    # Resample to target channels
    if len(indices) != target_channels:
        print(f"  Resampling to {target_channels} channels...")
        extracted = resample_frequency(extracted, target_channels)

    return extracted


def resample_frequency(data: np.ndarray, target_channels: int) -> np.ndarray:
    """
    Resample frequency axis to target number of channels.

    Args:
        data: Input data (freq, time)
        target_channels: Target number of channels

    Returns:
        Resampled data (target_channels, time)
    """
    n_freq, n_time = data.shape

    if n_freq == target_channels:
        return data

    # Use scipy for high-quality resampling
    try:
        from scipy.signal import resample

        resampled = resample(data, target_channels, axis=0)
        return resampled.astype(np.float32)

    except ImportError:
        # Fallback: simple averaging or repetition
        if n_freq > target_channels:
            # Downsample by averaging
            factor = n_freq // target_channels
            remainder = n_freq % target_channels

            if remainder == 0:
                # Perfect division
                downsampled = data[:target_channels * factor, :].reshape(
                    target_channels, factor, n_time
                ).mean(axis=1)
            else:
                # Approximate
                warnings.warn(f"Imperfect downsampling: {n_freq} -> {target_channels}")
                downsampled = data[:target_channels, :]

            return downsampled.astype(np.float32)
        else:
            # Upsample by repetition
            factor = target_channels // n_freq
            upsampled = np.repeat(data, factor, axis=0)[:target_channels, :]
            return upsampled.astype(np.float32)


def extract_chunks(
    data: np.ndarray,
    chunk_size: int = 4096,
    overlap: int = 2048
) -> List[Tuple[np.ndarray, int]]:
    """
    Extract overlapping chunks from filterbank data.

    Args:
        data: Input data (freq, time)
        chunk_size: Chunk size in time bins
        overlap: Overlap between chunks

    Returns:
        List of (chunk, start_index) tuples
    """
    n_freq, n_time = data.shape
    stride = chunk_size - overlap

    chunks = []

    for start in range(0, n_time - chunk_size + 1, stride):
        chunk = data[:, start:start + chunk_size]

        if chunk.shape[1] < chunk_size:
            # Pad if needed
            pad_width = chunk_size - chunk.shape[1]
            chunk = np.pad(chunk, ((0, 0), (0, pad_width)), mode='edge')

        chunks.append((chunk, start))

    # Handle remainder
    if n_time % stride != 0 and n_time >= chunk_size:
        start = n_time - chunk_size
        chunk = data[:, start:]
        if chunk.shape[1] < chunk_size:
            pad_width = chunk_size - chunk.shape[1]
            chunk = np.pad(chunk, ((0, 0), (0, pad_width)), mode='edge')
        chunks.append((chunk, start))

    print(f"\nExtracted {len(chunks)} chunks:")
    print(f"  Chunk size: {chunk_size} time bins")
    print(f"  Overlap: {overlap} bins ({100*overlap/chunk_size:.0f}%)")
    print(f"  Total time coverage: {n_time * data.shape[0]} bins")

    return chunks


def normalize_chunk(chunk: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    """Normalize chunk to [0, 1]."""
    vmin = np.percentile(chunk, 100 - percentile)
    vmax = np.percentile(chunk, percentile)

    normalized = (chunk - vmin) / (vmax - vmin + 1e-8)
    normalized = np.clip(normalized, 0, 1)

    return normalized.astype(np.float32)


def run_inference_on_chunk(
    model: torch.nn.Module,
    chunk: np.ndarray,
    device: str = 'cuda'
) -> Tuple[bool, float, float]:
    """
    Run inference on a single chunk.

    Args:
        model: Trained model
        chunk: Input chunk (4096, 4096)
        device: Device

    Returns:
        frb_detected: True/False
        frb_probability: Detection probability
        dm_estimate: DM estimate (if detected)
    """
    model.eval()

    # Normalize
    chunk_norm = normalize_chunk(chunk)

    # Convert to RGB
    image = np.stack([chunk_norm, chunk_norm, chunk_norm], axis=0)
    image_tensor = torch.from_numpy(image).float().unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        frb_detected, frb_prob, dm_estimate = model.predict(image_tensor)

    return (
        frb_detected[0].item(),
        frb_prob[0].item(),
        dm_estimate[0].item()
    )


def plot_detection(
    chunk: np.ndarray,
    chunk_start: int,
    frb_prob: float,
    dm_estimate: float,
    metadata: Dict,
    save_path: str,
    freq_range: Tuple[float, float] = (300, 500),
):
    """
    Plot detection.

    Args:
        chunk: Data chunk (freq, time)
        chunk_start: Start index of chunk
        frb_prob: FRB detection probability
        dm_estimate: Estimated DM
        metadata: Filterbank metadata
        save_path: Path to save plot
        freq_range: (fmin, fmax) in MHz
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    fmax, fmin = freq_range  # High to low
    tsamp = metadata['tsamp']
    tstart = chunk_start * tsamp
    tend = tstart + chunk.shape[1] * tsamp

    extent = [tstart, tend, fmin, fmax]

    # Plot 1: Filterbank data
    ax = axes[0]
    im = ax.imshow(chunk, aspect='auto', origin='upper', cmap='viridis', extent=extent)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (MHz)')
    ax.set_title('Filterbank Data')
    plt.colorbar(im, ax=ax, label='Intensity')

    # Plot 2: Time series
    ax = axes[1]
    time_series = chunk.mean(axis=0)
    times = np.linspace(tstart, tend, len(time_series))
    ax.plot(times, time_series, 'c-', linewidth=2)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Mean Intensity')
    ax.set_title(f'FRB Detection (P={frb_prob:.3f}, DM={dm_estimate:.1f} pc/cm³)')
    ax.grid(True, alpha=0.3)

    # Mark peak
    peak_idx = np.argmax(time_series)
    ax.axvline(times[peak_idx], color='red', linestyle='--', alpha=0.5, label='Peak')
    ax.legend()

    fig.suptitle(f"{metadata.get('source_name', 'Unknown')} - t={tstart:.1f} ms", fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_filterbank(
    filterbank_path: str,
    model_path: str,
    config_path: str,
    output_dir: str,
    detection_threshold: float = 0.5,
    device: str = 'cuda',
):
    """
    Process filterbank file and detect FRBs.

    Args:
        filterbank_path: Path to filterbank
        model_path: Path to trained model
        config_path: Path to config file
        output_dir: Output directory
        detection_threshold: Detection threshold
        device: Device
    """
    print("="*70)
    print("DINOV3 FRB DETECTION")
    print("="*70)

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    inf_config = config['inference']

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading model from {model_path}...")

    # Get DM range
    dm_range = (
        config['data']['dispersion']['dm_min'],
        config['data']['dispersion']['dm_max']
    )

    model = DINOv3FRBDetector(
        model_type=config['model']['backbone'],
        freeze_backbone=True,  # Always freeze for inference
        hidden_dims=config['model']['head']['hidden_dims'],
        dropout=0.0,  # No dropout for inference
        dm_range=dm_range,
    )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")

    # Read filterbank
    print(f"\nReading filterbank: {filterbank_path}")
    data, metadata = read_filterbank_with_header(filterbank_path)

    # Extract and normalize frequency range
    data_processed = extract_and_normalize_frequency_range(
        data,
        metadata,
        target_fmin=inf_config['target_freq_min'],
        target_fmax=inf_config['target_freq_max'],
        target_channels=inf_config['target_freq_channels'],
    )

    # Extract chunks
    chunks = extract_chunks(
        data_processed,
        chunk_size=inf_config['chunk_size'],
        overlap=inf_config['overlap'],
    )

    # Process chunks
    print(f"\nProcessing {len(chunks)} chunks...")
    detections = []

    for i, (chunk, start_idx) in enumerate(chunks):
        frb_detected, frb_prob, dm_estimate = run_inference_on_chunk(
            model, chunk, device
        )

        if frb_detected and frb_prob > detection_threshold:
            # Check DM range
            if inf_config['min_dm'] <= dm_estimate <= inf_config['max_dm']:
                print(f"  Chunk {i+1}/{len(chunks)}: FRB DETECTED! (P={frb_prob:.3f}, DM={dm_estimate:.1f})")

                # Save plot
                plot_path = output_path / f"detection_chunk_{i:04d}.png"
                plot_detection(
                    chunk, start_idx, frb_prob, dm_estimate, metadata,
                    str(plot_path),
                    freq_range=(inf_config['target_freq_max'], inf_config['target_freq_min'])
                )

                detections.append({
                    'chunk_index': i,
                    'start_time_ms': start_idx * metadata['tsamp'],
                    'probability': float(frb_prob),
                    'dm_estimate': float(dm_estimate),
                    'plot_path': str(plot_path),
                })
            else:
                print(f"  Chunk {i+1}/{len(chunks)}: Candidate rejected (DM={dm_estimate:.1f} out of range)")
        else:
            if i % 10 == 0:
                print(f"  Chunk {i+1}/{len(chunks)}: No FRB (P={frb_prob:.3f})")

    # Summary
    print("\n" + "="*70)
    print("DETECTION SUMMARY")
    print("="*70)
    print(f"Total chunks: {len(chunks)}")
    print(f"FRB detections: {len(detections)}")

    if detections:
        print("\nDetected FRBs:")
        for det in detections:
            print(f"  t={det['start_time_ms']:.1f} ms, DM={det['dm_estimate']:.1f}, P={det['probability']:.3f}")

    # Save JSON
    summary_path = output_path / 'detections.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'filterbank': filterbank_path,
            'model': model_path,
            'total_chunks': len(chunks),
            'num_detections': len(detections),
            'detections': detections,
        }, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='DINOv3 FRB Detection Inference')
    parser.add_argument('--filterbank', type=str, required=True,
                        help='Path to filterbank file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='config/dinov3_frb_config.yaml',
                        help='Path to config file')
    parser.add_argument('--output', type=str, default='detections_dinov3',
                        help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Detection threshold')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device')

    args = parser.parse_args()

    process_filterbank(
        filterbank_path=args.filterbank,
        model_path=args.model,
        config_path=args.config,
        output_dir=args.output,
        detection_threshold=args.threshold,
        device=args.device,
    )


if __name__ == "__main__":
    main()
