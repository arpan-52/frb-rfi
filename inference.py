"""
Inference script for detecting FRBs in real filterbank data.

This script loads a trained model and processes filterbank files to detect FRBs
and estimate their dispersion measures.

Usage:
    python inference.py --filterbank my_data.fil --model best_model.pt --output results/
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import warnings

from frb_rfi_detector.models import SAMFRBDetector


def read_filterbank(filename: str) -> Tuple[np.ndarray, Dict]:
    """
    Read a filterbank file.

    You can use sigpyproc, your, or implement custom reader.
    This is a template - adjust based on your filterbank reader.

    Parameters:
        filename (str): Path to filterbank file

    Returns:
        Tuple[np.ndarray, Dict]: (data array (freq, time), metadata dict)
    """
    try:
        # Try using sigpyproc
        from sigpyproc.Readers import FilReader

        fil = FilReader(filename)

        # Read data
        data = fil.readBlock(0, fil.header.nsamples)  # (time, freq)
        data = data.T  # Convert to (freq, time)

        # Get metadata
        metadata = {
            'nchans': fil.header.nchans,
            'fch1': fil.header.fch1,  # MHz
            'foff': fil.header.foff,  # MHz
            'tsamp': fil.header.tsamp * 1000,  # Convert to ms
            'tstart': fil.header.tstart,
            'source_name': fil.header.source_name,
            'nsamples': fil.header.nsamples,
        }

        print(f"Loaded filterbank: {filename}")
        print(f"  Shape: {data.shape} (freq, time)")
        print(f"  Frequency: {metadata['fch1']:.2f} MHz (top), bandwidth: {metadata['foff'] * metadata['nchans']:.2f} MHz")
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
                'tsamp': yr.your_header.tsamp * 1000,  # Convert to ms
                'tstart': yr.your_header.tstart,
                'source_name': yr.your_header.source_name,
                'nsamples': yr.your_header.nspectra,
            }

            print(f"Loaded filterbank: {filename}")
            print(f"  Shape: {data.shape} (freq, time)")

            return data, metadata

        except ImportError:
            raise ImportError("Please install sigpyproc or your: pip install sigpyproc-python OR pip install your")


def extract_frequency_range(data: np.ndarray, metadata: Dict,
                           target_fmin: float = 300.0, target_fmax: float = 500.0) -> np.ndarray:
    """
    Extract a specific frequency range from filterbank data.

    Parameters:
        data (np.ndarray): Input data (freq, time)
        metadata (Dict): Filterbank metadata with fch1, foff, nchans
        target_fmin (float): Target minimum frequency in MHz
        target_fmax (float): Target maximum frequency in MHz

    Returns:
        np.ndarray: Extracted frequency range
    """
    fch1 = metadata['fch1']  # Top frequency (0th channel)
    foff = metadata['foff']  # Channel width (negative if decreasing)
    nchans = metadata['nchans']

    # Create frequency array for all channels
    freqs = fch1 + np.arange(nchans) * foff

    print(f"Original filterbank:")
    print(f"  Frequency range: {freqs.min():.2f} - {freqs.max():.2f} MHz")
    print(f"  Channel 0: {fch1:.2f} MHz")
    print(f"  Channel width: {foff:.4f} MHz")

    # Find channels within target range
    # We want freqs between target_fmin and target_fmax
    if foff < 0:  # Frequencies decreasing
        mask = (freqs >= target_fmin) & (freqs <= target_fmax)
    else:  # Frequencies increasing
        mask = (freqs >= target_fmin) & (freqs <= target_fmax)

    indices = np.where(mask)[0]

    if len(indices) == 0:
        raise ValueError(f"No channels found in range {target_fmin}-{target_fmax} MHz!")

    # Extract channels
    extracted = data[indices, :]

    print(f"Extracted frequency range {target_fmin}-{target_fmax} MHz:")
    print(f"  Channels: {indices[0]} to {indices[-1]} ({len(indices)} channels)")
    print(f"  Actual range: {freqs[indices[0]]:.2f} - {freqs[indices[-1]]:.2f} MHz")

    # Check if we need to flip (model expects high freq first)
    # If fch1 is at top and foff is negative, it's already high->low (correct)
    # If fch1 is at bottom and foff is positive, we need to flip
    first_freq = freqs[indices[0]]
    last_freq = freqs[indices[-1]]

    if first_freq < last_freq:
        print(f"  Flipping frequency axis (low->high to high->low)")
        extracted = np.flip(extracted, axis=0)
    else:
        print(f"  Frequency order correct (high->low)")

    return extracted


def downsample_frequency(data: np.ndarray, target_channels: int = 1024) -> np.ndarray:
    """
    Downsample frequency channels to target number.

    Parameters:
        data (np.ndarray): Input data (freq, time)
        target_channels (int): Target number of frequency channels

    Returns:
        np.ndarray: Downsampled data (target_channels, time)
    """
    n_freq, n_time = data.shape

    if n_freq == target_channels:
        return data

    # Simple averaging downsampling
    factor = n_freq // target_channels

    if factor < 1:
        # Need to upsample - just repeat
        factor = target_channels // n_freq
        downsampled = np.repeat(data, factor, axis=0)[:target_channels, :]
        print(f"Upsampled from {n_freq} to {target_channels} channels (factor: {factor})")
    else:
        # Downsample by averaging
        downsampled = data[:target_channels * factor, :].reshape(target_channels, factor, n_time).mean(axis=1)
        print(f"Downsampled from {n_freq} to {target_channels} channels (factor: {factor})")

    return downsampled


def normalize_chunk(chunk: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    """
    Normalize a data chunk to [0, 1] range.

    Parameters:
        chunk (np.ndarray): Input chunk
        percentile (float): Percentile for clipping

    Returns:
        np.ndarray: Normalized chunk
    """
    vmin = np.percentile(chunk, 100 - percentile)
    vmax = np.percentile(chunk, percentile)

    normalized = (chunk - vmin) / (vmax - vmin + 1e-8)
    normalized = np.clip(normalized, 0, 1)

    return normalized


def extract_chunks(data: np.ndarray, chunk_size: int = 1024, overlap: int = 256) -> List[Tuple[np.ndarray, int]]:
    """
    Extract overlapping chunks from filterbank data.

    Parameters:
        data (np.ndarray): Input data (freq, time)
        chunk_size (int): Size of each chunk in time bins
        overlap (int): Overlap between chunks

    Returns:
        List[Tuple[np.ndarray, int]]: List of (chunk, start_index) tuples
    """
    n_freq, n_time = data.shape
    stride = chunk_size - overlap

    chunks = []

    for start in range(0, n_time - chunk_size + 1, stride):
        chunk = data[:, start:start + chunk_size]

        # Pad if needed to exactly chunk_size
        if chunk.shape[1] < chunk_size:
            pad_width = chunk_size - chunk.shape[1]
            chunk = np.pad(chunk, ((0, 0), (0, pad_width)), mode='edge')

        chunks.append((chunk, start))

    print(f"Extracted {len(chunks)} chunks with size {chunk_size} and overlap {overlap}")

    return chunks


def run_inference(model: torch.nn.Module, chunk: np.ndarray, device: str = 'cuda') -> Tuple[np.ndarray, np.ndarray]:
    """
    Run model inference on a single chunk.

    Parameters:
        model: Trained model
        chunk (np.ndarray): Input chunk (1024, 1024)
        device (str): Device to run on

    Returns:
        Tuple[np.ndarray, np.ndarray]: (prediction, probabilities)
    """
    model.eval()

    # Normalize
    chunk_norm = normalize_chunk(chunk)

    # Convert to RGB tensor
    image = np.stack([chunk_norm, chunk_norm, chunk_norm], axis=0)
    image_tensor = torch.from_numpy(image).float().unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        prediction, probabilities = model.predict(image_tensor)

    # Convert to numpy
    prediction = prediction[0].cpu().numpy()
    probabilities = probabilities[0].cpu().numpy()

    return prediction, probabilities


def estimate_dm_from_detection(chunk: np.ndarray, mask: np.ndarray, freq_range: Tuple[float, float],
                                 time_resolution: float = 1.3) -> Optional[float]:
    """
    Estimate DM from detected FRB by measuring the sweep.

    Parameters:
        chunk (np.ndarray): Data chunk (freq, time)
        mask (np.ndarray): Detection mask (freq, time)
        freq_range (Tuple[float, float]): (fmin, fmax) in MHz
        time_resolution (float): Time resolution in ms

    Returns:
        Optional[float]: Estimated DM in pc/cm³, or None if no clear sweep
    """
    # Find FRB pixels
    frb_mask = (mask == 2)

    if frb_mask.sum() < 10:  # Too few pixels
        return None

    # Find the time of arrival at each frequency
    freq_indices, time_indices = np.where(frb_mask)

    if len(np.unique(freq_indices)) < 10:  # Not enough frequency coverage
        return None

    # For each frequency, find the median time of detection
    arrival_times = []
    freqs = []

    for freq_idx in np.unique(freq_indices):
        times = time_indices[freq_indices == freq_idx]
        arrival_times.append(np.median(times))

        # Map freq index to actual frequency
        fmax, fmin = freq_range
        freq_mhz = fmax - (fmax - fmin) * freq_idx / len(mask)
        freqs.append(freq_mhz)

    freqs = np.array(freqs)
    arrival_times = np.array(arrival_times)

    # Sort by frequency
    sort_idx = np.argsort(freqs)
    freqs = freqs[sort_idx]
    arrival_times = arrival_times[sort_idx]

    # Fit DM: t = A + DM * 4.15 * (f^-2)
    # Convert to GHz and time to ms
    freqs_ghz = freqs / 1000.0
    times_ms = arrival_times * time_resolution

    # Linear fit: times_ms vs (1/f^2)
    inv_f_sq = 1.0 / (freqs_ghz ** 2)

    # Fit
    try:
        coeffs = np.polyfit(inv_f_sq, times_ms, 1)
        dm_estimate = coeffs[0] / 4.15

        # Sanity check: DM should be positive and reasonable (0-2000)
        if 0 < dm_estimate < 2000:
            return dm_estimate
        else:
            return None

    except:
        return None


def plot_detection(chunk: np.ndarray, prediction: np.ndarray, probabilities: np.ndarray,
                   chunk_start: int, dm_estimate: Optional[float], metadata: Dict,
                   save_path: str):
    """
    Create a nice detection plot.

    Parameters:
        chunk (np.ndarray): Data chunk (freq, time)
        prediction (np.ndarray): Predicted mask (freq, time)
        probabilities (np.ndarray): Class probabilities (3, freq, time)
        chunk_start (int): Start index of chunk
        dm_estimate (Optional[float]): Estimated DM
        metadata (Dict): Filterbank metadata
        save_path (str): Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Compute frequency range
    fch1 = metadata['fch1']
    foff = metadata['foff']
    nchans = chunk.shape[0]

    fmax = fch1
    fmin = fch1 + foff * (metadata['nchans'] - 1)

    # Time range
    tsamp = metadata['tsamp']
    tstart = chunk_start * tsamp
    tend = tstart + chunk.shape[1] * tsamp

    extent = [tstart, tend, fmin, fmax]

    # Plot 1: Original data
    ax = axes[0, 0]
    im = ax.imshow(chunk, aspect='auto', origin='upper', cmap='viridis', extent=extent)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (MHz)')
    ax.set_title('Filterbank Data')
    plt.colorbar(im, ax=ax, label='Intensity')

    # Plot 2: Segmentation
    ax = axes[0, 1]
    colors = np.array([
        [0, 0, 0],      # Background
        [1, 0, 0],      # RFI
        [0, 1, 1],      # FRB
    ])
    mask_rgb = colors[prediction]
    ax.imshow(mask_rgb, aspect='auto', origin='upper', extent=extent)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (MHz)')

    title = 'Detection'
    if dm_estimate is not None:
        title += f' (DM ≈ {dm_estimate:.1f} pc/cm³)'
    ax.set_title(title)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='black', label='Background'),
        Patch(facecolor='red', label='RFI'),
        Patch(facecolor='cyan', label='FRB'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Plot 3: FRB probability
    ax = axes[1, 0]
    im = ax.imshow(probabilities[2], aspect='auto', origin='upper', cmap='hot',
                   extent=extent, vmin=0, vmax=1)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (MHz)')
    ax.set_title('FRB Probability')
    plt.colorbar(im, ax=ax, label='Probability')

    # Plot 4: Dedispersed time series (if FRB detected)
    ax = axes[1, 1]

    frb_detected = (prediction == 2).sum() > 0

    if frb_detected:
        # Time series at each frequency weighted by FRB probability
        time_series = (chunk * (prediction == 2)).sum(axis=0)
        times = np.linspace(tstart, tend, len(time_series))

        ax.plot(times, time_series, 'c-', linewidth=2)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Integrated Intensity')
        ax.set_title('FRB Time Profile')
        ax.grid(True, alpha=0.3)

        # Mark peak
        peak_idx = np.argmax(time_series)
        ax.axvline(times[peak_idx], color='red', linestyle='--', alpha=0.5, label='Peak')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No FRB Detected', ha='center', va='center',
                transform=ax.transAxes, fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])

    # Overall title
    fig.suptitle(f"{metadata.get('source_name', 'Unknown')} - Chunk starting at {tstart:.1f} ms",
                 fontsize=16)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved detection plot: {save_path}")


def process_filterbank(filterbank_path: str, model_path: str, output_dir: str,
                       chunk_size: int = 1024, overlap: int = 256,
                       detection_threshold: float = 0.5, device: str = 'cuda'):
    """
    Process a filterbank file and detect FRBs.

    Parameters:
        filterbank_path (str): Path to filterbank file
        model_path (str): Path to trained model checkpoint
        output_dir (str): Directory to save results
        chunk_size (int): Size of processing chunks
        overlap (int): Overlap between chunks
        detection_threshold (float): Probability threshold for FRB detection
        device (str): Device to use ('cuda' or 'cpu')
    """
    print("="*70)
    print("FRB DETECTION IN FILTERBANK DATA")
    print("="*70)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading model from {model_path}...")
    model = SAMFRBDetector(model_type='vit_b', num_classes=3)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")

    # Read filterbank
    print(f"\nReading filterbank: {filterbank_path}")
    data, metadata = read_filterbank(filterbank_path)

    # Extract 300-500 MHz range (matching training data)
    print(f"\nExtracting 300-500 MHz frequency range...")
    data_extracted = extract_frequency_range(data, metadata, target_fmin=300.0, target_fmax=500.0)

    # Downsample frequency to 1024 channels
    print(f"\nDownsampling frequency channels...")
    data_downsampled = downsample_frequency(data_extracted, target_channels=1024)

    # Extract chunks
    print(f"\nExtracting chunks...")
    chunks = extract_chunks(data_downsampled, chunk_size=chunk_size, overlap=overlap)

    # Process each chunk
    print(f"\nProcessing {len(chunks)} chunks...")
    detections = []

    for i, (chunk, start_idx) in enumerate(chunks):
        print(f"  Processing chunk {i+1}/{len(chunks)} (start: {start_idx * metadata['tsamp']:.1f} ms)...", end=' ')

        # Run inference
        prediction, probabilities = run_inference(model, chunk, device=device)

        # Check for FRB detection
        frb_pixels = (prediction == 2).sum()
        frb_prob_max = probabilities[2].max()

        if frb_pixels > 50 and frb_prob_max > detection_threshold:
            # Estimate DM (using 300-500 MHz range, high freq first)
            freq_range = (500.0, 300.0)  # (fmax, fmin) - model expects high freq first
            dm_estimate = estimate_dm_from_detection(chunk, prediction, freq_range, metadata['tsamp'])

            # Filter by DM range (model trained on DM 100-200)
            if dm_estimate is not None and 100 <= dm_estimate <= 200:
                print(f"FRB DETECTED! ({frb_pixels} pixels, max prob: {frb_prob_max:.3f})")
                print(f"    Estimated DM: {dm_estimate:.1f} pc/cm³")
            elif dm_estimate is not None:
                print(f"Candidate rejected (DM={dm_estimate:.1f} outside 100-200 range)")
                continue  # Skip this detection
            else:
                print(f"Candidate rejected (DM estimation failed)")
                continue  # Skip this detection

            # Save detection plot
            plot_path = output_path / f"detection_chunk_{i:04d}.png"
            plot_detection(chunk, prediction, probabilities, start_idx, dm_estimate,
                          metadata, str(plot_path))

            detections.append({
                'chunk_index': i,
                'start_time_ms': start_idx * metadata['tsamp'],
                'frb_pixels': int(frb_pixels),
                'max_probability': float(frb_prob_max),
                'dm_estimate': dm_estimate,
                'plot_path': str(plot_path),
            })
        else:
            print("No FRB")

    # Summary
    print("\n" + "="*70)
    print(f"DETECTION SUMMARY")
    print("="*70)
    print(f"Total chunks processed: {len(chunks)}")
    print(f"FRB detections: {len(detections)}")

    if detections:
        print("\nDetected FRBs:")
        for det in detections:
            dm_str = f"DM={det['dm_estimate']:.1f}" if det['dm_estimate'] else "DM=unknown"
            print(f"  - t={det['start_time_ms']:.1f} ms, {dm_str}, prob={det['max_probability']:.3f}")

    # Save detection summary
    import json
    summary_path = output_path / 'detection_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'filterbank': filterbank_path,
            'model': model_path,
            'total_chunks': len(chunks),
            'num_detections': len(detections),
            'detections': detections,
        }, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Detect FRBs in filterbank data')
    parser.add_argument('--filterbank', type=str, required=True,
                        help='Path to filterbank file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str, default='detections',
                        help='Output directory for results')
    parser.add_argument('--chunk_size', type=int, default=1024,
                        help='Chunk size in time bins')
    parser.add_argument('--overlap', type=int, default=256,
                        help='Overlap between chunks')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Detection threshold (probability)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device to use')

    args = parser.parse_args()

    process_filterbank(
        filterbank_path=args.filterbank,
        model_path=args.model,
        output_dir=args.output,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        detection_threshold=args.threshold,
        device=args.device,
    )


if __name__ == "__main__":
    main()
