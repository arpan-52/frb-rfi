"""
Visualization utilities for FRB-RFI detection.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Optional, Tuple
import os


def plot_waterfall(
    dynamic_spectrum: np.ndarray,
    title: str = 'Dynamic Spectrum',
    freq_range: Tuple[float, float] = (550, 750),
    time_range: Optional[Tuple[float, float]] = None,
    cmap: str = 'viridis',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
):
    """
    Plot a waterfall (dynamic spectrum).

    Parameters:
        dynamic_spectrum (np.ndarray): Dynamic spectrum (freq, time)
        title (str): Plot title
        freq_range (Tuple[float, float]): Frequency range in MHz
        time_range (Optional[Tuple[float, float]]): Time range in ms
        cmap (str): Colormap
        vmin (Optional[float]): Min value for colormap
        vmax (Optional[float]): Max value for colormap
        figsize (Tuple[int, int]): Figure size
        save_path (Optional[str]): Path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot
    extent = [
        0 if time_range is None else time_range[0],
        dynamic_spectrum.shape[1] if time_range is None else time_range[1],
        freq_range[0],
        freq_range[1]
    ]

    im = ax.imshow(
        dynamic_spectrum,
        aspect='auto',
        origin='upper',
        cmap=cmap,
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Frequency (MHz)', fontsize=12)
    ax.set_title(title, fontsize=14)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Intensity', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig, ax


def plot_segmentation_comparison(
    spectrum: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    title: str = 'Segmentation Comparison',
    freq_range: Tuple[float, float] = (550, 750),
    figsize: Tuple[int, int] = (18, 5),
    save_path: Optional[str] = None,
):
    """
    Plot side-by-side comparison of spectrum, ground truth, and prediction.

    Parameters:
        spectrum (np.ndarray): Dynamic spectrum (freq, time)
        ground_truth (np.ndarray): Ground truth mask (freq, time)
        prediction (np.ndarray): Predicted mask (freq, time)
        title (str): Plot title
        freq_range (Tuple[float, float]): Frequency range in MHz
        figsize (Tuple[int, int]): Figure size
        save_path (Optional[str]): Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Color mapping for classes
    # 0: Background (black), 1: RFI (red), 2: FRB (cyan)
    colors = np.array([
        [0, 0, 0],      # Background - black
        [1, 0, 0],      # RFI - red
        [0, 1, 1],      # FRB - cyan
    ])

    extent = [0, spectrum.shape[1], freq_range[0], freq_range[1]]

    # Plot spectrum
    ax = axes[0]
    im = ax.imshow(spectrum, aspect='auto', origin='upper', cmap='viridis', extent=extent)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (MHz)')
    ax.set_title('Dynamic Spectrum')
    plt.colorbar(im, ax=ax, label='Intensity')

    # Plot ground truth
    ax = axes[1]
    gt_rgb = colors[ground_truth]
    ax.imshow(gt_rgb, aspect='auto', origin='upper', extent=extent)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (MHz)')
    ax.set_title('Ground Truth')

    # Plot prediction
    ax = axes[2]
    pred_rgb = colors[prediction]
    ax.imshow(pred_rgb, aspect='auto', origin='upper', extent=extent)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (MHz)')
    ax.set_title('Prediction')

    # Add legend
    patches = [
        mpatches.Patch(color='black', label='Background'),
        mpatches.Patch(color='red', label='RFI'),
        mpatches.Patch(color='cyan', label='FRB'),
    ]
    axes[2].legend(handles=patches, loc='upper right', fontsize=10)

    fig.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig, axes


def plot_training_history(
    train_losses: list,
    val_losses: list,
    val_metrics: list,
    save_path: Optional[str] = None,
):
    """
    Plot training history.

    Parameters:
        train_losses (list): Training losses
        val_losses (list): Validation losses
        val_metrics (list): Validation metrics
        save_path (Optional[str]): Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    ax = axes[0, 0]
    ax.plot(epochs, [t['loss'] for t in train_losses], 'b-', label='Train Loss')
    ax.plot(epochs, [v['loss'] for v in val_losses], 'r-', label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # mIoU plot
    ax = axes[0, 1]
    ax.plot(epochs, [v['miou'] for v in val_metrics], 'g-', label='Val mIoU')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mIoU')
    ax.set_title('Validation Mean IoU')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Per-class IoU
    ax = axes[1, 0]
    ax.plot(epochs, [v['iou_class_0'] for v in val_metrics], 'k-', label='Background')
    ax.plot(epochs, [v['iou_class_1'] for v in val_metrics], 'r-', label='RFI')
    ax.plot(epochs, [v['iou_class_2'] for v in val_metrics], 'c-', label='FRB')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('IoU')
    ax.set_title('Per-Class IoU')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # FRB F1 Score
    ax = axes[1, 1]
    ax.plot(epochs, [v['f1_class_2'] for v in val_metrics], 'c-', label='FRB F1')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_title('FRB Detection F1 Score')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig, axes


def plot_examples(
    generator,
    n_examples: int = 4,
    save_dir: Optional[str] = None,
):
    """
    Generate and plot example data.

    Parameters:
        generator: FRBRFIDataGenerator instance
        n_examples (int): Number of examples to generate
        save_dir (Optional[str]): Directory to save plots
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for i in range(n_examples):
        # Generate sample
        spectrum, mask, metadata = generator.generate_sample()

        # Create RGB visualization of mask
        colors = np.array([
            [0, 0, 0],      # Background
            [1, 0, 0],      # RFI
            [0, 1, 1],      # FRB
        ])
        mask_rgb = colors[mask]

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        extent = [0, spectrum.shape[1], 550, 750]

        # Spectrum
        ax = axes[0]
        im = ax.imshow(spectrum, aspect='auto', origin='upper', cmap='viridis', extent=extent)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Frequency (MHz)')
        ax.set_title('Dynamic Spectrum (Noise + RFI + FRB)')
        plt.colorbar(im, ax=ax, label='Intensity')

        # Mask
        ax = axes[1]
        ax.imshow(mask_rgb, aspect='auto', origin='upper', extent=extent)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Frequency (MHz)')
        ax.set_title('Ground Truth Labels')

        # Add legend
        patches = [
            mpatches.Patch(color='black', label='Background'),
            mpatches.Patch(color='red', label='RFI'),
            mpatches.Patch(color='cyan', label='FRB'),
        ]
        ax.legend(handles=patches, loc='upper right')

        # Title with metadata
        title = f"Example {i+1}"
        if metadata['has_frb']:
            title += f" | FRB: DM={metadata['frb_dm']:.1f}, SNR={metadata['frb_snr']:.1f}"
        fig.suptitle(title, fontsize=14)

        plt.tight_layout()

        if save_dir:
            save_path = os.path.join(save_dir, f'example_{i+1}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved example {i+1} to {save_path}")

        plt.close()


if __name__ == "__main__":
    # Test visualization
    from frb_rfi_detector.data_generator import FRBRFIDataGenerator

    print("Generating example plots...")

    generator = FRBRFIDataGenerator()

    # Generate examples
    plot_examples(generator, n_examples=2, save_dir='test_plots')

    print("Done!")
