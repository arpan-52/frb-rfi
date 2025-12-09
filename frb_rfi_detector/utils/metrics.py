"""
Metrics for evaluating FRB-RFI detection performance.
"""

import numpy as np
from typing import Dict
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def compute_iou(pred: np.ndarray, target: np.ndarray, class_id: int) -> float:
    """
    Compute Intersection over Union (IoU) for a specific class.

    Parameters:
        pred (np.ndarray): Predicted labels
        target (np.ndarray): Ground truth labels
        class_id (int): Class ID to compute IoU for

    Returns:
        float: IoU score
    """
    pred_mask = (pred == class_id)
    target_mask = (target == class_id)

    intersection = np.logical_and(pred_mask, target_mask).sum()
    union = np.logical_or(pred_mask, target_mask).sum()

    if union == 0:
        return 0.0

    iou = intersection / union
    return iou


def compute_metrics(pred: np.ndarray, target: np.ndarray, num_classes: int = 3) -> Dict[str, float]:
    """
    Compute comprehensive metrics for multi-class segmentation.

    Classes:
        0: Background
        1: RFI
        2: FRB

    Parameters:
        pred (np.ndarray): Predicted labels (N, H, W)
        target (np.ndarray): Ground truth labels (N, H, W)
        num_classes (int): Number of classes

    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    # Flatten arrays
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    # Compute confusion matrix
    cm = confusion_matrix(target_flat, pred_flat, labels=np.arange(num_classes))

    # Compute per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        target_flat, pred_flat, labels=np.arange(num_classes), average=None, zero_division=0
    )

    # Compute IoU for each class
    iou_per_class = []
    for class_id in range(num_classes):
        iou = compute_iou(pred, target, class_id)
        iou_per_class.append(iou)

    # Mean IoU
    miou = np.mean(iou_per_class)

    # Overall accuracy
    accuracy = (pred_flat == target_flat).sum() / len(pred_flat)

    # Compile metrics
    metrics = {
        'accuracy': accuracy,
        'miou': miou,
    }

    # Add per-class metrics
    class_names = ['background', 'rfi', 'frb']
    for i, name in enumerate(class_names):
        metrics[f'iou_class_{i}'] = iou_per_class[i]
        metrics[f'precision_class_{i}'] = precision[i]
        metrics[f'recall_class_{i}'] = recall[i]
        metrics[f'f1_class_{i}'] = f1[i]
        metrics[f'{name}_iou'] = iou_per_class[i]
        metrics[f'{name}_f1'] = f1[i]

    return metrics


def print_metrics(metrics: Dict[str, float]):
    """Pretty print metrics."""
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)

    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Mean IoU: {metrics['miou']:.4f}")

    print(f"\nPer-Class Metrics:")
    class_names = ['Background', 'RFI', 'FRB']
    for i, name in enumerate(class_names):
        print(f"\n  {name}:")
        print(f"    IoU:       {metrics[f'iou_class_{i}']:.4f}")
        print(f"    Precision: {metrics[f'precision_class_{i}']:.4f}")
        print(f"    Recall:    {metrics[f'recall_class_{i}']:.4f}")
        print(f"    F1 Score:  {metrics[f'f1_class_{i}']:.4f}")

    print("\n" + "="*50)


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)

    # Create dummy predictions and targets
    pred = np.random.randint(0, 3, size=(10, 1024, 1024))
    target = np.random.randint(0, 3, size=(10, 1024, 1024))

    # Compute metrics
    metrics = compute_metrics(pred, target)

    # Print metrics
    print_metrics(metrics)
