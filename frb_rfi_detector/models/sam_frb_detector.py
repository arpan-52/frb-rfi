"""
SAM-based FRB Detector

This module adapts the Segment Anything Model (SAM) for multi-class
segmentation of FRBs and RFI in radio astronomy data.

Classes:
    0: Background/Noise
    1: RFI
    2: FRB
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class SAMFRBDetector(nn.Module):
    """
    SAM-based detector for FRB and RFI segmentation.

    This model uses a pre-trained SAM encoder and adds a custom decoder
    for 3-class segmentation (Background, RFI, FRB).

    Parameters:
        model_type (str): SAM model type ('vit_b', 'vit_l', 'vit_h')
        checkpoint_path (Optional[str]): Path to SAM checkpoint
        num_classes (int): Number of output classes (default: 3)
        freeze_encoder (bool): Whether to freeze encoder weights
    """

    def __init__(
        self,
        model_type: str = 'vit_b',
        checkpoint_path: Optional[str] = None,
        num_classes: int = 3,
        freeze_encoder: bool = False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.model_type = model_type

        try:
            from segment_anything import sam_model_registry, SamPredictor
            self.sam_available = True
        except ImportError:
            print("Warning: segment_anything not installed. Using simplified model.")
            self.sam_available = False

        if self.sam_available:
            # Load SAM model
            if checkpoint_path:
                sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            else:
                # Initialize without checkpoint (will need to download)
                print(f"Initializing {model_type} SAM model without checkpoint")
                print("Note: For best results, download SAM checkpoint from:")
                print("https://github.com/facebookresearch/segment-anything#model-checkpoints")
                sam = sam_model_registry[model_type]()

            self.image_encoder = sam.image_encoder
            self.prompt_encoder = sam.prompt_encoder
            self.mask_decoder = sam.mask_decoder

            # Freeze encoder if specified
            if freeze_encoder:
                for param in self.image_encoder.parameters():
                    param.requires_grad = False

            # Get encoder output dimension
            if model_type == 'vit_h':
                encoder_dim = 1280
            elif model_type == 'vit_l':
                encoder_dim = 1024
            else:  # vit_b
                encoder_dim = 768

            # Custom segmentation head for multi-class output
            self.segmentation_head = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, num_classes, kernel_size=1),
            )

        else:
            # Simplified model without SAM (for testing)
            self.encoder = self._create_simple_encoder()
            self.decoder = self._create_simple_decoder(num_classes)

    def _create_simple_encoder(self):
        """Create a simple encoder for testing when SAM is not available."""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def _create_simple_decoder(self, num_classes):
        """Create a simple decoder for testing when SAM is not available."""
        return nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters:
            x (torch.Tensor): Input images (B, 3, H, W)

        Returns:
            torch.Tensor: Segmentation logits (B, num_classes, H, W)
        """
        if self.sam_available:
            # Encode image with SAM encoder
            # Output shape: (B, 256, 64, 64) for ViT-B/L or (B, 256, 64, 64) for all
            image_embeddings = self.image_encoder(x)

            # Apply our custom segmentation head directly to encoder output
            # This bypasses SAM's prompt-based mask decoder
            masks = self.segmentation_head(image_embeddings)

            # Upsample to input size (1024, 1024)
            masks = F.interpolate(masks, size=(1024, 1024), mode='bilinear', align_corners=False)

        else:
            # Simple model forward pass
            features = self.encoder(x)
            masks = self.decoder(features)

            # Upsample to input size if needed
            if masks.shape[-2:] != x.shape[-2:]:
                masks = F.interpolate(masks, size=x.shape[-2:], mode='bilinear', align_corners=False)

        return masks

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict class labels and probabilities.

        Parameters:
            x (torch.Tensor): Input images (B, 3, H, W)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Class predictions (B, H, W)
                - Class probabilities (B, num_classes, H, W)
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        return preds, probs


class FRBDetectionLoss(nn.Module):
    """
    Combined loss for FRB and RFI detection.

    Combines:
    - Cross-entropy loss for segmentation
    - Dice loss for better boundary detection
    - Optional class weighting for imbalanced classes
    """

    def __init__(
        self,
        num_classes: int = 3,
        class_weights: Optional[torch.Tensor] = None,
        dice_weight: float = 0.5,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.class_weights = class_weights
        self.dice_weight = dice_weight

        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.

        Parameters:
            pred (torch.Tensor): Predictions (B, C, H, W)
            target (torch.Tensor): Targets (B, H, W)

        Returns:
            torch.Tensor: Dice loss
        """
        # Convert target to one-hot
        target_one_hot = F.one_hot(target, num_classes=self.num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        # Apply softmax to predictions
        pred_soft = F.softmax(pred, dim=1)

        # Compute Dice coefficient
        intersection = (pred_soft * target_one_hot).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice = (2.0 * intersection + 1e-8) / (union + 1e-8)
        dice_loss = 1.0 - dice.mean()

        return dice_loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Parameters:
            pred (torch.Tensor): Predictions (B, C, H, W)
            target (torch.Tensor): Targets (B, H, W)

        Returns:
            Dict[str, torch.Tensor]: Dictionary with loss components
        """
        # Cross-entropy loss
        ce = self.ce_loss(pred, target)

        # Dice loss
        dice = self.dice_loss(pred, target)

        # Combined loss
        total_loss = (1 - self.dice_weight) * ce + self.dice_weight * dice

        return {
            'loss': total_loss,
            'ce_loss': ce,
            'dice_loss': dice,
        }


if __name__ == "__main__":
    print("Testing SAM FRB Detector...")

    # Test model creation
    model = SAMFRBDetector(model_type='vit_b', num_classes=3)
    print(f"Model created successfully")
    print(f"SAM available: {model.sam_available}")

    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 3, 1024, 1024)
    y = torch.randint(0, 3, (batch_size, 1024, 1024))

    print(f"\nInput shape: {x.shape}")
    print(f"Target shape: {y.shape}")

    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")

    # Test loss
    criterion = FRBDetectionLoss(num_classes=3)
    losses = criterion(output, y)

    print(f"\nLoss components:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")

    # Test prediction
    preds, probs = model.predict(x)
    print(f"\nPredictions shape: {preds.shape}")
    print(f"Probabilities shape: {probs.shape}")
    print(f"Unique predictions: {torch.unique(preds)}")
