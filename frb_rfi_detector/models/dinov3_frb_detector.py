"""
DINOv3-based FRB Detector

This module uses DINOv2/v3 as a frozen or fine-tunable backbone for detecting FRBs
in radio astronomy data and estimating their dispersion measures.

The model learns holistic FRB patterns in the presence of RFI and noise,
rather than pixel-wise segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import warnings


class FRBDetectionHead(nn.Module):
    """
    Lightweight detection head for FRB detection and DM estimation.

    Takes DINOv3 embeddings and produces:
    1. FRB detection (binary classification)
    2. DM estimation (regression)
    """

    def __init__(
        self,
        embed_dim: int = 1536,
        hidden_dims: List[int] = [512, 128],
        dropout: float = 0.1,
        dm_range: Tuple[float, float] = (30, 2000),
    ):
        super().__init__()

        self.dm_min, self.dm_max = dm_range

        # Shared adapter
        layers = []
        in_dim = embed_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        self.adapter = nn.Sequential(*layers)

        # Task-specific heads
        self.frb_classifier = nn.Linear(hidden_dims[-1], 1)  # Binary classification
        self.dm_regressor = nn.Linear(hidden_dims[-1], 1)    # DM value

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: DINOv3 embeddings [B, embed_dim]

        Returns:
            detection_logits: [B, 1] - logits for FRB presence
            dm_values: [B, 1] - estimated DM values (unnormalized)
        """
        features = self.adapter(x)

        detection_logits = self.frb_classifier(features)
        dm_raw = self.dm_regressor(features)

        # Normalize DM to valid range using sigmoid
        dm_normalized = torch.sigmoid(dm_raw) * (self.dm_max - self.dm_min) + self.dm_min

        return detection_logits, dm_normalized


class DINOv3FRBDetector(nn.Module):
    """
    Complete FRB detector using DINOv3 backbone.

    Args:
        model_type: DINOv2/v3 model type ('dinov2_vits14', 'dinov2_vitb14',
                    'dinov2_vitl14', 'dinov2_vitg14')
        freeze_backbone: Whether to freeze backbone weights
        unfreeze_last_n_blocks: Unfreeze last N transformer blocks (0 = all frozen)
        hidden_dims: Hidden dimensions for detection head
        dropout: Dropout probability
        dm_range: (min_dm, max_dm) range
        use_lora: Use LoRA for efficient fine-tuning
        lora_config: LoRA configuration dict
    """

    def __init__(
        self,
        model_type: str = 'dinov2_vitg14',
        freeze_backbone: bool = True,
        unfreeze_last_n_blocks: int = 0,
        hidden_dims: List[int] = [512, 128],
        dropout: float = 0.1,
        dm_range: Tuple[float, float] = (30, 2000),
        use_lora: bool = False,
        lora_config: Optional[Dict] = None,
    ):
        super().__init__()

        self.model_type = model_type
        self.freeze_backbone = freeze_backbone
        self.unfreeze_last_n_blocks = unfreeze_last_n_blocks
        self.use_lora = use_lora

        # Load DINOv2 backbone
        try:
            print(f"Loading {model_type} from torch.hub...")
            self.backbone = torch.hub.load('facebookresearch/dinov2', model_type)
            print(f"Successfully loaded {model_type}")
        except Exception as e:
            raise RuntimeError(f"Failed to load {model_type}: {e}")

        # Get embedding dimension
        embed_dim = self.backbone.embed_dim
        print(f"Backbone embedding dimension: {embed_dim}")

        # Apply freezing strategy
        self._apply_freezing_strategy()

        # Apply LoRA if requested
        if use_lora and unfreeze_last_n_blocks > 0:
            print(f"Applying LoRA with config: {lora_config}")
            self._apply_lora(lora_config)

        # Detection head
        self.head = FRBDetectionHead(
            embed_dim=embed_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            dm_range=dm_range,
        )

        # Print trainable parameters
        self._print_trainable_params()

    def _apply_freezing_strategy(self):
        """Apply freezing strategy to backbone."""
        if self.freeze_backbone:
            # Freeze all backbone parameters
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Frozen entire backbone ❄️")

            # Unfreeze last N blocks if requested
            if self.unfreeze_last_n_blocks > 0:
                total_blocks = len(self.backbone.blocks)
                blocks_to_unfreeze = self.backbone.blocks[-self.unfreeze_last_n_blocks:]

                for block in blocks_to_unfreeze:
                    for param in block.parameters():
                        param.requires_grad = True

                print(f"Unfroze last {self.unfreeze_last_n_blocks}/{total_blocks} blocks 🔥")
        else:
            print("Backbone fully trainable 🔥")

    def _apply_lora(self, lora_config: Optional[Dict]):
        """Apply LoRA to unfrozen blocks."""
        try:
            from peft import get_peft_model, LoraConfig, TaskType

            if lora_config is None:
                lora_config = {
                    'r': 16,
                    'lora_alpha': 32,
                    'target_modules': ['qkv', 'proj'],
                    'lora_dropout': 0.1,
                }

            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                **lora_config
            )

            self.backbone = get_peft_model(self.backbone, peft_config)
            print("Applied LoRA to backbone")
        except ImportError:
            warnings.warn("peft library not installed. Install with: pip install peft")
            print("Skipping LoRA application")

    def _print_trainable_params(self):
        """Print statistics about trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        head_trainable = sum(p.numel() for p in self.head.parameters() if p.requires_grad)

        print(f"\nParameter Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"    - Backbone: {backbone_trainable:,}")
        print(f"    - Head: {head_trainable:,}")
        print()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input images [B, C, H, W] where C=3 (RGB)

        Returns:
            detection_logits: [B, 1] - FRB detection logits
            dm_values: [B, 1] - Estimated DM values
        """
        # Extract features with DINOv3
        with torch.set_grad_enabled(not self.freeze_backbone or self.unfreeze_last_n_blocks > 0):
            # DINOv2 outputs [B, num_patches, embed_dim]
            features = self.backbone(x)

            # Global average pooling over patches
            if len(features.shape) == 3:
                features = features.mean(dim=1)  # [B, embed_dim]

        # Detection head
        detection_logits, dm_values = self.head(features)

        return detection_logits, dm_values

    def predict(
        self,
        x: torch.Tensor,
        detection_threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inference with thresholding.

        Args:
            x: Input images [B, C, H, W]
            detection_threshold: Threshold for FRB detection

        Returns:
            frb_detected: [B] - Binary predictions (True/False)
            frb_probabilities: [B] - Detection probabilities
            dm_estimates: [B] - DM estimates
        """
        detection_logits, dm_values = self.forward(x)

        frb_probabilities = torch.sigmoid(detection_logits).squeeze(-1)
        frb_detected = frb_probabilities > detection_threshold
        dm_estimates = dm_values.squeeze(-1)

        return frb_detected, frb_probabilities, dm_estimates


class CombinedLoss(nn.Module):
    """
    Combined loss for FRB detection and DM regression.
    """

    def __init__(
        self,
        detection_weight: float = 1.0,
        dm_weight: float = 0.5,
        dm_loss_type: str = 'smooth_l1',
    ):
        super().__init__()

        self.detection_weight = detection_weight
        self.dm_weight = dm_weight

        # Detection loss (binary cross-entropy)
        self.detection_loss = nn.BCEWithLogitsLoss()

        # DM regression loss
        if dm_loss_type == 'mse':
            self.dm_loss = nn.MSELoss()
        elif dm_loss_type == 'l1':
            self.dm_loss = nn.L1Loss()
        elif dm_loss_type == 'smooth_l1':
            self.dm_loss = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown dm_loss_type: {dm_loss_type}")

    def forward(
        self,
        detection_logits: torch.Tensor,
        dm_predictions: torch.Tensor,
        detection_labels: torch.Tensor,
        dm_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            detection_logits: [B, 1] - Detection logits
            dm_predictions: [B, 1] - Predicted DM values
            detection_labels: [B, 1] - Ground truth detection (0 or 1)
            dm_labels: [B, 1] - Ground truth DM values

        Returns:
            Dict with 'total_loss', 'detection_loss', 'dm_loss'
        """
        # Detection loss
        det_loss = self.detection_loss(detection_logits, detection_labels)

        # DM loss (only for samples with FRB present)
        frb_mask = detection_labels.squeeze(-1) > 0.5

        if frb_mask.sum() > 0:
            dm_loss = self.dm_loss(
                dm_predictions[frb_mask],
                dm_labels[frb_mask]
            )
        else:
            dm_loss = torch.tensor(0.0, device=detection_logits.device)

        # Combined loss
        total_loss = (
            self.detection_weight * det_loss +
            self.dm_weight * dm_loss
        )

        return {
            'total_loss': total_loss,
            'detection_loss': det_loss,
            'dm_loss': dm_loss,
        }


if __name__ == "__main__":
    # Test model creation
    print("Testing DINOv3FRBDetector...")

    model = DINOv3FRBDetector(
        model_type='dinov2_vitb14',  # Smaller for testing
        freeze_backbone=True,
        unfreeze_last_n_blocks=4,
        hidden_dims=[512, 128],
        dropout=0.1,
        dm_range=(30, 2000),
    )

    # Test forward pass
    x = torch.randn(2, 3, 4096, 4096)
    detection_logits, dm_values = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Detection logits shape: {detection_logits.shape}")
    print(f"DM values shape: {dm_values.shape}")
    print(f"DM values: {dm_values.squeeze().tolist()}")

    # Test prediction
    frb_detected, frb_probs, dm_estimates = model.predict(x)
    print(f"FRB detected: {frb_detected}")
    print(f"FRB probabilities: {frb_probs}")
    print(f"DM estimates: {dm_estimates}")

    print("\n✅ Model test passed!")
