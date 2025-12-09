"""
Training script for SAM-based FRB detector.

This script trains the model to detect FRBs in the presence of RFI.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os
import argparse
from tqdm import tqdm
from typing import Dict
import json
from datetime import datetime

from models.sam_frb_detector import SAMFRBDetector, FRBDetectionLoss
from dataset import FRBRFIDataModule
from utils.metrics import compute_metrics


class Trainer:
    """
    Trainer class for FRB-RFI detection model.

    Parameters:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        num_epochs (int): Number of epochs
        output_dir (str): Directory to save checkpoints and logs
        mixed_precision (bool): Use mixed precision training
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion,
        optimizer,
        device: torch.device,
        num_epochs: int = 50,
        output_dir: str = 'outputs',
        mixed_precision: bool = True,
        scheduler = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.output_dir = output_dir
        self.mixed_precision = mixed_precision

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)

        # Mixed precision scaler
        self.scaler = GradScaler() if mixed_precision else None

        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        self.best_val_loss = float('inf')
        self.best_val_iou = 0.0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_ce_loss = 0.0
        total_dice_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} [Train]')

        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.mixed_precision:
                with autocast():
                    outputs = self.model(images)
                    losses = self.criterion(outputs, masks)
                    loss = losses['loss']

                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                losses = self.criterion(outputs, masks)
                loss = losses['loss']

                # Backward pass
                loss.backward()
                self.optimizer.step()

            # Track losses
            total_loss += loss.item()
            total_ce_loss += losses['ce_loss'].item()
            total_dice_loss += losses['dice_loss'].item()

            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'ce': losses['ce_loss'].item(),
                'dice': losses['dice_loss'].item(),
            })

        # Average losses
        avg_loss = total_loss / len(self.train_loader)
        avg_ce = total_ce_loss / len(self.train_loader)
        avg_dice = total_dice_loss / len(self.train_loader)

        return {
            'loss': avg_loss,
            'ce_loss': avg_ce,
            'dice_loss': avg_dice,
        }

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()

        total_loss = 0.0
        total_ce_loss = 0.0
        total_dice_loss = 0.0

        all_preds = []
        all_targets = []

        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} [Val]')

        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            if self.mixed_precision:
                with autocast():
                    outputs = self.model(images)
                    losses = self.criterion(outputs, masks)
            else:
                outputs = self.model(images)
                losses = self.criterion(outputs, masks)

            # Track losses
            total_loss += losses['loss'].item()
            total_ce_loss += losses['ce_loss'].item()
            total_dice_loss += losses['dice_loss'].item()

            # Get predictions
            preds = torch.argmax(outputs, dim=1)

            # Store for metrics
            all_preds.append(preds.cpu().numpy())
            all_targets.append(masks.cpu().numpy())

            pbar.set_postfix({
                'loss': losses['loss'].item(),
            })

        # Compute metrics
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        metrics = compute_metrics(all_preds, all_targets, num_classes=3)

        # Average losses
        avg_loss = total_loss / len(self.val_loader)
        avg_ce = total_ce_loss / len(self.val_loader)
        avg_dice = total_dice_loss / len(self.val_loader)

        return {
            'loss': avg_loss,
            'ce_loss': avg_ce,
            'dice_loss': avg_dice,
            **metrics
        }

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save latest checkpoint
        checkpoint_path = os.path.join(self.output_dir, 'checkpoints', f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.output_dir, 'checkpoints', 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"  Saved best model to {best_path}")

    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.mixed_precision}")
        print(f"Output directory: {self.output_dir}")

        for epoch in range(self.num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            self.train_losses.append(train_metrics)

            # Validate
            val_metrics = self.validate(epoch)
            self.val_losses.append(val_metrics)
            self.val_metrics.append(val_metrics)

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_metrics['loss'])

            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.num_epochs} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val mIoU: {val_metrics['miou']:.4f}")
            print(f"  Val FRB IoU: {val_metrics['iou_class_2']:.4f}")
            print(f"  Val FRB F1: {val_metrics['f1_class_2']:.4f}")

            # Save checkpoint
            is_best = val_metrics['miou'] > self.best_val_iou
            if is_best:
                self.best_val_iou = val_metrics['miou']

            self.save_checkpoint(epoch, val_metrics, is_best=is_best)

            # Save training history
            history = {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'val_metrics': self.val_metrics,
            }
            with open(os.path.join(self.output_dir, 'training_history.json'), 'w') as f:
                json.dump(history, f, indent=2)

        print("\nTraining complete!")
        print(f"Best validation mIoU: {self.best_val_iou:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train FRB-RFI detector')

    # Model args
    parser.add_argument('--model_type', type=str, default='vit_b',
                        choices=['vit_b', 'vit_l', 'vit_h'],
                        help='SAM model type')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to SAM checkpoint')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze encoder weights')

    # Data args
    parser.add_argument('--train_size', type=int, default=5000,
                        help='Training set size')
    parser.add_argument('--val_size', type=int, default=500,
                        help='Validation set size')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--dm_min', type=float, default=100,
                        help='Minimum DM')
    parser.add_argument('--dm_max', type=float, default=200,
                        help='Maximum DM')
    parser.add_argument('--snr_min', type=float, default=5,
                        help='Minimum SNR')
    parser.add_argument('--snr_max', type=float, default=15,
                        help='Maximum SNR')

    # Training args
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--dice_weight', type=float, default=0.5,
                        help='Weight for Dice loss')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--no_mixed_precision', action='store_true',
                        help='Disable mixed precision training')

    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # Save args
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Create data module
    print("\nInitializing data module...")
    data_module = FRBRFIDataModule(
        train_size=args.train_size,
        val_size=args.val_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        generate_on_fly=True,
        dm_range=(args.dm_min, args.dm_max),
        frb_snr_range=(args.snr_min, args.snr_max),
    )

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model
    print("\nInitializing model...")
    model = SAMFRBDetector(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint,
        num_classes=3,
        freeze_encoder=args.freeze_encoder,
    )
    model = model.to(device)

    # Create loss
    criterion = FRBDetectionLoss(
        num_classes=3,
        dice_weight=args.dice_weight,
    )

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        output_dir=output_dir,
        mixed_precision=not args.no_mixed_precision,
        scheduler=scheduler,
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
