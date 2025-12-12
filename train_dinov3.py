"""
Training script for DINOv3 FRB detector.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os
import argparse
import yaml
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import json

from frb_rfi_detector.models.dinov3_frb_detector import DINOv3FRBDetector, CombinedLoss
from frb_rfi_detector.dataset_dinov3 import create_dataloaders


class Trainer:
    """Trainer for DINOv3 FRB detector."""

    def __init__(self, config: dict, output_dir: str):
        self.config = config
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create output directories
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir, 'checkpoints').mkdir(exist_ok=True)
        Path(output_dir, 'logs').mkdir(exist_ok=True)

        # Save config
        with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)

        # Initialize model
        self.model = self._create_model()
        self.model = self.model.to(self.device)

        # Initialize loss
        self.criterion = CombinedLoss(
            detection_weight=config['training']['loss']['detection_weight'],
            dm_weight=config['training']['loss']['dm_regression_weight'],
            dm_loss_type=config['training']['loss']['dm_loss_type'],
        )

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Initialize scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision
        self.use_amp = config['training']['mixed_precision']
        self.scaler = GradScaler() if self.use_amp else None

        # Tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.train_history = []
        self.val_history = []

    def _create_model(self) -> nn.Module:
        """Create DINOv3 model from config."""
        model_config = self.config['model']

        # Get DM range from data config
        dm_range = (
            self.config['data']['dispersion']['dm_min'],
            self.config['data']['dispersion']['dm_max']
        )

        # LoRA config if needed
        lora_config = None
        if model_config.get('use_lora', False):
            lora_config = {
                'r': model_config.get('lora_rank', 16),
                'lora_alpha': model_config.get('lora_alpha', 32),
            }

        model = DINOv3FRBDetector(
            model_type=model_config['backbone'],
            freeze_backbone=model_config['freeze_backbone'],
            unfreeze_last_n_blocks=model_config['unfreeze_last_n_blocks'],
            hidden_dims=model_config['head']['hidden_dims'],
            dropout=model_config['head']['dropout'],
            dm_range=dm_range,
            use_lora=model_config.get('use_lora', False),
            lora_config=lora_config,
        )

        return model

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with differential learning rates."""
        opt_config = self.config['training']['optimizer']

        # Separate parameters
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'backbone' in name:
                    backbone_params.append(param)
                else:
                    head_params.append(param)

        # Create parameter groups
        param_groups = []
        if len(head_params) > 0:
            param_groups.append({
                'params': head_params,
                'lr': opt_config['lr'],
                'name': 'head',
            })
        if len(backbone_params) > 0:
            param_groups.append({
                'params': backbone_params,
                'lr': opt_config['backbone_lr'],
                'name': 'backbone',
            })

        optimizer = optim.AdamW(
            param_groups,
            weight_decay=opt_config['weight_decay'],
            betas=opt_config['betas'],
        )

        print(f"\nOptimizer created:")
        print(f"  Head params: {sum(p.numel() for p in head_params):,} @ LR={opt_config['lr']}")
        print(f"  Backbone params: {sum(p.numel() for p in backbone_params):,} @ LR={opt_config['backbone_lr']}")

        return optimizer

    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        sched_config = self.config['training']['scheduler']
        sched_type = sched_config['type']

        if sched_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['num_epochs'],
                eta_min=sched_config['min_lr'],
            )
        elif sched_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1,
            )
        elif sched_type == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
            )
        else:
            scheduler = None

        return scheduler

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_det_loss = 0.0
        total_dm_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')

        for images, frb_labels, dm_values in pbar:
            images = images.to(self.device)
            frb_labels = frb_labels.to(self.device)
            dm_values = dm_values.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            if self.use_amp:
                with autocast():
                    det_logits, dm_preds = self.model(images)
                    losses = self.criterion(det_logits, dm_preds, frb_labels, dm_values)
                    loss = losses['total_loss']

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                det_logits, dm_preds = self.model(images)
                losses = self.criterion(det_logits, dm_preds, frb_labels, dm_values)
                loss = losses['total_loss']

                loss.backward()
                self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            total_det_loss += losses['detection_loss'].item()
            total_dm_loss += losses['dm_loss'].item()

            # Accuracy
            det_probs = torch.sigmoid(det_logits).squeeze(-1)
            det_preds = (det_probs > 0.5).float()
            correct += (det_preds == frb_labels.squeeze(-1)).sum().item()
            total += frb_labels.size(0)

            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': correct / total,
            })

        metrics = {
            'loss': total_loss / len(train_loader),
            'det_loss': total_det_loss / len(train_loader),
            'dm_loss': total_dm_loss / len(train_loader),
            'accuracy': correct / total,
        }

        return metrics

    def validate(self, val_loader: DataLoader, epoch: int) -> dict:
        """Validate model."""
        self.model.eval()

        total_loss = 0.0
        total_det_loss = 0.0
        total_dm_loss = 0.0
        correct = 0
        total = 0
        dm_errors = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]')

            for images, frb_labels, dm_values in pbar:
                images = images.to(self.device)
                frb_labels = frb_labels.to(self.device)
                dm_values = dm_values.to(self.device)

                # Forward pass
                det_logits, dm_preds = self.model(images)
                losses = self.criterion(det_logits, dm_preds, frb_labels, dm_values)

                # Metrics
                total_loss += losses['total_loss'].item()
                total_det_loss += losses['detection_loss'].item()
                total_dm_loss += losses['dm_loss'].item()

                # Accuracy
                det_probs = torch.sigmoid(det_logits).squeeze(-1)
                det_preds = (det_probs > 0.5).float()
                correct += (det_preds == frb_labels.squeeze(-1)).sum().item()
                total += frb_labels.size(0)

                # DM error (only for samples with FRB)
                frb_mask = frb_labels.squeeze(-1) > 0.5
                if frb_mask.sum() > 0:
                    dm_error = torch.abs(dm_preds[frb_mask] - dm_values[frb_mask])
                    dm_errors.extend(dm_error.cpu().numpy().tolist())

        metrics = {
            'loss': total_loss / len(val_loader),
            'det_loss': total_det_loss / len(val_loader),
            'dm_loss': total_dm_loss / len(val_loader),
            'accuracy': correct / total,
            'dm_mae': np.mean(dm_errors) if dm_errors else 0.0,
        }

        return metrics

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop."""
        num_epochs = self.config['training']['num_epochs']

        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)

        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            self.train_history.append(train_metrics)

            # Validate
            val_metrics = self.validate(val_loader, epoch)
            self.val_history.append(val_metrics)

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

            # Print summary
            print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, DM MAE: {val_metrics['dm_mae']:.2f}")

            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics)

        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)

    def save_checkpoint(self, epoch: int, metrics: dict):
        """Save model checkpoint."""
        save_config = self.config['training']

        # Always save latest
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
        }

        latest_path = os.path.join(self.output_dir, 'checkpoints', 'latest.pt')
        torch.save(checkpoint, latest_path)

        # Save best
        monitor = save_config['monitor_metric']
        if monitor == 'val_loss':
            is_best = metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = metrics['loss']
        elif monitor == 'val_detection_acc':
            is_best = metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = metrics['accuracy']
        else:
            is_best = False

        if is_best:
            best_path = os.path.join(self.output_dir, 'checkpoints', 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"  💾 Saved best model ({monitor}={metrics.get(monitor.replace('val_', ''), 0):.4f})")

        # Periodic save
        if (epoch + 1) % save_config['save_every'] == 0:
            epoch_path = os.path.join(self.output_dir, 'checkpoints', f'epoch_{epoch+1}.pt')
            torch.save(checkpoint, epoch_path)


def main():
    parser = argparse.ArgumentParser(description='Train DINOv3 FRB Detector')
    parser.add_argument('--config', type=str, default='config/dinov3_frb_config.yaml',
                        help='Path to config file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (overrides config)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device to use')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override device
    if not torch.cuda.is_available() and args.device == 'cuda':
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Create output directory with timestamp
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(config['paths']['output_dir'], f'run_{timestamp}')

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        config=config,
        samples_per_epoch=config['training']['samples_per_epoch'],
        val_split=config['training']['val_split'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create trainer
    trainer = Trainer(config=config, output_dir=output_dir)

    # Train
    trainer.train(train_loader, val_loader)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
