"""CNN training script."""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Dict, Tuple
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.models.cnn.model import AnomalyDetector, export_torchscript, export_onnx
from backend.models.cnn.dataset import create_dataloaders
from backend.models.cnn.eval import evaluate_model
from backend.utils.config import load_yaml_config, get_device


def train_epoch(
    model: nn.Module,
    train_loader,
    criterion,
    optimizer,
    device: str,
    epoch: int,
    writer: SummaryWriter,
    use_amp: bool = True,
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        writer: TensorBoard writer
        use_amp: Use automatic mixed precision

    Returns:
        Dictionary of training metrics
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    scaler = torch.cuda.amp.GradScaler() if use_amp and device == "cuda" else None

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Mixed precision forward pass
        if use_amp and device == "cuda":
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            "loss": loss.item(),
            "acc": 100.0 * correct / total,
        })

        # Log to tensorboard
        global_step = epoch * len(train_loader) + batch_idx
        writer.add_scalar("train/loss_step", loss.item(), global_step)

    # Epoch metrics
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
    }


def validate_epoch(
    model: nn.Module,
    val_loader,
    criterion,
    device: str,
) -> Dict[str, float]:
    """Validate model.

    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on

    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Prob of positive class

    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "predictions": np.array(all_preds),
        "labels": np.array(all_labels),
        "probs": np.array(all_probs),
    }


def train_model(config_path: str) -> None:
    """Main training function.

    Args:
        config_path: Path to configuration YAML
    """
    # Load configuration
    config = load_yaml_config(config_path)

    # Extract config sections
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    data_config = config.get("data", {})
    logging_config = config.get("logging", {})
    export_config = config.get("export", {})

    # Setup device
    device = get_device()
    print(f"Using device: {device}")

    # Create directories
    checkpoint_dir = Path(logging_config.get("checkpoint_dir", "./checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(logging_config.get("log_dir", "./logs/cnn"))
    log_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter(log_dir)

    # Create data loaders
    print("Creating data loaders...")
    dataloaders = create_dataloaders(
        data_dir=data_config.get("dataset_path", "./data/synth"),
        batch_size=training_config.get("batch_size", 32),
        num_workers=data_config.get("num_workers", 4),
        img_size=data_config.get("img_size", 224),
        pin_memory=data_config.get("pin_memory", True),
    )

    # Create model
    print("Creating model...")
    model = AnomalyDetector(
        num_classes=model_config.get("num_classes", 2),
        pretrained=model_config.get("pretrained", True),
        dropout=model_config.get("dropout", 0.3),
    )
    model = model.to(device)

    # Loss function
    if training_config.get("loss") == "bce_with_logits":
        pos_weight = torch.tensor([training_config.get("pos_weight", 2.0)]).to(device)
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config.get("learning_rate", 3e-4),
        weight_decay=training_config.get("weight_decay", 1e-4),
        betas=training_config.get("betas", [0.9, 0.999]),
    )

    # Learning rate scheduler
    if training_config.get("scheduler") == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_config.get("epochs", 20),
            eta_min=training_config.get("min_lr", 1e-6),
        )
    else:
        scheduler = None

    # Training loop
    epochs = training_config.get("epochs", 20)
    best_val_loss = float("inf")
    patience = training_config.get("patience", 5)
    patience_counter = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # Train
        train_metrics = train_epoch(
            model=model,
            train_loader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            writer=writer,
            use_amp=training_config.get("use_amp", True),
        )

        # Validate
        val_metrics = validate_epoch(
            model=model,
            val_loader=dataloaders["val"],
            criterion=criterion,
            device=device,
        )

        # Log metrics
        print(f"Train Loss: {train_metrics['loss']:.4f}, "
              f"Train Acc: {train_metrics['accuracy']:.2f}%")
        print(f"Val Loss: {val_metrics['loss']:.4f}, "
              f"Val Acc: {val_metrics['accuracy']:.2f}%")

        writer.add_scalar("train/loss", train_metrics["loss"], epoch)
        writer.add_scalar("train/accuracy", train_metrics["accuracy"], epoch)
        writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)

        # Learning rate scheduler
        if scheduler:
            scheduler.step()
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }

            checkpoint_path = checkpoint_dir / "cnn_best.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break

    # Save final model
    final_checkpoint_path = checkpoint_dir / "cnn_last.pt"
    torch.save({
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, final_checkpoint_path)
    print(f"Saved final model to {final_checkpoint_path}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(
        model=model,
        test_loader=dataloaders["test"],
        device=device,
    )

    print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"Test ROC-AUC: {test_metrics.get('roc_auc', 0.0):.4f}")
    print(f"Test PR-AUC: {test_metrics.get('pr_auc', 0.0):.4f}")
    print(f"Test F1: {test_metrics.get('f1', 0.0):.4f}")

    # Export models
    if export_config.get("torchscript", True):
        export_dir = Path(export_config.get("output_dir", "./checkpoints"))
        torchscript_path = export_dir / "cnn_best_torchscript.pt"
        export_torchscript(model, str(torchscript_path))

    if export_config.get("onnx", True):
        export_dir = Path(export_config.get("output_dir", "./checkpoints"))
        onnx_path = export_dir / "cnn_best.onnx"
        export_onnx(model, str(onnx_path))

    writer.close()
    print("Training complete!")


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description="Train CNN for anomaly detection")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_cnn.yaml",
        help="Path to configuration YAML",
    )
    args = parser.parse_args()

    train_model(args.config)


if __name__ == "__main__":
    main()

