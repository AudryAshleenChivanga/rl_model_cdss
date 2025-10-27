"""CNN evaluation utilities."""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from typing import Dict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def evaluate_model(
    model: torch.nn.Module,
    test_loader,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluate model on test set.

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to use

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds) * 100,
        "precision": precision_score(all_labels, all_preds, average="binary", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="binary", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="binary", zero_division=0),
    }

    # ROC-AUC and PR-AUC (only if we have both classes)
    if len(np.unique(all_labels)) > 1:
        metrics["roc_auc"] = roc_auc_score(all_labels, all_probs)
        metrics["pr_auc"] = average_precision_score(all_labels, all_probs)
    else:
        metrics["roc_auc"] = 0.0
        metrics["pr_auc"] = 0.0

    # Confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(all_labels, all_preds)

    # Classification report
    metrics["classification_report"] = classification_report(
        all_labels, all_preds, target_names=["Normal", "Lesion"], zero_division=0
    )

    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: str,
    class_names: list = ["Normal", "Lesion"],
) -> None:
    """Plot confusion matrix.

    Args:
        cm: Confusion matrix
        output_path: Output file path
        class_names: Class names
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def plot_roc_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    output_path: str,
) -> None:
    """Plot ROC curve.

    Args:
        labels: True labels
        probs: Predicted probabilities
        output_path: Output file path
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"ROC curve saved to {output_path}")


def plot_precision_recall_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    output_path: str,
) -> None:
    """Plot Precision-Recall curve.

    Args:
        labels: True labels
        probs: Predicted probabilities
        output_path: Output file path
    """
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR Curve (AP = {ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"PR curve saved to {output_path}")


def generate_evaluation_report(
    model: torch.nn.Module,
    test_loader,
    device: str,
    output_dir: str,
) -> None:
    """Generate comprehensive evaluation report with plots.

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to use
        output_dir: Output directory for plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate
    print("Evaluating model...")
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute metrics
    metrics = evaluate_model(model, test_loader, device)

    # Print report
    print("\n" + "=" * 50)
    print("EVALUATION REPORT")
    print("=" * 50)
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print("\nClassification Report:")
    print(metrics['classification_report'])
    print("=" * 50)

    # Save plots
    if len(np.unique(all_labels)) > 1:
        plot_confusion_matrix(
            metrics['confusion_matrix'],
            str(output_dir / "confusion_matrix.png"),
        )
        plot_roc_curve(all_labels, all_probs, str(output_dir / "roc_curve.png"))
        plot_precision_recall_curve(
            all_labels, all_probs, str(output_dir / "pr_curve.png")
        )

    # Save metrics to file
    with open(output_dir / "metrics.txt", "w") as f:
        f.write("EVALUATION METRICS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Accuracy: {metrics['accuracy']:.2f}%\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1']:.4f}\n")
        f.write(f"ROC-AUC: {metrics['roc_auc']:.4f}\n")
        f.write(f"PR-AUC: {metrics['pr_auc']:.4f}\n")
        f.write("\n" + metrics['classification_report'])

    print(f"\nEvaluation report saved to {output_dir}")

