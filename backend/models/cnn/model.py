"""CNN model for anomaly detection."""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class AnomalyDetector(nn.Module):
    """ResNet18-based binary classifier for ulcer detection."""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        """Initialize anomaly detector.

        Args:
            num_classes: Number of output classes
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout probability
        """
        super().__init__()

        # Load pretrained ResNet18
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.backbone = models.resnet18(weights=weights)
        else:
            self.backbone = models.resnet18(weights=None)

        # Get number of features from last layer
        num_features = self.backbone.fc.in_features

        # Replace final fully connected layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Logits (B, num_classes)
        """
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Probabilities (B, num_classes)
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Predict class labels.

        Args:
            x: Input tensor (B, C, H, W)
            threshold: Decision threshold for binary classification

        Returns:
            Predicted labels (B,)
        """
        probs = self.predict_proba(x)
        # For binary classification, use prob of positive class
        positive_prob = probs[:, 1]
        return (positive_prob >= threshold).long()


def load_model(
    checkpoint_path: str,
    device: str = "cpu",
    num_classes: int = 2,
) -> AnomalyDetector:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        num_classes: Number of classes

    Returns:
        Loaded model
    """
    model = AnomalyDetector(num_classes=num_classes, pretrained=False)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


def export_torchscript(
    model: AnomalyDetector,
    output_path: str,
    example_input: Optional[torch.Tensor] = None,
) -> None:
    """Export model to TorchScript format.

    Args:
        model: Model to export
        output_path: Output file path
        example_input: Example input tensor for tracing
    """
    model.eval()

    if example_input is None:
        example_input = torch.randn(1, 3, 224, 224)

    # Trace model
    traced_model = torch.jit.trace(model, example_input)

    # Save
    traced_model.save(output_path)
    print(f"TorchScript model saved to: {output_path}")


def export_onnx(
    model: AnomalyDetector,
    output_path: str,
    example_input: Optional[torch.Tensor] = None,
    opset_version: int = 14,
) -> None:
    """Export model to ONNX format.

    Args:
        model: Model to export
        output_path: Output file path
        example_input: Example input tensor
        opset_version: ONNX opset version
    """
    model.eval()

    if example_input is None:
        example_input = torch.randn(1, 3, 224, 224)

    # Export to ONNX
    torch.onnx.export(
        model,
        example_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"ONNX model saved to: {output_path}")

