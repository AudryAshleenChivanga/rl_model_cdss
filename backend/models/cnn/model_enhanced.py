"""Enhanced CNN model for multi-disease classification in endoscopy images."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Tuple
import numpy as np


class EndoscopyMultiClassCNN(nn.Module):
    """Enhanced ResNet18-based CNN for multi-disease classification.
    
    Detects multiple gastrointestinal conditions:
    - Normal tissue
    - H. pylori gastritis
    - Peptic ulcer
    - Gastric tumor
    - Inflammation
    """
    
    # Disease class labels
    CLASSES = [
        "normal",
        "h_pylori_gastritis",
        "peptic_ulcer",
        "gastric_tumor",
        "inflammation",
    ]
    
    NUM_CLASSES = len(CLASSES)

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.5,
        freeze_backbone: bool = False,
    ):
        """Initialize multi-class CNN model.

        Args:
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout probability
            freeze_backbone: Freeze backbone weights for transfer learning
        """
        super().__init__()

        # Load ResNet18 backbone
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get backbone output features
        in_features = self.backbone.fc.in_features
        
        # Replace final FC layer with multi-head classifier
        self.backbone.fc = nn.Identity()  # Remove original FC
        
        # Multi-head classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, self.NUM_CLASSES),
        )
        
        # Auxiliary attention head for interpretability
        self.attention = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            Tuple of (logits, attention_weights)
            - logits: (B, NUM_CLASSES) class logits
            - attention_weights: (B, 1) attention scores
        """
        # Extract features
        features = self.backbone(x)
        
        # Classification
        logits = self.classifier(features)
        
        # Attention weights
        attention = self.attention(features)
        
        return logits, attention

    def predict(
        self,
        x: torch.Tensor,
        return_probs: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Run inference and return predictions.

        Args:
            x: Input tensor (B, 3, H, W) or single image (3, H, W)
            return_probs: Return class probabilities

        Returns:
            Dictionary with:
            - 'class_ids': Predicted class indices (B,)
            - 'class_names': Predicted class names (B,)
            - 'confidences': Max class probabilities (B,)
            - 'probs': All class probabilities (B, NUM_CLASSES)
            - 'attention': Attention weights (B, 1)
        """
        was_training = self.training
        self.eval()
        
        with torch.no_grad():
            # Handle single image
            if x.ndim == 3:
                x = x.unsqueeze(0)
            
            # Forward pass
            logits, attention = self.forward(x)
            
            # Get probabilities
            probs = F.softmax(logits, dim=1)
            
            # Get predictions
            confidences, class_ids = torch.max(probs, dim=1)
            
            # Convert to numpy
            class_ids_np = class_ids.cpu().numpy()
            confidences_np = confidences.cpu().numpy()
            probs_np = probs.cpu().numpy()
            attention_np = attention.cpu().numpy()
            
            # Get class names
            class_names = [self.CLASSES[idx] for idx in class_ids_np]
            
            results = {
                'class_ids': class_ids_np,
                'class_names': class_names,
                'confidences': confidences_np,
                'attention': attention_np,
            }
            
            if return_probs:
                results['probs'] = probs_np
        
        if was_training:
            self.train()
        
        return results

    def predict_single(
        self,
        image: np.ndarray,
        device: str = 'cpu',
    ) -> Tuple[str, float, Dict[str, float]]:
        """Predict disease class for a single image.

        Args:
            image: RGB image array (H, W, 3) or tensor (3, H, W)
            device: Device to run inference on

        Returns:
            Tuple of (disease_name, confidence, all_probs_dict)
        """
        # Convert numpy to tensor if needed
        if isinstance(image, np.ndarray):
            # Normalize to [0, 1]
            if image.max() > 1.0:
                image = image.astype(np.float32) / 255.0
            
            # Convert HWC to CHW
            if image.shape[-1] == 3:
                image = np.transpose(image, (2, 0, 1))
            
            image = torch.from_numpy(image).float()
        
        # Move to device
        image = image.to(device)
        self.to(device)
        
        # Predict
        results = self.predict(image, return_probs=True)
        
        disease_name = results['class_names'][0]
        confidence = float(results['confidences'][0])
        
        # Build probability dict
        probs_dict = {
            cls: float(prob)
            for cls, prob in zip(self.CLASSES, results['probs'][0])
        }
        
        return disease_name, confidence, probs_dict


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.
    
    From: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(
        self,
        alpha: torch.Tensor = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        """Initialize Focal Loss.

        Args:
            alpha: Class weights (NUM_CLASSES,)
            gamma: Focusing parameter
            reduction: Reduction mode ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Logits (B, NUM_CLASSES)
            targets: Target class indices (B,)

        Returns:
            Loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_class_weights(
    class_counts: Dict[str, int],
    device: str = 'cpu',
) -> torch.Tensor:
    """Compute inverse frequency class weights.

    Args:
        class_counts: Dictionary mapping class names to sample counts
        device: Device to put tensor on

    Returns:
        Class weights tensor (NUM_CLASSES,)
    """
    total = sum(class_counts.values())
    weights = []
    
    for cls in EndoscopyMultiClassCNN.CLASSES:
        count = class_counts.get(cls, 1)
        weight = total / (len(class_counts) * count)
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32, device=device)


def create_model(
    pretrained: bool = True,
    dropout: float = 0.5,
    freeze_backbone: bool = False,
    device: str = 'cpu',
) -> EndoscopyMultiClassCNN:
    """Factory function to create and initialize model.

    Args:
        pretrained: Use ImageNet pretrained weights
        dropout: Dropout probability
        freeze_backbone: Freeze backbone for transfer learning
        device: Device to put model on

    Returns:
        Initialized model
    """
    model = EndoscopyMultiClassCNN(
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )
    model.to(device)
    return model


if __name__ == "__main__":
    # Test model
    model = create_model()
    
    # Test forward pass
    batch = torch.randn(4, 3, 224, 224)
    logits, attention = model(batch)
    
    print(f"Logits shape: {logits.shape}")  # (4, 5)
    print(f"Attention shape: {attention.shape}")  # (4, 1)
    
    # Test prediction
    results = model.predict(batch)
    print(f"\nPredictions:")
    for i, (name, conf) in enumerate(zip(results['class_names'], results['confidences'])):
        print(f"  Sample {i}: {name} ({conf:.3f})")

