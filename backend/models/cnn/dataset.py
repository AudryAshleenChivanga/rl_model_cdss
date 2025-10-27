"""Dataset for CNN training."""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import json


class EndoscopyDataset(Dataset):
    """Dataset for endoscopy frames with lesion labels."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        img_size: int = 224,
    ):
        """Initialize dataset.

        Args:
            data_dir: Root directory containing images/ and labels/
            split: Dataset split ('train', 'val', 'test')
            transform: Image transformations
            img_size: Image size for resizing
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.transform = transform

        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = []

        # Load labels
        labels_path = self.data_dir / "labels" / "labels.csv"
        if labels_path.exists():
            self.labels_df = pd.read_csv(labels_path)
        else:
            # Create dummy labels if not available
            self.labels_df = pd.DataFrame({
                "frame_id": list(range(len(self.metadata))),
                "has_lesion": [0] * len(self.metadata),
            })

        # Split data
        self._create_splits()

        # Get samples for this split
        if split == "train":
            self.samples = self.train_samples
        elif split == "val":
            self.samples = self.val_samples
        elif split == "test":
            self.samples = self.test_samples
        else:
            raise ValueError(f"Unknown split: {split}")

        # Default transform if none provided
        if self.transform is None:
            self.transform = self._get_default_transform()

    def _create_splits(self, train_ratio: float = 0.7, val_ratio: float = 0.15):
        """Create train/val/test splits.

        Args:
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
        """
        n_samples = len(self.labels_df)
        indices = np.arange(n_samples)
        np.random.seed(42)
        np.random.shuffle(indices)

        train_end = int(train_ratio * n_samples)
        val_end = train_end + int(val_ratio * n_samples)

        self.train_samples = indices[:train_end].tolist()
        self.val_samples = indices[train_end:val_end].tolist()
        self.test_samples = indices[val_end:].tolist()

    def _get_default_transform(self) -> transforms.Compose:
        """Get default image transforms.

        Returns:
            Transform pipeline
        """
        if self.split == "train":
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.1,
                    hue=0.05,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get item by index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image, label)
        """
        # Get sample index
        sample_idx = self.samples[idx]

        # Load image
        frame_id = self.labels_df.iloc[sample_idx]["frame_id"]
        img_path = self.data_dir / "images" / f"frame_{frame_id:06d}.jpg"

        if not img_path.exists():
            # Return dummy image if file not found
            image = Image.new("RGB", (self.img_size, self.img_size), color=(128, 128, 128))
        else:
            image = Image.open(img_path).convert("RGB")

        # Get label
        label = int(self.labels_df.iloc[sample_idx]["has_lesion"])

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced dataset.

        Returns:
            Class weights tensor
        """
        labels = self.labels_df.iloc[self.samples]["has_lesion"].values
        class_counts = np.bincount(labels)
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum()
        return torch.tensor(class_weights, dtype=torch.float32)


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """Create data loaders for train/val/test splits.

    Args:
        data_dir: Root data directory
        batch_size: Batch size
        num_workers: Number of worker processes
        img_size: Image size
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        Dictionary of data loaders
    """
    # Create datasets
    train_dataset = EndoscopyDataset(
        data_dir=data_dir,
        split="train",
        img_size=img_size,
    )

    val_dataset = EndoscopyDataset(
        data_dir=data_dir,
        split="val",
        img_size=img_size,
    )

    test_dataset = EndoscopyDataset(
        data_dir=data_dir,
        split="test",
        img_size=img_size,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }

