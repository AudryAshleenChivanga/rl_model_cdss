"""Tests for CNN model."""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.models.cnn.model import AnomalyDetector


class TestCNN:
    """Test suite for CNN model."""

    @pytest.fixture
    def model(self):
        """Create test model."""
        model = AnomalyDetector(num_classes=2, pretrained=False)
        model.eval()
        return model

    def test_model_creation(self, model):
        """Test model can be created."""
        assert model is not None

    def test_forward_pass(self, model):
        """Test forward pass."""
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, 2)

    def test_predict_proba(self, model):
        """Test probability prediction."""
        x = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            probs = model.predict_proba(x)
        
        assert probs.shape == (1, 2)
        assert torch.allclose(probs.sum(dim=1), torch.ones(1), atol=1e-5)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_predict(self, model):
        """Test class prediction."""
        x = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            predictions = model.predict(x)
        
        assert predictions.shape == (2,)
        assert predictions.dtype == torch.int64
        assert (predictions >= 0).all() and (predictions <= 1).all()

    def test_batch_processing(self, model):
        """Test different batch sizes."""
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 3, 224, 224)
            
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (batch_size, 2)

    def test_different_input_sizes(self, model):
        """Test model with different input sizes."""
        # Model should work with different spatial dimensions
        for size in [224, 256, 320]:
            x = torch.randn(1, 3, size, size)
            
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (1, 2)


def test_cnn_smoke():
    """Quick smoke test."""
    model = AnomalyDetector(num_classes=2, pretrained=False)
    model.eval()
    
    # Forward pass
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    
    assert output.shape == (2, 2)
    print("CNN smoke test passed!")


if __name__ == "__main__":
    test_cnn_smoke()

