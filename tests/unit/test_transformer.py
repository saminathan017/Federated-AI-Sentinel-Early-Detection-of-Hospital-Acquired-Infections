"""Unit tests for transformer model."""

import pytest
import torch

from src.modeling.temporal.transformer_model import TemporalTransformer, create_model, count_parameters


def test_model_creation():
    """Test model initialization."""
    model = create_model(input_dim=20, device="cpu")
    
    assert isinstance(model, TemporalTransformer)
    assert model.input_dim == 20
    assert model.d_model == 64


def test_model_forward_pass():
    """Test forward pass with random input."""
    batch_size = 4
    seq_len = 10
    input_dim = 20
    
    model = create_model(input_dim=input_dim, device="cpu")
    
    # Create random input (seq_len, batch, input_dim)
    x = torch.randn(seq_len, batch_size, input_dim)
    
    # Forward pass
    logits = model(x)
    
    assert logits.shape == (batch_size,)
    
    # Check that logits are finite
    assert torch.isfinite(logits).all()


def test_model_predict_proba():
    """Test probability prediction."""
    model = create_model(input_dim=20, device="cpu")
    
    x = torch.randn(10, 1, 20)
    
    probs = model.predict_proba(x)
    
    assert probs.shape == (1,)
    assert 0 <= probs.item() <= 1


def test_parameter_count():
    """Test parameter counting."""
    model = create_model(input_dim=20, device="cpu")
    
    num_params = count_parameters(model)
    
    assert num_params > 0
    assert isinstance(num_params, int)


def test_model_eval_mode():
    """Test model switches to eval mode."""
    model = create_model(input_dim=20, device="cpu")
    
    model.eval()
    
    assert not model.training
    
    # Predictions should be deterministic in eval mode
    x = torch.randn(10, 2, 20)
    
    pred1 = model(x)
    pred2 = model(x)
    
    assert torch.allclose(pred1, pred2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

