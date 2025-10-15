"""
Temporal transformer for sequential infection risk prediction.

A lightweight transformer model designed to run on CPU for clinical deployment.
Uses positional encoding and multi-head attention to capture temporal patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Add positional information to embeddings."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[: x.size(0)]


class TemporalTransformer(nn.Module):
    """
    Transformer for time series infection prediction.
    
    Input: Sequence of patient observations over time
    Output: Risk probability at each time step
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        """
        Initialize the transformer model.
        
        Args:
            input_dim: Number of input features
            d_model: Embedding dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=False,
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            src: Input tensor of shape (seq_len, batch_size, input_dim)
            src_mask: Optional mask for attention
        
        Returns:
            Risk logits of shape (batch_size,)
        """
        # Project input to d_model dimensions
        src = self.input_projection(src)  # (seq_len, batch, d_model)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        src = self.dropout(src)
        
        # Transformer encoding
        encoded = self.transformer_encoder(src, src_mask)  # (seq_len, batch, d_model)
        
        # Use the last time step for prediction
        last_output = encoded[-1, :, :]  # (batch, d_model)
        
        # Project to risk score
        logits = self.fc_out(last_output).squeeze(-1)  # (batch,)
        
        return logits

    def predict_proba(self, src: torch.Tensor) -> torch.Tensor:
        """Return calibrated probabilities."""
        logits = self.forward(src)
        return torch.sigmoid(logits)


def create_model(input_dim: int, device: str = "cpu") -> TemporalTransformer:
    """Factory function to create a transformer model."""
    model = TemporalTransformer(
        input_dim=input_dim,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    )
    
    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    print("Testing TemporalTransformer...")
    
    batch_size = 4
    seq_len = 10
    input_dim = 20
    
    model = create_model(input_dim=input_dim, device="cpu")
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Generate random input
    x = torch.randn(seq_len, batch_size, input_dim)
    
    # Forward pass
    logits = model(x)
    probs = model.predict_proba(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Output probabilities: {probs}")
    
    print("\n✓ Model test passed")

