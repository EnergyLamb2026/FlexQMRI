"""Standard Transformer model for fixed-length input regression."""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        Returns:
            x + positional encoding
        """
        return x + self.pe[:, :x.size(1), :]


class Transformer(nn.Module):
    """
    Standard Transformer model for regression on fixed-length input sequences.
    
    Uses PyTorch's nn.TransformerEncoder for the core architecture.
    Takes fixed-length input and predicts output parameters.
    """
    
    def __init__(self, 
                 input_channels,
                 output_channels,
                 max_length,
                 d_model=64,
                 n_head=4,
                 n_layers=2,
                 d_inner=None,
                 activation='relu',
                 dropout=0.1):
        """
        Args:
            input_channels (int): Number of input features
            output_channels (int): Number of output parameters
            max_length (int): Maximum sequence length
            d_model (int): Dimension of model (default: 64)
            n_head (int): Number of attention heads (default: 4)
            n_layers (int): Number of transformer blocks (default: 2)
            d_inner (int): Dimension of feed-forward inner layer (default: 4*d_model)
            activation (str): Activation function name (default: 'relu')
            dropout (float): Dropout rate (default: 0.1)
        """
        super(Transformer, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.max_length = max_length
        self.d_model = d_model
        self.n_head = n_head
        self.n_layers = n_layers
        self.dropout = dropout
        
        # Set d_inner if not provided
        if d_inner is None:
            d_inner = int(4 * d_model)
        self.d_inner = d_inner
        
        # Input projection: embed input features to d_model dimension
        self.input_projection = nn.Linear(input_channels, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_length)
        
        # PyTorch's TransformerEncoderLayer and TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_inner,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Output projection: aggregate sequence and project to output
        self.output_projection = nn.Linear(d_model, output_channels)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, 2) where channel 0 is the
                x-axis (TEs or b-values) and channel 1 is the signal. Only the
                signal channel is used; ordering/position is encoded by the
                sinusoidal positional encoding.

        Returns:
            Output tensor of shape (batch, output_channels)
        """
        # Use the signal channel only (channel 1); keep the trailing feature dim
        x = x[:, :, 1:2]  # (batch, seq_len, 1)

        # Project input to d_model dimension
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.positional_encoding(x)  # (batch, seq_len, d_model)
        
        # Apply dropout
        x = self.dropout_layer(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Global average pooling: aggregate sequence dimension
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Project to output dimension
        output = self.output_projection(x)  # (batch, output_channels)
        
        return output
