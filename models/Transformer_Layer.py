import torch
import torch.nn as nn
import torch.nn.functional as F


def get_torch_trans(heads=8, layers=1, channels=64):
    """
    Creates a Transformer encoder module to process MTS timestamps/features as sequences.

    Parameters:
    - heads (int): Number of attention heads.
    - layers (int): Number of encoder layers.
    - channels (int): Dimensionality of the model (d_model in Transformer terminology).

    Returns:
    - nn.TransformerEncoder: A Transformer encoder object.
    """
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers, enable_nested_tensor=False)
