import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from linear_transformer import Transformer_F_w_embed
from analysis_utils import AttentionVisualizer

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a simple transformer model
d = 9  # dimension
n_layer = 2  # number of layers
n_head = 2  # number of heads
var = 0.0001  # variance for weight initialization
hidden_dim = d * 4  # hidden dimension

# Initialize model
model = Transformer_F_w_embed(n_layer, n_head, d, var, hidden_dim=hidden_dim, device=device)
model = model.to(device)
print("Model created")

# Create visualizer
attention_visualizer = AttentionVisualizer(model)
print("Attention visualizer created")

# Create random input data
batch_size = 4
seq_len = 10
input_dim = d + 1  # input dimension (plus the last column)

# Generate random input
Z = torch.randn(batch_size, seq_len, input_dim, device=device)
print(f"Input shape: {Z.shape}")

# Forward pass and get attention weights
print("Performing forward pass and collecting attention maps...")
attention_visualizer.get_attention_maps(model, Z)

# Generate attention heatmaps
print("Generating attention maps...")
os.makedirs('attention_test', exist_ok=True)
attention_visualizer.plot_attention_maps('attention_test/attention_map.png')

# Check if attention weights were captured
attn_maps = attention_visualizer.get_attn_map()
print(f"Number of attention weights captured: {len(attn_maps)}")
for key, value in attn_maps.items():
    print(f"Key: {key}, Shape: {value.shape}")

print("Done!") 