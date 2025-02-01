#%%
import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformer_dataset import TransformerDataset

# ----------------------
# Positional Encoding
# ----------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Implements the standard positional encoding as described in
        "Attention is All You Need".
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a long enough PEs in advance
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Compute the div term
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, sequence_length, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# ----------------------
# Transformer Model
# ----------------------
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, num_heads, num_layers, num_classes, dropout=0.1):
        """
        A simple Transformer-based classifier.
        
        Args:
            input_size (int): Number of features in the input.
            d_model (int): The dimension of the model (after projection).
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer encoder layers.
            num_classes (int): Number of output classes.
            dropout (float): Dropout rate.
        """
        super(TransformerModel, self).__init__()
        # Project input features to the model dimension
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Create the transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True  # so that input shape is (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final classifier head; here we use the output from the last time step.
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        Returns:
            out: Logits of shape (batch_size, num_classes)
        """
        x = self.input_projection(x)  # (batch_size, sequence_length, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # (batch_size, sequence_length, d_model)
        # Use the last time step's output for classification (similar to the RNN model)
        out = self.fc(x[:, -1, :])
        return out

# ----------------------
# Testing the Transformer with Real Data
# ----------------------
if __name__ == "__main__":

    def run_test(test_func):
        """
        A simple decorator to run a test function and print whether it passed.
        """
        def wrapper():
            try:
                test_func()
                print(f"\033[92m✅ {test_func.__name__} passed\033[0m")
            except AssertionError as e:
                print(f"\033[91m❌ {test_func.__name__} failed: {e}\033[0m")
            except Exception as e:
                print(f"\033[91m❌ {test_func.__name__} failed with an unexpected error: {e}\033[0m")
        wrapper()

    # Set up dataset parameters (adjust paths as needed)
    data_file = os.path.join("data", "2024-11-24T08_57_30_d300sec_w250ms", "training.csv")
    label_file = os.path.join("data", "2024-11-24T08_57_30_d300sec_w250ms", "training_y.csv")
    window_size = 15

    dataset = TransformerDataset(data_file, label_file, window_size)
    num_classes = 5
    batch_size = 1
    # Get one sample to determine input dimensions
    windows, labels = dataset[0]
    sequence_length = windows.shape[0]
    num_features = windows.shape[1]

    # Transformer hyperparameters
    d_model = 64       # The internal model dimension (can be different from num_features)
    num_heads = 4      # Must divide d_model
    num_layers = 2

    # Create the Transformer model
    model = TransformerModel(
        input_size=num_features,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=0.1
    )

    @run_test
    def test_transformer_real_data_batch():
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for batch in dataloader:
            windows, labels = batch
            print("windows.shape:", windows.shape)  # Expected: [batch_size, sequence_length, num_features]
            print("labels.shape:", labels.shape)      # Expected: [batch_size]
            output = model(windows)
            # Check that the output has the expected shape: (batch_size, num_classes)
            assert output.shape == (windows.shape[0], num_classes), (
                f"Model output shape mismatch: {output.shape}, expected ({windows.shape[0]}, {num_classes})"
            )
            break

