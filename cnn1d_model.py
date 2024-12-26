#%%

import torch
import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self, num_features, sequence_length, num_classes):
        super(CNN1D, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(128 * (sequence_length // 4), 128),  # Adjust based on pooling layers
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)



import torch
import os
from cnn1d_model import CNN1D
from legacy_data_loader import data_loader

if __name__ == "__main__":

    def run_test(test_func):
        """
        Decorator to run a test function, catching assertions and printing
        a green checkmark for success or error message for failure.
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

    # Test configuration
    num_features = 72
    sequence_length = 30
    num_classes = 5
    batch_size = 32
    model = CNN1D(num_features=num_features, sequence_length=sequence_length, num_classes=num_classes)

    @run_test
    def test_output_shape():
        """Test if the model produces the correct output shape."""
        input_tensor = torch.randn(batch_size, num_features, sequence_length)
        output = model(input_tensor)
        assert output.shape == (batch_size, num_classes), (
            f"Unexpected output shape: {output.shape}, expected ({batch_size}, {num_classes})"
        )

    @run_test
    def test_no_nan_inf():
        """Test if the model output contains NaN or Inf values."""
        input_tensor = torch.randn(batch_size, num_features, sequence_length)
        output = model(input_tensor)
        assert not torch.isnan(output).any(), "Output contains NaN values."
        assert not torch.isinf(output).any(), "Output contains Inf values."

    @run_test
    def test_varying_batch_sizes():
        """Test if the model handles varying batch sizes."""
        for b_size in [1, 16, 64]:
            input_tensor = torch.randn(b_size, num_features, sequence_length)
            output = model(input_tensor)
            assert output.shape == (b_size, num_classes), (
                f"Unexpected output shape for batch size {b_size}: {output.shape}, expected ({b_size}, {num_classes})"
            )

    @run_test
    def test_dataset_compatibility():
        """Test if the model is compatible with simulated dataset input."""
        input_tensor = torch.randn(1, num_features, sequence_length)  # Simulating one window from dataset
        output = model(input_tensor)
        assert output.shape == (1, num_classes), (
            f"Dataset simulation failed: {output.shape} instead of (1, {num_classes})"
        )

    @run_test
    def test_with_real_data():
        """Test if the model runs with real data from training.csv."""
        file_path = os.path.join("data", "2024-11-24T10_13_28_d300sec_w1000ms", "training.csv")
        data = data_loader(file_path, num_features, sequence_length)

        # Take a batch from the real data
        b_size = min(batch_size, data.size(0))  # Ensure batch size is valid
        input_tensor = data[:b_size]

        # Pass through the model
        output = model(input_tensor)
        assert output.shape == (b_size, num_classes), (
            f"Unexpected output shape: {output.shape}, expected ({b_size}, {num_classes})"
        )


