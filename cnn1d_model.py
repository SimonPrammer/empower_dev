#%%

import os
import torch
import torch.nn as nn

from sliding_window_dataset import SlidingWindowDataset

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

 
    #load dataset
    data_file = os.path.join("data", "2024-11-24T08_57_30_d300sec_w250ms", "training.csv")
    label_file = os.path.join("data", "2024-11-24T08_57_30_d300sec_w250ms", "training_y.csv")
    window_size = 15

    dataset = SlidingWindowDataset(data_file, label_file, window_size)

    # Configuration for CNN1D
    batch_size = 1
    num_features = dataset.windows.shape[1]  # Number of features per timestep
    sequence_length = dataset.windows.shape[2]  # Sequence length (window size)
    num_classes = len(torch.unique(dataset.targets))  # Unique class labels

    print("num_features:",num_features)
    print("sequence_length:",sequence_length)

    # Initialize the model
    model = CNN1D(num_features=num_features, sequence_length=sequence_length, num_classes=num_classes)

    def test_output_shape():
        model = CNN1D(num_features=72, sequence_length=15)
        x = torch.randn(32, 72, 15)  # batch_size, features, sequence
        out = model(x)
        assert out.shape == (32, 5), f"Expected output shape (32, 5), got {out.shape}"


    @run_test
    def test_dataset_input_shape():
        """Test if the model accepts input directly from the dataset."""
        input_tensor, _ = dataset[0]  # Simulate a single window
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
        output = model(input_tensor)
        assert output.shape == (1, num_classes), (
            f"Model did not produce expected output shape: {output.shape}, expected (1, {num_classes})"
        )

    @run_test
    def test_output_range():
        """Test if model outputs valid class predictions"""
        x = torch.randn(32, 72, 15)
        out = model(x)
        pred = torch.argmax(out, dim=1)
        assert torch.all((pred >= 0) & (pred < 5)), "Predictions must be between 0 and 4"

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

    # Test using real data from the dataset
    @run_test
    def test_real_data_batch():
        """Test if the model works with a batch of real dataset input."""
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for batch in dataloader:
            windows, labels = batch
            print("windows.shape",windows.shape)
            print("labels.shape",labels.shape)
            output = model(windows)
            assert output.shape == (windows.shape[0], num_classes), (
                f"Model output shape mismatch: {output.shape}, expected ({windows.shape[0]}, {num_classes})"
            )
            break  # Test only the first batch

        print("output.shape:",output.shape)
        print("output[0]:",output[0])

