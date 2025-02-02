#%%

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from cnn2d_dataset import Cnn2dDataset

class CNN2D(nn.Module):
    def __init__(self, num_features, sequence_length, num_classes):
        super(CNN2D, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Flatten(),
            nn.Linear(128 * (num_features // 4) * (sequence_length // 4), 128),
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

    dataset = Cnn2dDataset(data_file, label_file, window_size)

    # config
    batch_size = 1
    windows,labels = dataset[0] # Returns (batch, 1, features, time), label
   
    channels = windows.shape[1]  # Should be 1
    num_features = windows.shape[2] if len(windows.shape) == 4 else windows.shape[1]
    sequence_length = window_size
    num_classes = 5  # Known from dataset (20-24 mapped to 0-4)


    # init model
    model = CNN2D(num_features=num_features, sequence_length=sequence_length, num_classes=num_classes)

    
    @run_test
    def test_forward_pass():
        """Test forward pass with correct shapes"""
        batch_size_test = 2
        dummy_input = torch.randn(batch_size_test, 1, num_features, sequence_length)
        print("Test input shape:", dummy_input.shape)
        
        output = model(dummy_input)
        print("Test output shape:", output.shape)
        
        expected_shape = (batch_size_test, num_classes)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    @run_test
    def test_real_data_batch():
        """Test if the model works with a batch of real dataset input."""
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