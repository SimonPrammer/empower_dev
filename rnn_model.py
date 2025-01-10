#%%
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from rnn_dataset import RNNDataset


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

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

    dataset = RNNDataset(data_file, label_file, window_size)

    num_classes = 5
    batch_size = 1
    windows, labels = dataset[0]
    sequence_length = windows.shape[0]
    num_features = windows.shape[1]

    model = RNNModel(input_size=num_features, hidden_size=128, num_layers=2, num_classes=num_classes)

    @run_test
    def test_real_data_batch():
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for batch in dataloader:
            windows, labels = batch
            print("windows.shape:", windows.shape)  # Expected: [batch_size, sequence_length, num_features]
            print("labels.shape:", labels.shape)    # Expected: [batch_size]
            output = model(windows)
            assert output.shape == (windows.shape[0], num_classes), (
                f"Model output shape mismatch: {output.shape}, expected ({windows.shape[0]}, {num_classes})"
            )
            break