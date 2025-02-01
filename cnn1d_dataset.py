#%%
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np


class Cnn1dDataset(Dataset):
    def __init__(self, data_file, label_file, window_size, use_relative=False):
        """
        Args:
            data_file (str): Path to the CSV file containing sensor data.
            label_file (str): Path to the CSV file containing labels.
            window_size (int): The number of time steps per window.
            use_relative (bool): If True, compute the relative (delta) values
                                 over time from the normalized absolute values.
        """
        self.window_size = window_size
        self.use_relative = use_relative
        raw_windows = []  # to store the raw (absolute) sensor windows
        self.targets = []
        
        # Load data and labels
        self.raw_data = pd.read_csv(data_file)
        self.raw_labels = pd.read_csv(label_file)
        
        # Create a lookup by grouping data by ('gidx', 'widx')
        data_lookup = self.raw_data.groupby(['gidx', 'widx'])
        
        # Iterate over each label row
        for idx, label_row in self.raw_labels.iterrows():
            g = label_row['gidx']
            w = label_row['widx']
            
            try:
                group_df = data_lookup.get_group((g, w))
                # Drop non-sensor columns and get sensor data
                data = group_df.drop(columns=['gidx', 'widx', 'tidx']).values
                label = label_row['sidx']
                
                # Only process if there is enough data for the window
                if len(data) >= window_size:
                    # Transpose so that the shape becomes (features, window_size)
                    window = data[:window_size].T
                    raw_windows.append(window)
                    self.targets.append(label)
                    
            except KeyError:
                print(f"Warning: No data found for gidx={g}, widx={w}")
                continue

        # Convert lists to tensors (raw windows, still absolute values)
        self.windows = torch.FloatTensor(np.array(raw_windows))
        self.targets = torch.LongTensor(self.targets)
        self.original_targets = self.targets.clone()

        # normalize
        self.normalize_windows()

        # compute relative AFTER normalizing
        if self.use_relative:
            # Compute differences along the time axis.
            # For each sample, for each feature: diff = current - previous.
            windows_diff = self.windows[:, :, 1:] - self.windows[:, :, :-1]
            # Pad the first time step with zeros to keep the same window size.
            pad = torch.zeros((self.windows.shape[0], self.windows.shape[1], 1), dtype=self.windows.dtype)
            self.windows = torch.cat([pad, windows_diff], dim=2)
        
        # Check consistency.
        assert len(self.targets) == len(self.windows), "Number of targets and windows don't match"

    def normalize_windows(self):
        """Normalize the windows feature-wise."""
        # Mean and std computed over all samples and time steps for each feature.
        mean = self.windows.mean(dim=(0, 2), keepdim=True)
        std = self.windows.std(dim=(0, 2), keepdim=True)
        self.windows = (self.windows - mean) / (std + 1e-6)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        # Adjust the label so that classes 20-24 map to 0-4
        return self.windows[idx], self.targets[idx] - 20  


if __name__ == "__main__":
    def run_test(test_func):
        """
        Decorator to register and run a test function, catching assertions and printing
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

    # Setup paths and parameters
    data_file = os.path.join("data", "2024-11-24T08_57_30_d300sec_w250ms", "training.csv")
    label_file = os.path.join("data", "2024-11-24T08_57_30_d300sec_w250ms", "training_y.csv")
    window_size = 15

    # compare relative vs absolute
    dataset_absolute = Cnn1dDataset(data_file, label_file, window_size, use_relative=False)
    dataset_relative = Cnn1dDataset(data_file, label_file, window_size, use_relative=True)

    entry_index = 0 
    window, label = dataset_relative[entry_index]

    print(f"Window (features) Shape: {window.shape}")
    print(f"Window (features) Content:\n{window}")
    print(f"Label: {label.item()}")
    print("dataset_relative len:", len(dataset_relative))

    # check label transformation
    @run_test
    def test_first_label():
        """Test if the first label is correctly transformed."""
        _, label = dataset_relative[0]
        assert label.item() == 0, f"First label should be 0 (transformed from 20), got {label.item()}"
    
    @run_test
    def test_last_label():
        """Test if the last label is correctly transformed."""
        _, label = dataset_relative[-1]
        assert label.item() == 4, f"Last label should be 4 (transformed from 24), got {label.item()}"

    @run_test
    def test_relative_delta_computation():
        """
        For each sample, test that for every time step > 0, the relative value equals the difference
        between the normalized absolute sensor values of the current and previous time steps.
        """
        # check first sample.
        window_abs, _ = dataset_absolute[0]
        window_rel, _ = dataset_relative[0]
        for i in range(1, window_size):
            expected_delta = window_abs[:, i] - window_abs[:, i - 1]
            np.testing.assert_allclose(window_rel[:, i].numpy(), expected_delta.numpy(), atol=1e-5,
                                       err_msg=f"Delta at time step {i} is incorrect.")

    @run_test
    def test_all_sample_shapes():
        """Test that all samples have consistent shapes in both datasets."""
        for i in range(len(dataset_absolute)):
            window_abs, _ = dataset_absolute[i]
            window_rel, _ = dataset_relative[i]
            assert window_abs.shape == window_rel.shape, f"Shape mismatch in sample {i}: {window_abs.shape} vs {window_rel.shape}"

