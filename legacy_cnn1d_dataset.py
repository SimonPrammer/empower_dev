
#%%
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np


class Cnn1dDataset(Dataset):
    def __init__(self, data_file, label_file, window_size, use_relative=False):
        self.window_size = window_size
        self.windows = []
        self.targets = []
        
        #load data
        self.raw_data = pd.read_csv(data_file)
        self.raw_labels = pd.read_csv(label_file)
        
        # print(f"Number of label entries: {len(self.raw_labels)}")
        
        # create data lookup by grouping
        data_lookup = self.raw_data.groupby(['gidx', 'widx'])
        
        # iterate over labels
        for idx, label_row in self.raw_labels.iterrows():
            g = label_row['gidx']
            w = label_row['widx']
            
            try:
                group_df = data_lookup.get_group((g, w))
                # Get data and label
                data = group_df.drop(columns=['gidx', 'widx', 'tidx']).values
                label = label_row['sidx']
                
                # Take one window from start of sequence 
                if len(data) >= window_size:
                    window = data[:window_size].T
                    self.windows.append(window)
                    self.targets.append(label)
                    
            except KeyError:
                print(f"Warning: No data found for gidx={g}, widx={w}")
                continue

        #convert to tensors
        self.windows = torch.FloatTensor(np.array(self.windows))
        self.targets = torch.LongTensor(self.targets)
        self.original_targets = self.targets.clone()

        # normalize feature-wise
        self.normalize_windows()

        assert len(self.targets) == len(self.windows), "Number of targets and windows don't match"

    def normalize_windows(self):
        """Normalize the windows feature-wise."""
        mean = self.windows.mean(dim=(0, 2), keepdim=True)  # Mean over batches and time
        std = self.windows.std(dim=(0, 2), keepdim=True)    # Std over batches and time
        self.windows = (self.windows - mean) / (std + 1e-6)  # Normalize and avoid division by zero
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return self.windows[idx], self.targets[idx] - 20  # Transform 20-24 to 0-4 otherwise pytorch is screaming

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

    # Setup for all tests for 250ms
    data_file = os.path.join("data", "2024-11-24T08_57_30_d300sec_w250ms", "training.csv")
    label_file = os.path.join("data", "2024-11-24T08_57_30_d300sec_w250ms", "training_y.csv")
    window_size = 15

    dataset = Cnn1dDataset(data_file, label_file, window_size)

    entry_index = 0 
    window, label = dataset[entry_index]

    print(f"Window (features) Shape: {window.shape}")
    print(f"Window (features) Content:\n{window}")
    print(f"Label: {label.item()}")
    print("dataset len",len(dataset))

    raw_data = pd.read_csv(data_file, engine='python')
    raw_labels = pd.read_csv(label_file, engine='python')

    # valid_gidx = set(raw_labels['gidx'])
    # raw_data = raw_data[raw_data['gidx'].isin(valid_gidx)]

    #check that first window is the same as the first window in the dataset
    #i normalized so wont work anymore
    # @run_test  
    # def test_first_window():
    #     """Test if the first window is the same as the first window in the dataset."""
        
    #     data_lookup = raw_data.groupby(['gidx', 'widx'])
    #     g = raw_labels['gidx'][0]
    #     w = raw_labels['widx'][0]
    #     group_df = data_lookup.get_group((g, w))
    #     data = group_df.drop(columns=['gidx', 'widx', 'tidx']).values
    #     window = data[:window_size].T
    #     dataset_window, _ = dataset[0]
    #     assert torch.allclose(torch.FloatTensor(window), dataset_window), (
    #         f"First window does not match: {dataset_window} instead of {window}"
    #     )

    #hard check first label should be int 20 aka mapped to 0
    @run_test
    def test_first_label():
        """Test if the first label is correctly transformed."""
        _, label = dataset[0]
        assert label.item() == 0, f"First label should be 0 (transformed from 20), got {label.item()}"
    
    #test last window and label
    #i normalized so wont work anymore
    # @run_test
    # def test_last_window():
    #     """Test if the last window is the same as the last window in the dataset."""
    #     data_lookup = raw_data.groupby(['gidx', 'widx'])
    #     g = raw_labels['gidx'].iloc[-1]
    #     w = raw_labels['widx'].iloc[-1]
    #     group_df = data_lookup.get_group((g, w))
    #     data = group_df.drop(columns=['gidx', 'widx', 'tidx']).values
    #     window = data[:window_size].T
    #     dataset_window, _ = dataset[-1]
    #     assert torch.allclose(torch.FloatTensor(window), dataset_window), (
    #         f"Last window does not match: {dataset_window} instead of {window}"
    #     )

    #hard check last label should be int 24 aka mapped to 4
    @run_test
    def test_last_label():
        """Test if the last label is correctly transformed."""
        _, label = dataset[-1]
        assert label.item() == 4, f"Last label should be 4 (transformed from 24), got {label.item()}"

 