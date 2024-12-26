import pandas as pd
import torch
from torch.utils.data import Dataset

class SlidingWindowDataset(Dataset):
    def __init__(self, data_file, label_file, window_size, stride):
        # Load data file with error handling
        try:
            self.data = pd.read_csv(data_file, engine='python')  # Specify the Python engine for parsing
        except Exception as e:
            raise ValueError(f"Error reading data file {data_file}: {e}")
        
        # Load label file with error handling
        try:
            self.labels = pd.read_csv(label_file, engine='python')  # Specify the Python engine for parsing
        except Exception as e:
            raise ValueError(f"Error reading label file {label_file}: {e}")
        
        self.window_size = window_size
        self.stride = stride

        # Calculate total windows
        self.total_windows = (len(self.data) - window_size) // stride + 1

        # Ensure the label indices (gidx) match the data windows
        self.label_mapping = self.labels.set_index("gidx")  # Map gidx to labels
        self.data_gidx = self.data["gidx"].values[: self.total_windows]  # Match gidx in the data

    def __len__(self):
        return self.total_windows

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size

        # Extract the window data
        window = self.data.iloc[start_idx:end_idx].drop(columns=["gidx", "widx", "tidx"]).values

        # Retrieve the corresponding label using gidx
        gidx = self.data_gidx[idx]
        label_row = self.label_mapping.loc[gidx]
        label = label_row["sidx"]

        return torch.tensor(window, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
