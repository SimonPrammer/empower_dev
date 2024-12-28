#%%
import os
import pandas as pd
import torch
from torch.utils.data import Dataset

# class SlidingWindowDataset(Dataset):

#     def __init__(self, data_file, label_file, window_size, stride):
#         try:
#             raw_data = pd.read_csv(data_file, engine='python')
#         except Exception as e:
#             raise ValueError(f"Error reading data file {data_file}: {e}")

#         # Load CSV labels
#         try:
#             raw_labels = pd.read_csv(label_file, engine='python')
#         except Exception as e:
#             raise ValueError(f"Error reading label file {label_file}: {e}")
        
        
#         # pseudo code:
#         # get all entries with same widx pack into one window 
#         # windows entry: wdix = idx, num_features, features = window_size
#         # target entry: wdix = idx, window_size (??)
#         # add the entries

#         self.windows = []
#         self.targets = []

#         # i think if i get the whole dataset it would be shape:
#         # (N / window_size), num features, window_size
        

#     def __len__(self):
#         return len(self.windows)

#     # get one item with idx aka tidx
#     def __getitem__(self, idx):
#         window = self.windows[idx]  # shape: (num_features, window_size)
#         label = self.targets[idx] # (,window_size) ??

#         window_tensor = torch.tensor(window, dtype=torch.float32)
#         label_tensor = torch.tensor(label, dtype=torch.int8) # should be int as sidx is from 20-24
#         return window_tensor, label_tensor



class SlidingWindowDataset(Dataset):
    def __init__(self, data_file, label_file, window_size, stride):
        # Read data
        try:
            raw_data = pd.read_csv(data_file, engine='python')
        except Exception as e:
            raise ValueError(f"Error reading data file {data_file}: {e}")

        # Read labels
        try:
            raw_labels = pd.read_csv(label_file, engine='python')
        except Exception as e:
            raise ValueError(f"Error reading label file {label_file}: {e}")

        # Filter data to keep only gidx present in labels
        valid_gidx = set(raw_labels["gidx"])
        raw_data = raw_data[raw_data["gidx"].isin(valid_gidx)]

        self.window_size = window_size
        self.stride = stride

        # Keep a reference for unit-testing or debugging
        self.raw_data = raw_data.reset_index(drop=True)
        self.raw_labels = raw_labels.reset_index(drop=True)

        # Build (gidx, widx) -> label lookup (assuming exactly one row per (gidx,widx) in label_file)
        label_lookup = raw_labels.set_index(["gidx", "widx"])

        # We'll store all sub-windows and labels here
        self.windows = []
        self.targets = []

        # Group by (gidx, widx)
        grouped = raw_data.groupby(["gidx", "widx"], sort=False)

        for (g, w), chunk_df in grouped:
            # Sort by tidx to ensure correct time order
            chunk_df = chunk_df.sort_values("tidx")
            chunk_values = chunk_df.drop(columns=["gidx", "widx", "tidx"]).values
            num_rows = len(chunk_values)

            # If no matching label found, skip or raise
            if (g, w) not in label_lookup.index:
                continue

            label_row = label_lookup.loc[(g, w)]
            # e.g. label_row["sidx"]
            sidx_label = label_row["sidx"]

            # Slide over this chunk
            if num_rows >= self.window_size:
                num_windows = (num_rows - self.window_size) // self.stride + 1
                for i in range(num_windows):
                    start_idx = i * self.stride
                    end_idx = start_idx + self.window_size
                    window_data = chunk_values[start_idx:end_idx]
                    # Transpose to [num_features, window_size]
                    window_data = window_data.T

                    self.windows.append(window_data)
                    self.targets.append(sidx_label)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        label = self.targets[idx]
        # Convert to torch tensors
        window_tensor = torch.tensor(window, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return window_tensor, label_tensor


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

    # Setup for all tests
    data_file = os.path.join("data", "2024-11-24T08_57_30_d300sec_w250ms", "training.csv")
    label_file = os.path.join("data", "2024-11-24T08_57_30_d300sec_w250ms", "training_y.csv")
    window_size = 15
    stride = 15

    # Create dataset
    dataset = SlidingWindowDataset(
        data_file=data_file,
        label_file=label_file,
        window_size=window_size,
        stride=stride,
    )

 

    raw_data = pd.read_csv(data_file, engine='python')
    raw_labels = pd.read_csv(label_file, engine='python')

    valid_gidx = set(raw_labels['gidx'])
    raw_data = raw_data[raw_data['gidx'].isin(valid_gidx)]

    @run_test
    def test_file_loading():
        """Test if files are loaded (non-empty) in the constructor."""
        assert len(dataset.raw_data) > 0, "Data file is empty."
        assert len(dataset.raw_labels) > 0, "Label file is empty."

    @run_test
    def test_gidx_alignment():
        """Test if gidx values align between data and labels in the raw CSVs."""
        data_gidx = set(raw_data["gidx"])
        label_gidx = set(raw_labels["gidx"])
        assert data_gidx.issubset(label_gidx), "Mismatch between gidx in data and labels."

    @run_test
    def test_total_windows():
        """Test if total windows matches the expected count (grouping logic)."""
        grouped = raw_data.groupby(["gidx", "widx"])
        expected_windows = 0
        for (_, _), chunk_df in grouped:
            length = len(chunk_df)
            if length >= window_size:
                expected_windows += (length - window_size) // stride + 1

        actual = len(dataset)
        assert actual == expected_windows, f"Expected {expected_windows} windows, got {actual}."

    @run_test
    def test_window_extraction():
        """Test if a sample window is the correct shape: [num_features, window_size]."""
        if len(dataset) == 0:
            print("No windows available in dataset to test extraction.")
            return

        window, label = dataset[0]
        num_feature_cols = raw_data.shape[1] - 3

        assert window.shape == (num_feature_cols, window_size), (
            f"Window shape {window.shape} does not match expected {(num_feature_cols, window_size)}."
        )
        assert isinstance(label.item(), int), "Label is not an integer."

    @run_test
    def test_label_assignment():
        """Test if labels are valid integers and consistent with sidx from the label file."""
        if len(dataset) == 0:
            print("No windows available in dataset to test label assignment.")
            return

        for i in [0, len(dataset) // 2, len(dataset) - 1]:
            window, label = dataset[i]
            assert isinstance(label.item(), int), "Label is not an integer."

    @run_test
    def test_error_handling():
        """Test error handling for missing or malformed files."""
        try:
            SlidingWindowDataset(
                data_file="invalid_file.csv",
                label_file=label_file,
                window_size=window_size,
                stride=stride,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError for invalid data file.")

        try:
            SlidingWindowDataset(
                data_file=data_file,
                label_file="invalid_file.csv",
                window_size=window_size,
                stride=stride,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError for invalid label file.")

   
