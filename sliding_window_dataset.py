import pandas as pd
import torch
from torch.utils.data import Dataset

class SlidingWindowDataset(Dataset):
    def __init__(self, data_file, label_file, window_size, stride):
        # Read data
        try:
            data = pd.read_csv(data_file, engine='python')
        except Exception as e:
            raise ValueError(f"Error reading data file {data_file}: {e}")

        # Read labels
        try:
            labels = pd.read_csv(label_file, engine='python')
        except Exception as e:
            raise ValueError(f"Error reading label file {label_file}: {e}")

        # Optional: Filter data to keep only gidx present in labels
        valid_gidx = set(labels["gidx"])
        data = data[data["gidx"].isin(valid_gidx)]

        self.window_size = window_size
        self.stride = stride

        # We'll accumulate all sub-windows and their labels into lists
        self.windows = []
        self.targets = []

        # Make sure (gidx, widx) is unique in labels (or handle duplicates)
        # We'll create a lookup table: (gidx, widx) -> row in labels
        # If your label CSV has exactly one row per (gidx, widx),
        # then set_index is enough:
        label_lookup = labels.set_index(["gidx", "widx"])

        # Group data by (gidx, widx) so we can apply sliding windows chunk by chunk
        grouped = data.groupby(["gidx", "widx"], sort=False)

        for (g, w), chunk_df in grouped:
            # Sort each chunk by tidx to ensure correct time order
            chunk_df = chunk_df.sort_values("tidx")

            # Drop identifying columns to leave only feature columns
            chunk_values = chunk_df.drop(columns=["gidx", "widx", "tidx"]).values
            num_rows = len(chunk_values)

            # Check if we have a label row for (gidx, widx)
            if (g, w) not in label_lookup.index:
                # If no matching label found, skip (or raise an error)
                continue

            # Get the single label from that row
            label_row = label_lookup.loc[(g, w)]
            # If there's only 1 row per (g,w), label_row["sidx"] is scalar
            sidx_label = label_row["sidx"]

            # Compute how many sub-windows can be extracted from this chunk
            num_windows = (num_rows - window_size) // stride + 1
            if num_windows < 1:
                # No sub-window fits for this chunk
                continue

            # Slide over the chunk
            for i in range(num_windows):
                start_idx = i * stride
                end_idx = start_idx + window_size
                window_data = chunk_values[start_idx:end_idx]

                # Transpose to [num_features, window_size]
                window_data = window_data.T

                # Store the window and label
                self.windows.append(window_data)
                self.targets.append(sidx_label)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        label = self.targets[idx]
        # Convert to tensors
        window_tensor = torch.tensor(window, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return window_tensor, label_tensor
