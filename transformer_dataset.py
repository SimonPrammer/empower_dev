import os
from cnn1d_dataset import Cnn1dDataset


class TransformerDataset(Cnn1dDataset):
    def __init__(self, data_file, label_file, window_size, normalize_after=False, use_relative=False, normalize_before=False):
        super().__init__(data_file, label_file, window_size, normalize_after, use_relative, normalize_before)

    def __getitem__(self, idx):
        # Reshape from [num_features, time] -> [time, num_features]
        window = self.windows[idx].transpose(1, 0)  # Swap axes for Transformer compatibility
        label = self.targets[idx]  # Transform labels from 20-24 to 0-4
        return window, label  # Returns (time, features), label
    