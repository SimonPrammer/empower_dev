import unittest
from torch.utils.data import DataLoader
import os
from sliding_window_dataset import SlidingWindowDataset

class TestSlidingWindowDataset(unittest.TestCase):
    def setUp(self):
        self.data_file = os.path.join("data", "2024-11-24T10_13_28_d300sec_w1000ms", "training.csv")
        self.label_file = os.path.join("data", "2024-11-24T10_13_28_d300sec_w1000ms", "training_y.csv")
        self.window_size = 30
        self.stride = 15

        # Create dataset
        self.dataset = SlidingWindowDataset(
            data_file=self.data_file,
            label_file=self.label_file,
            window_size=self.window_size,
            stride=self.stride
        )

    def test_dataset_length(self):
        """Test if the dataset length matches expected number of windows."""
        expected_length = (len(self.dataset.data) - self.window_size) // self.stride + 1
        self.assertEqual(len(self.dataset), expected_length,
                         f"Dataset length mismatch: {len(self.dataset)} vs {expected_length}")

    def test_window_shape(self):
        """Test if each data window has the correct shape."""
        window, _ = self.dataset[0]
        self.assertEqual(window.shape, (self.window_size, self.dataset.data.shape[1] - 3),
                         f"Window shape mismatch: {window.shape}")

    def test_label_alignment(self):
        """Test if labels are correctly aligned with data windows."""
        for idx in range(len(self.dataset)):
            _, label = self.dataset[idx]
            gidx = self.dataset.data_gidx[idx]
            expected_label = self.dataset.label_mapping.loc[gidx]["sidx"]
            self.assertEqual(label.item(), expected_label,
                             f"Label mismatch at index {idx}: {label.item()} vs {expected_label}")

    def test_dataloader_integration(self):
        """Test if the dataset works with a PyTorch DataLoader."""
        dataloader = DataLoader(self.dataset, batch_size=32, shuffle=True)
        for batch_idx, (data, labels) in enumerate(dataloader):
            self.assertEqual(data.shape[1:], (self.window_size, self.dataset.data.shape[1] - 3),
                             f"Batch data shape mismatch: {data.shape}")
            self.assertEqual(labels.shape[0], data.shape[0],
                             f"Batch label shape mismatch: {labels.shape} vs {data.shape[0]}")
            if batch_idx > 2:  # Limit to a few batches for testing
                break

if __name__ == "__main__":
    unittest.main()
