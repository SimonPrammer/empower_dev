import os
import unittest
from sliding_window_dataset import SlidingWindowDataset

class TestSlidingWindowDataset(unittest.TestCase):
    def setUp(self):
        # Define file paths
        self.data_file = os.path.join("data", "2024-11-24T10_13_28_d300sec_w1000ms", "training.csv")
        self.label_file = os.path.join("data", "2024-11-24T10_13_28_d300sec_w1000ms", "training_y.csv")
        self.window_size = 30
        self.stride = 15

        # Load the dataset
        self.dataset = SlidingWindowDataset(
            data_file=self.data_file,
            label_file=self.label_file,
            window_size=self.window_size,
            stride=self.stride,
        )

    def test_file_loading(self):
        """Test if files are loaded correctly."""
        self.assertGreater(len(self.dataset.data), 0, "Data file is empty.")
        self.assertGreater(len(self.dataset.labels), 0, "Label file is empty.")

    def test_gidx_alignment(self):
        """Test if gidx values align between data and labels."""
        data_gidx = set(self.dataset.data["gidx"])
        label_gidx = set(self.dataset.labels["gidx"])
        self.assertTrue(data_gidx.issubset(label_gidx), "Mismatch between gidx in data and labels.")

    def test_total_windows(self):
        """Test if the total windows are calculated correctly."""
        expected_windows = (len(self.dataset.data) - self.window_size) // self.stride + 1
        self.assertEqual(len(self.dataset), expected_windows, "Total windows mismatch.")

    def test_window_extraction(self):
        """Test if data windows are correctly extracted."""
        window, label = self.dataset[0]
        self.assertEqual(window.shape, (self.window_size, self.dataset.data.shape[1] - 3), "Window shape mismatch.")
        self.assertIsInstance(label.item(), int, "Label is not an integer.")

    def test_label_assignment(self):
        """Test if labels are assigned correctly based on gidx."""
        for idx in range(len(self.dataset)):
            _, label = self.dataset[idx]
            gidx = self.dataset.data_gidx[idx]
            expected_label = self.dataset.label_mapping.loc[gidx]["sidx"]
            self.assertEqual(label.item(), expected_label, f"Label mismatch at index {idx}.")

    def test_error_handling(self):
        """Test error handling for missing or malformed files."""
        with self.assertRaises(ValueError):
            SlidingWindowDataset("invalid_file.csv", self.label_file, self.window_size, self.stride)

        with self.assertRaises(ValueError):
            SlidingWindowDataset(self.data_file, "invalid_file.csv", self.window_size, self.stride)

if __name__ == "__main__":
    unittest.main()
