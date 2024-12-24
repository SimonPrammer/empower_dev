import torch
import unittest
from cnn1d import CNN1D

class TestCNN1D(unittest.TestCase):
    def setUp(self):
        self.num_features = 72
        self.sequence_length = 30
        self.num_classes = 5
        self.batch_size = 32
        self.model = CNN1D(num_features=self.num_features, sequence_length=self.sequence_length, num_classes=self.num_classes)

    def test_output_shape(self):
        """Test if the model produces the correct output shape."""
        input_tensor = torch.randn(self.batch_size, self.num_features, self.sequence_length)
        output = self.model(input_tensor)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes),
                         f"Unexpected output shape: {output.shape}, expected ({self.batch_size}, {self.num_classes})")

    def test_no_nan_inf(self):
        """Test if the model output contains NaN or Inf values."""
        input_tensor = torch.randn(self.batch_size, self.num_features, self.sequence_length)
        output = self.model(input_tensor)
        self.assertFalse(torch.isnan(output).any(), "Output contains NaN values.")
        self.assertFalse(torch.isinf(output).any(), "Output contains Inf values.")

    def test_varying_batch_sizes(self):
        """Test if the model handles varying batch sizes."""
        for batch_size in [1, 16, 64]:
            input_tensor = torch.randn(batch_size, self.num_features, self.sequence_length)
            output = self.model(input_tensor)
            self.assertEqual(output.shape, (batch_size, self.num_classes),
                             f"Unexpected output shape for batch size {batch_size}: {output.shape}, expected ({batch_size}, {self.num_classes})")

    def test_dataset_compatibility(self):
        """Test if the model is compatible with simulated dataset input."""
        input_tensor = torch.randn(1, self.num_features, self.sequence_length)  # Simulating one window from dataset
        output = self.model(input_tensor)
        self.assertEqual(output.shape, (1, self.num_classes),
                         f"Dataset simulation failed: {output.shape} instead of (1, {self.num_classes})")

if __name__ == "__main__":
    unittest.main()
