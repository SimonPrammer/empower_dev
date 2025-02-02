#%%
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np


class Cnn1dDataset(Dataset):
    def __init__(self, data_file, label_file, window_size, normalize_after=False, use_relative=False, normalize_before=False):
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
        self.normalize_before = normalize_before
        self.normalize_after = normalize_after
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
                group_df = data_lookup.get_group((g, w)) #.sort_values('tidx')
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

        #transform the labels to be 0-4 instead of 20-24
        # self.targets = torch.LongTensor(self.targets)
        # self.original_targets = self.targets.clone()
        self.targets = torch.LongTensor(self.targets) - 20
        self.original_targets = self.targets.clone() + 20  # Keep a copy of the original labels

        # interresting insights:
        # normalize first -> z-scores -> RNN/Transformers can't learn anything from the data 
        # only relative - no normalization -> RNN/Transformer learn something but quite slow
        # relative + normalize after -> RNN/Transformer learn well
        
        if self.normalize_before:
            self.normalize_windows()

        # noticed a big difference for RNNs/Transformers if we normalize before/after relative values or not normalize at all.
        if self.use_relative:
            # Compute differences along the time axis.
            # For each sample, for each feature: diff = current - previous.
            windows_diff = self.windows[:, :, 1:] - self.windows[:, :, :-1]
            # Pad the first time step with zeros to keep the same window size.
            pad = torch.zeros((self.windows.shape[0], self.windows.shape[1], 1), dtype=self.windows.dtype)
            self.windows = torch.cat([pad, windows_diff], dim=2)
        
        if self.normalize_after:
            self.normalize_windows()

        # Check consistency.
        assert len(self.targets) == len(self.windows), "Number of targets and windows don't match"

    def normalize_windows(self):
        """Normalize the windows feature-wise."""
        # Mean and std computed over all samples and time steps for each feature.

        # (0,1) seems to perform a lot worse than (0,2)
        # mean = self.windows.mean(dim=(0, 1), keepdim=True)
        # std = self.windows.std(dim=(0, 1), keepdim=True)

        mean = self.windows.mean(dim=(0, 2), keepdim=True)
        std = self.windows.std(dim=(0, 2), keepdim=True)
        self.windows = (self.windows - mean) / (std + 1e-6)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return self.windows[idx], self.targets[idx] 


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



    import torch
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import Counter
    from torch.utils.data import DataLoader
    from torch.nn import CrossEntropyLoss
    from cnn1d_dataset import Cnn1dDataset
    from cnn1d_model import CNN1D

    # Parameters
    num_features = 72 
    window_size = 60  # Adjust per file
    num_classes = 5
    batch_size = 32
    num_epochs = 40
    learning_rate = 0.001
    validation_split = 0.2

    # File paths
    data_dir = os.path.join("data", "2024-11-24T10_13_28_d300sec_w1000ms")
    train_data_file = os.path.join(data_dir, "training.csv")
    train_label_file = os.path.join(data_dir, "training_y.csv")
    test_data_file = os.path.join(data_dir, "testing.csv")
    test_label_file = os.path.join(data_dir, "testing_y.csv")

    # Load datasets
    train_dataset = Cnn1dDataset(train_data_file, train_label_file, window_size, use_relative=False)
    test_dataset = Cnn1dDataset(test_data_file, test_label_file, window_size, use_relative=False)

    # Configuration for CNN1D
    batch_size = 32
    num_features = train_dataset.windows.shape[1]  # Number of features per timestep
    sequence_length = train_dataset.windows.shape[2]  # Sequence length (window size)
    num_classes = len(torch.unique(train_dataset.targets))  # Unique class labels

    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1D(num_features=num_features, sequence_length=sequence_length, num_classes=num_classes).to(device)
    criterion = CrossEntropyLoss()

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



    @run_test
    def test_data_leakage():
        """Check if training and test datasets have overlapping (gidx, widx) pairs."""
        train_gidx_widx = set(zip(train_dataset.raw_labels['gidx'], train_dataset.raw_labels['widx']))
        test_gidx_widx = set(zip(test_dataset.raw_labels['gidx'], test_dataset.raw_labels['widx']))
        
        overlap = train_gidx_widx.intersection(test_gidx_widx)
        assert len(overlap) == 0, f"❌ Data Leakage Detected! {len(overlap)} overlapping samples."
        print("✅ No data leakage detected.")


    @run_test
    def test_label_mapping():
        """Check if all labels are correctly mapped from 20-24 to 0-4."""
        original_labels = train_dataset.original_targets.numpy()
        transformed_labels = train_dataset.targets.numpy()

        expected_mapping = {20: 0, 21: 1, 22: 2, 23: 3, 24: 4}

        print("\nOriginal Labels (First 10):", original_labels[:10])
        print("Transformed Labels (First 10):", transformed_labels[:10])

        for orig, trans in zip(original_labels, transformed_labels):
            if trans != expected_mapping[orig]:
                print(f"❌ Incorrect label mapping: {orig} -> {trans}")
                assert False, f"❌ Incorrect label mapping: {orig} -> {trans}"

        print("✅ Label mapping is correct.")


    @run_test
    def test_class_distribution():
        """Check if the train_dataset is imbalanced by plotting label distribution."""
        class_counts = Counter(train_dataset.targets.numpy())

        plt.bar(class_counts.keys(), class_counts.values(), tick_label=[20, 21, 22, 23, 24])
        plt.xlabel("Class (Original)")
        plt.ylabel("Count")
        plt.title("Class Distribution")
        plt.show()
        
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())

        imbalance_ratio = max_count / min_count
        assert imbalance_ratio < 5, f"❌ Class imbalance detected. Ratio: {imbalance_ratio:.2f}"
        print("✅ No severe class imbalance detected.")


    @run_test
    def test_overfitting():
        """Train for 5 epochs and check if training accuracy is much higher than test accuracy."""
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(5):
            model.train()
            correct_train = 0
            total_train = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs, 1)
                correct_train += (predicted == labels).sum().item()
                total_train += labels.size(0)

            train_acc = 100 * correct_train / total_train

            # Evaluate on test data
            model.eval()
            correct_test = 0
            total_test = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    correct_test += (predicted == labels).sum().item()
                    total_test += labels.size(0)

            test_acc = 100 * correct_test / total_test

            print(f"Epoch {epoch+1}: Train Acc = {train_acc:.2f}%, Test Acc = {test_acc:.2f}%")

        assert abs(train_acc - test_acc) < 10, f"❌ Possible overfitting! Train Acc = {train_acc:.2f}%, Test Acc = {test_acc:.2f}%"
        print("✅ No extreme overfitting detected.")


    @run_test
    def test_model_output_distribution():
        """Check if the model is predicting only one class (biased learning)."""
        model.eval()
        all_preds = []

        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())

        class_counts = Counter(all_preds)
        print("Model prediction distribution:", class_counts)

        assert len(class_counts) > 1, "❌ Model is predicting only one class!"
        print("✅ Model is making diverse predictions.")


    @run_test
    def test_no_nan_in_model():
        """Check if the model outputs contain NaN or Inf values."""
        model.eval()

        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)

                assert not torch.isnan(outputs).any(), "❌ Model output contains NaN values!"
                assert not torch.isinf(outputs).any(), "❌ Model output contains Inf values!"

        print("✅ No NaN or Inf values in model outputs.")
