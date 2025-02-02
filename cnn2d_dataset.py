#%%
import os
from cnn1d_dataset import Cnn1dDataset


class Cnn2dDataset(Cnn1dDataset):
    def __init__(self, data_file, label_file, window_size, normalize_after=False, use_relative=False, normalize_before=False):
        super().__init__(data_file, label_file, window_size, normalize_after, use_relative, normalize_before)
        # Reshape from [batch_size, num_features, window_size] -> [batch_size, channels=1, num_features, window_size]
        self.windows = self.windows.unsqueeze(1)

    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        return x, y  # Returns (batch, 1, features, time), label
    

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

    dataset = Cnn2dDataset(data_file, label_file, window_size)

    entry_index = 0 
    window, label = dataset[entry_index]

    print(f"Window (features) Shape: {window.shape}")
    print(f"Window (features) Content:\n{window}")
    print(f"Label: {label.item()}")
    print("dataset len",len(dataset))

    # raw_data = pd.read_csv(data_file, engine='python')
    # raw_labels = pd.read_csv(label_file, engine='python')

    # valid_gidx = set(raw_labels['gidx'])
    # raw_data = raw_data[raw_data['gidx'].isin(valid_gidx)]
