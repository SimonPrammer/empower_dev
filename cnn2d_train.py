#%%

#TODO: save the model weights
#TODO: nice graph of train and validation loss

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torch.optim as optim
from cnn2d_dataset import Cnn2dDataset
from cnn2d_model import CNN2D


# Parameters
num_features = 72 
window_size = 60  # Adjust per file
num_classes = 5
batch_size = 32
num_epochs = 10
learning_rate = 0.001
validation_split = 0.2


# File paths
data_dir = os.path.join("data", "2024-11-24T10_13_28_d300sec_w1000ms")
train_data_file = os.path.join(data_dir, "training.csv")
train_label_file = os.path.join(data_dir, "training_y.csv")
test_data_file = os.path.join(data_dir, "testing.csv")
test_label_file = os.path.join(data_dir, "testing_y.csv")

# Load datasets
train_dataset = Cnn2dDataset(train_data_file, train_label_file, window_size, use_relative=True)
test_dataset = Cnn2dDataset(test_data_file, test_label_file, window_size, use_relative=True)


# Configuration for CNN2D
batch_size = 32
windows, _ = train_dataset[0]  # Get sample window to determine shape
channels = windows.shape[1]  # Should be 1
num_features = windows.shape[2] if len(windows.shape) == 4 else windows.shape[1]
sequence_length = window_size
num_classes = 5  # Known from dataset (20-24 mapped to 0-4)

# Initialize the model
model = CNN2D(num_features=num_features, sequence_length=sequence_length, num_classes=num_classes)


# Sanity check
if len(train_dataset) == 0:
    print("Warning: Train dataset is empty after grouping/filtering.")
if len(test_dataset) == 0:
    print("Warning: Test dataset is empty after grouping/filtering.")

# Test shape compatibility
if len(train_dataset) > 0:
    sample_window, sample_label = train_dataset[0]
    print(f"Sample window shape: {sample_window.shape} (should be [1, num_features, window_size])")
    print(f"Sample label: {sample_label}")

# Split train dataset into training and validation sets
train_size = int((1 - validation_split) * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Model, Loss, Optimizer
model = CNN2D(num_features=num_features, sequence_length=window_size, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop with validation
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for inputs, labels in progress_bar:
        # inputs shape: [batch_size, num_features, window_size]
        optimizer.zero_grad()
        outputs = model(inputs)  # No permute here
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            # Also no permute here
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
    avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
    val_accuracy = 100.0 * correct / total if total > 0 else 0.0

    print(
        f"Epoch {epoch+1}/{num_epochs}, "
        f"Train Loss: {avg_train_loss:.4f}, "
        f"Val Loss: {avg_val_loss:.4f}, "
        f"Val Acc: {val_accuracy:.2f}%"
    )

# Testing
model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        # No permute
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

avg_test_loss = test_loss / len(test_loader) if len(test_loader) > 0 else 0
test_accuracy = 100.0 * correct / total if total > 0 else 0.0
print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

# %%
