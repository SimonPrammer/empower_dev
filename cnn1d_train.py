#%%

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
from cnn1d_model import CNN1D
from legacy_data_loader import data_loader

# Parameters
num_features = 72
sequence_length = 30
num_classes = 5
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# File paths
train_X_file_path = "data/2024-11-24T10_13_28_d300sec_w1000ms/training.csv"
train_Y_file_path = "data/2024-11-24T10_13_28_d300sec_w1000ms/training_y.csv"
test_X_file_path = "data/2024-11-24T10_13_28_d300sec_w1000ms/testing.csv"
test_Y_file_path = "data/2024-11-24T10_13_28_d300sec_w1000ms/testing_y.csv"

# Data Loading
train_X = data_loader(train_X_file_path, num_features, sequence_length)
train_Y = torch.tensor(pd.read_csv(train_Y_file_path)['sidx'].values, dtype=torch.long)
test_X = data_loader(test_X_file_path, num_features, sequence_length)
test_Y = torch.tensor(pd.read_csv(test_Y_file_path)['sidx'].values, dtype=torch.long)

# Sanity check for size matches
assert len(train_X) == len(train_Y), "Mismatch between train_X and train_Y sizes!"
assert len(test_X) == len(test_Y), "Mismatch between test_X and test_Y sizes!"

# Split training data into training and validation sets
train_X, val_X, train_Y, val_Y = train_test_split(
    train_X, train_Y, test_size=0.2, random_state=42
)

# Create DataLoaders
train_dataset = TensorDataset(train_X, train_Y)
val_dataset = TensorDataset(val_X, val_Y)
test_dataset = TensorDataset(test_X, test_Y)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Model, Loss, Optimizer
model = CNN1D(num_features, sequence_length, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop with validation and progress bar
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for inputs, labels in progress_bar:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    # Validation after each epoch
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, "
        f"Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {100 * correct / total:.2f}%"
    )


#%%
# Testing
model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(
    f"Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {100 * correct / total:.2f}%"
)
