#%%
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt

from transformer_dataset import TransformerDataset  # Reusing your dataset class
from transformer_model import TransformerModel  # Import the transformer model

# ===========================
# Hyperparameters and Config
# ===========================
num_features = 72
window_size = 60
num_classes = 5
batch_size = 32
num_epochs = 10
learning_rate = 0.001
validation_split = 0.2

# Transformer-specific hyperparameters
d_model = 128         # Internal model dimension (after projecting input features)
num_heads = 4         # Number of attention heads (should divide d_model)
num_layers = 2        # Number of transformer encoder layers
dropout = 0.1         # Dropout rate

# ===========================
# File paths for data
# ===========================
data_dir = os.path.join("data", "2024-11-24T10_13_28_d300sec_w1000ms")
train_data_file = os.path.join(data_dir, "training.csv")
train_label_file = os.path.join(data_dir, "training_y.csv")
test_data_file = os.path.join(data_dir, "testing.csv")
test_label_file = os.path.join(data_dir, "testing_y.csv")

# ===========================
# Load and split datasets
# ===========================
# Note: The TransformerDataset is reused here; if it accepts a "use_relative" flag, pass it along.
train_dataset = TransformerDataset(train_data_file, train_label_file, window_size, use_relative=False)
test_dataset = TransformerDataset(test_data_file, test_label_file, window_size, use_relative=False)

# Split training dataset into training and validation sets
train_size = int((1 - validation_split) * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# DataLoaders for training, validation, and testing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size)
test_loader  = DataLoader(test_dataset, batch_size=batch_size)

# ===========================
# Initialize Model, Loss, Optimizer
# ===========================
model = TransformerModel(
    input_size=num_features,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    num_classes=num_classes,
    dropout=dropout
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ===========================
# Training Loop with Validation
# ===========================
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for inputs, labels in progress_bar:
        optimizer.zero_grad()
        outputs = model(inputs)         # Forward pass
        loss = criterion(outputs, labels)
        loss.backward()                 # Backward pass
        optimizer.step()                # Update weights

        train_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    # Validation phase
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

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100.0 * correct / total

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, "
          f"Val Acc: {val_accuracy:.2f}%")

# ===========================
# Save the trained model
# ===========================
torch.save(model.state_dict(), 'transformer_model.pth')

# ===========================
# Testing Phase
# ===========================
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

avg_test_loss = test_loss / len(test_loader)
test_accuracy = 100.0 * correct / total

print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

# ===========================
# Plot Training History
# ===========================
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
