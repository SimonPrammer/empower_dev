#%%
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_curve, average_precision_score)
from matplotlib.backends.backend_pdf import PdfPages

from cnn1d_dataset import Cnn1dDataset
from cnn1d_model import CNN1D

# ----------------------------
# Directories & File Paths
# ----------------------------
data_dir = os.path.join("data", "2024-11-24T10_13_28_d300sec_w1000ms")
save_model_dir = os.path.join("saved_models", 'cnn1d_model_normalized_relative_w1000ms.pth')
run_reports_dir = os.path.join("run_reports", 'cnn1d_model_normalized_relative_w1000ms.pdf')

train_data_file = os.path.join(data_dir, "training.csv")
train_label_file = os.path.join(data_dir, "training_y.csv")
test_data_file = os.path.join(data_dir, "testing.csv")
test_label_file = os.path.join(data_dir, "testing_y.csv")

# ----------------------------
# Hyperparameters & Settings
# ----------------------------
window_size = 60          # Window length (sequence length)
batch_size = 32
num_epochs = 40
learning_rate = 0.001
validation_split = 0.2

# ----------------------------
# Data Loading
# ----------------------------
train_dataset = Cnn1dDataset(train_data_file, train_label_file, window_size, use_relative=True)
test_dataset = Cnn1dDataset(test_data_file, test_label_file, window_size, use_relative=True)

# Determine configuration parameters from the dataset
num_features = train_dataset.windows.shape[1]  # Number of features per timestep
sequence_length = train_dataset.windows.shape[2]  # Should be equal to window_size
num_classes = len(torch.unique(train_dataset.targets))  # Unique class labels

# Sanity checks
if len(train_dataset) == 0:
    print("Warning: Train dataset is empty after grouping/filtering.")
if len(test_dataset) == 0:
    print("Warning: Test dataset is empty after grouping/filtering.")

# Display sample shape and label
if len(train_dataset) > 0:
    sample_window, sample_label = train_dataset[0]
    print(f"Sample window shape: {sample_window.shape} (expected: [num_features, window_size])")
    print(f"Sample label: {sample_label}")

# Split training dataset into training and validation sets
train_size = int((1 - validation_split) * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# ----------------------------
# Model, Loss, Optimizer
# ----------------------------
model = CNN1D(num_features=num_features, sequence_length=window_size, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ----------------------------
# Containers for Metrics Tracking
# ----------------------------
train_losses = []
val_losses = []
train_acc_list = []
val_acc_list = []

# ----------------------------
# Training Loop
# ----------------------------
print("Starting training...")
training_start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0.0
    correct_train = 0
    total_train = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False)
    for inputs, labels in progress_bar:
        optimizer.zero_grad()
        outputs = model(inputs)  # Expected shape: [batch_size, num_classes]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()

        # Calculate training accuracy for the batch
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_train_loss = epoch_train_loss / len(train_loader) if len(train_loader) > 0 else 0
    train_accuracy = 100.0 * correct_train / total_train if total_train > 0 else 0.0
    train_losses.append(avg_train_loss)
    train_acc_list.append(train_accuracy)

    # Validation phase
    model.eval()
    epoch_val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    avg_val_loss = epoch_val_loss / len(val_loader) if len(val_loader) > 0 else 0
    val_accuracy = 100.0 * correct_val / total_val if total_val > 0 else 0.0
    val_losses.append(avg_val_loss)
    val_acc_list.append(val_accuracy)

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

training_end_time = time.time()
total_training_time = training_end_time - training_start_time
print(f"Total Training Time: {total_training_time:.2f} seconds. Train time per epoch: {total_training_time/num_epochs:.2f} seconds")

# ----------------------------
# Save the Model
# ----------------------------
torch.save(model.state_dict(), save_model_dir)

# ----------------------------
# Testing Phase & Metrics Collection
# ----------------------------
model.eval()
test_loss = 0.0
correct_test = 0
total_test = 0

# Containers to store predictions and ground-truth labels
all_test_preds = []
all_test_labels = []

inference_start_time = time.time()

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing", leave=False):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

        all_test_preds.append(predicted.cpu().numpy())
        all_test_labels.append(labels.cpu().numpy())

inference_end_time = time.time()
inference_time = inference_end_time - inference_start_time

avg_test_loss = test_loss / len(test_loader) if len(test_loader) > 0 else 0
test_accuracy = 100.0 * correct_test / total_test if total_test > 0 else 0.0

print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
print(f"Total Inference Time on test set: {inference_time:.2f} seconds")

# Concatenate predictions and labels from all batches
all_test_preds = np.concatenate(all_test_preds)
all_test_labels = np.concatenate(all_test_labels)

# ----------------------------
# Classification Report & Confusion Matrix
# ----------------------------
clf_report = classification_report(all_test_labels, all_test_preds,
                                   target_names=[f"Class {i}" for i in range(num_classes)])
print("\nClassification Report:")
print(clf_report)

cm = confusion_matrix(all_test_labels, all_test_preds)
fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[f"Class {i}" for i in range(num_classes)],
            yticklabels=[f"Class {i}" for i in range(num_classes)],
            ax=ax_cm)
ax_cm.set_xlabel("Predicted Label")
ax_cm.set_ylabel("True Label")
ax_cm.set_title("Confusion Matrix")

# ----------------------------
# Precision-Recall Curves
# ----------------------------
all_test_probs = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        all_test_probs.append(probs.cpu().numpy())
all_test_probs = np.concatenate(all_test_probs)

fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
for class_idx in range(num_classes):
    # Binary labels for one-vs-rest
    binary_labels = (all_test_labels == class_idx).astype(int)
    precision, recall, _ = precision_recall_curve(binary_labels, all_test_probs[:, class_idx])
    avg_prec = average_precision_score(binary_labels, all_test_probs[:, class_idx])
    ax_pr.plot(recall, precision, label=f"Class {class_idx} (AP={avg_prec:.2f})")
ax_pr.set_xlabel("Recall")
ax_pr.set_ylabel("Precision")
ax_pr.set_title("Precision-Recall Curves")
ax_pr.legend()

# ----------------------------
# Learning Curves: Loss & Accuracy
# ----------------------------
fig_lc, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))
# Loss curves
ax_loss.plot(train_losses, label='Training Loss', marker='o')
ax_loss.plot(val_losses, label='Validation Loss', marker='o')
ax_loss.set_title("Loss Curve")
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Loss")
ax_loss.legend()

# Accuracy curves
ax_acc.plot(train_acc_list, label='Training Accuracy', marker='o')
ax_acc.plot(val_acc_list, label='Validation Accuracy', marker='o')
ax_acc.set_title("Accuracy Curve")
ax_acc.set_xlabel("Epoch")
ax_acc.set_ylabel("Accuracy (%)")
ax_acc.legend()

fig_lc.tight_layout()

# ----------------------------
# Model Size Information
# ----------------------------
model_size_bytes = os.path.getsize(save_model_dir)
model_size_mb = model_size_bytes / (1024 * 1024)
print(f"Model Size: {model_size_mb:.2f} MB")

# ----------------------------
# PDF Report Generation
# ----------------------------
with PdfPages(run_reports_dir) as pdf:
    # Summary Page
    fig_summary = plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    summary_text = (
        f"CNN1D Model Training Report\n\n"
        f"Total Training Time: {total_training_time:.2f} seconds\n"
        f"Total Inference Time: {inference_time:.2f} seconds\n\n"
        f"Train time per epoch: {total_training_time/num_epochs:.2f} seconds\n\n"
        f"Test Loss: {avg_test_loss:.4f}\n"
        f"Test Accuracy: {test_accuracy:.2f}%\n\n"
        f"Model Size: {model_size_mb:.2f} MB\n"
    )
    plt.text(0.5, 0.5, summary_text, fontsize=12, ha='center', va='center', wrap=True)
    pdf.savefig(fig_summary)
    plt.close(fig_summary)
    
    # Configuration & Hyperparameters Page
    fig_config = plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    config_text = (
        "Configuration and Hyperparameters\n\n"
        "Data Paths:\n"
        f"  Data Directory: {data_dir}\n"
        f"  Training Data File: {train_data_file}\n"
        f"  Test Data File: {test_data_file}\n"
        f"  Model Save Path: {save_model_dir}\n"
        f"  PDF Report Path: {run_reports_dir}\n\n"
        "Basic Config:\n"
        f"  num_features: {num_features}\n"
        f"  window_size: {window_size}\n"
        f"  num_classes: {num_classes}\n"
        f"  batch_size: {batch_size}\n"
        f"  num_epochs: {num_epochs}\n"
        f"  learning_rate: {learning_rate}\n"
        f"  validation_split: {validation_split}\n"
    )
    plt.text(0.01, 0.99, config_text, fontsize=10, ha='left', va='top', family='monospace')
    pdf.savefig(fig_config)
    plt.close(fig_config)
    
    # Classification Report Page
    fig_clf = plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    plt.title("Classification Report", fontsize=16)
    plt.text(0.01, 0.99, clf_report, fontsize=10, ha='left', va='top', family='monospace')
    pdf.savefig(fig_clf)
    plt.close(fig_clf)
    
    # Confusion Matrix Page
    pdf.savefig(fig_cm)
    plt.close(fig_cm)
    
    # Precision-Recall Curves Page
    pdf.savefig(fig_pr)
    plt.close(fig_pr)
    
    # Learning Curves Page
    pdf.savefig(fig_lc)
    plt.close(fig_lc)

print(f"PDF report saved to: {run_reports_dir}")
