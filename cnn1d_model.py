#%%

import torch
import torch.nn as nn
import unittest

class CNN1D(nn.Module):
    def __init__(self, num_features, sequence_length, num_classes):
        super(CNN1D, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(128 * (sequence_length // 4), 128),  # Adjust based on pooling layers
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

