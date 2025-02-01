#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft
import torch
from torch.utils.data import DataLoader, random_split
import os

# Plot Time-Series Signals
def plot_time_series(data, imu_id, feature_prefix="x_euler"):
    imu_columns = [f"{feature_prefix}_{imu_id}", f"y_euler_{imu_id}", f"z_euler_{imu_id}"]
    plt.figure(figsize=(12, 6))
    for col in imu_columns:
        plt.plot(data["tidx"], data[col], label=col)
    plt.title(f"Time-Series Signals for IMU {imu_id}")
    plt.xlabel("Time Index")
    plt.ylabel("Signal Value")
    plt.legend()
    plt.show()

# Correlation Heatmap
def plot_correlation_heatmap(data, imu_id):
    imu_columns = [col for col in data.columns if f"_{imu_id}" in col]
    corr_matrix = data[imu_columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(f"Feature Correlation Heatmap for IMU {imu_id}")
    plt.show()

# Fourier Transform
def plot_frequency_spectrum(data, feature):
    signal = data[feature]
    fft_values = np.abs(fft(signal))
    frequencies = np.fft.fftfreq(len(signal), d=1/60)  # Assuming 60Hz sampling rate
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies[:len(frequencies)//2], fft_values[:len(fft_values)//2])
    plt.title(f"Frequency Spectrum of {feature}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.show()

# Sliding Window Aggregates
def plot_sliding_window_aggregates(data, feature, window_size=30):
    rolling_mean = data[feature].rolling(window=window_size).mean()
    rolling_std = data[feature].rolling(window=window_size).std()
    plt.figure(figsize=(12, 6))
    plt.plot(data["tidx"], data[feature], label="Original Signal", alpha=0.7)
    plt.plot(data["tidx"], rolling_mean, label="Rolling Mean", linestyle="--")
    plt.plot(data["tidx"], rolling_std, label="Rolling Std Dev", linestyle="--")
    plt.title(f"Sliding Window Aggregates for {feature}")
    plt.xlabel("Time Index")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

# Principal Component Analysis (PCA)
def plot_pca(data, imu_id):
    from sklearn.decomposition import PCA

    imu_columns = [col for col in data.columns if f"_{imu_id}" in col]
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data[imu_columns].dropna())
    plt.figure(figsize=(8, 6))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.6)
    plt.title(f"PCA of Features for IMU {imu_id}")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

# Heatmap of Sliding Window Signals
def plot_signal_heatmap(data, imu_ids, start_idx, window_size):
    window_data = data.iloc[start_idx:start_idx + window_size]
    heatmap_data = np.array([window_data[[f"x_euler_{imu}" for imu in imu_ids]].values.T]).squeeze()
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, cmap="viridis", cbar=True)
    plt.title("Sliding Window Signal Heatmap")
    plt.xlabel("Time Index in Window")
    plt.ylabel("IMU")
    plt.show()

# Example Usage
def main():
    num_features = 72
    sequence_length = 30
    num_classes = 5
    batch_size = 32

    file_path = os.path.join("data", "2024-11-24T10_13_28_d300sec_w1000ms", "training.csv")
    data = data_loader(file_path, num_features, sequence_length)
        

    # Plot time-series for IMU 0
    plot_time_series(data, imu_id=0)

    # Correlation heatmap for IMU 0
    plot_correlation_heatmap(data, imu_id=0)

    # Frequency spectrum for x_euler_0
    plot_frequency_spectrum(data, feature="x_euler_0")

    # Sliding window aggregates for x_euler_0
    plot_sliding_window_aggregates(data, feature="x_euler_0")

    # PCA for IMU 0
    plot_pca(data, imu_id=0)

    # Signal heatmap for IMUs 0 and 1, starting at index 0, window size 50
    plot_signal_heatmap(data, imu_ids=[0, 1], start_idx=0, window_size=50)

if __name__ == "__main__":
    main()
