import pandas as pd
import numpy as np
import torch

def data_loader(file_path, num_features, sequence_length):
    """
    Loads and preprocesses data for the CNN model.
    
    Args:
        file_path (str): Path to the CSV file.
        num_features (int): Number of features per sample.
        sequence_length (int): Length of each sequence.
        
    Returns:
        torch.Tensor: Preprocessed tensor of shape (num_sequences, num_features, sequence_length).
    """
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Ensure the data has enough rows to form full sequences
    total_samples = len(df)
    samples_per_sequence = sequence_length * num_features
    total_usable_samples = (total_samples // samples_per_sequence) * samples_per_sequence

    # Trim and reshape the data
    data = df.values[:total_usable_samples]  # Trim excess rows
    data = data.reshape(-1, sequence_length, num_features)  # Reshape into sequences
    
    # Swap axes to match the model's expected input shape
    data = data.transpose(0, 2, 1)
    
    return torch.tensor(data, dtype=torch.float32)

