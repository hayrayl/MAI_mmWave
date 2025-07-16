# NOTE: Run this script from inside the wo_left_right directory with:
#   python extract_shadow_features.py
"""
This script extracts softmax features from the trained shadow model for membership inference attacks.
Each sample is labeled as a member (1) or non-member (0) based on whether it was used to train the shadow model.

The output is saved to `shadow_dataset.npz`, containing:
    - features: Softmax vectors from the shadow model
    - labels:   Membership labels (1 = member, 0 = non-member)
"""

import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from data_loader import HAR_Dataset
from model_trainer import CNN_LSTM_Model, train_model
import shadow_models as SM
from sklearn.model_selection import train_test_split

#function to load in rad data; abstracted out for modularity purposes
def load_raw_data():
    print("##################### LOADING RAW DATA #####################")
    doc = np.load('infocom24_dataset.npz')  # Load dataset from .npz file
    data, raw_labels = doc['data'], doc['label']  # Extract data and labels

    print("\r.", end='')
    # Map string labels to integer class indices
    label_map = {'push': 0, 'pull': 1, 'clockwise': 2, 'anticlockwise': 3}
    filtered_data, mapped_labels = [], []

    # Filter out samples not in label_map and map labels to integers
    for d, l in zip(data, raw_labels):
        if l in label_map:
            filtered_data.append(d)
            mapped_labels.append(label_map[l])
    print("\b\r..",end='')

    filtered_data = np.array(filtered_data)
    mapped_labels = np.array(mapped_labels)
    print("\b\b\r...")
    print("##################### RAW DATA LOADED #####################\n\n")

    return filtered_data, mapped_labels


## function to generate a noisy dataset for shadow train testing
# given initial dataset, number of shadow models, desired size of training subset per shadow, and perturbation fraction,
# we create a noisy dataset that can be decomposed into all the 

def noisy_data_generation(data, labels, num_shadows, sample_fraction=1.0, perturb_fraction=0.15, seed=None):
    print("##################### GENERATING NOISY DATA #####################")
    if seed is not None:
        np.random.seed(seed)
    
    dshape = data.shape
    
    n = len(data)
    total_features = dshape[1] * dshape[2] * dshape[3]
    noise_features = int(total_features * perturb_fraction)
    sample_size = int(n * sample_fraction)

    noisy_dataset = []
    noisy_labels = []

    for shadow_i in range(num_shadows):
        print(f"Generating data for shadow {shadow_i+1}/{num_shadows}", end="\r")
        
        sample_indx = np.random.choice(n, sample_size, replace=False)
        temp_copy = data[sample_indx].copy()

        for i in range(sample_size):
            flat_inds = np.random.choice(total_features, noise_features, replace=False)
            x, y, z = np.unravel_index(flat_inds, (dshape[1], dshape[2], dshape[3]))

            for j in range(noise_features):
                fudgefactor = np.random.uniform(-0.15, 0.15)
                temp_copy[i, x[j], y[j], z[j]] *= (1 + fudgefactor)

        noisy_dataset.append(temp_copy)
        noisy_labels.append(labels[sample_indx])  # only labels for the selected samples

    noisy_dataset = np.concatenate(noisy_dataset, axis=0)
    noisy_labels = np.concatenate(noisy_labels, axis=0)
    print("\n##################### NOISED DATASET GENERATED #####################\n\n")

    return noisy_dataset, noisy_labels

### function to handle training multiple shadow models at once.
## 
##
##
def train_multiple_shadows(data, labels, num_models, device, model_architecture):
    print("##################### TRAINING SHADOW MODELS #####################")
    models = []
    n = len(data)
    subset_size = n // num_models
    for i in range(num_models):
        print(f"training model {i+1}/{num_models}", end="\r")
        dataslice = data[i*subset_size: min((i+1)*subset_size, n)]
        labelslice = labels[i*subset_size: min((i+1)*subset_size, n)]

        seed = 42+i
        x_temp, x_test, y_temp, y_test = train_test_split(
                dataslice, labelslice, test_size=0.15, stratify=labelslice, random_state=seed)

        x_train, x_val, y_train, y_val = train_test_split(
            x_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=seed)

        global_min = np.min(x_temp)
        global_max = np.max(x_temp)

        train_dataset = HAR_Dataset(x_train, (global_min, global_max), y_train)
        val_dataset = HAR_Dataset(x_val, (global_min, global_max), y_val)
        test_dataset = HAR_Dataset(x_test, (global_min, global_max), y_test)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        shadow_model = train_model(train_loader, val_loader, device, model_architecture)
        models.append((shadow_model, train_loader, test_loader))

    print("##################### SHADOW MODELS TRAINED #####################")
    return models



# Function to extract softmax features from a model for a given dataset
# model: Trained PyTorch model
# data: Numpy array of input data
# labels: Numpy array of class labels (not used for prediction, but for dataset construction)
# member_flags: Array indicating membership (1 for member, 0 for non-member)
# device: 'cpu' or 'cuda'
def extract_softmax_features(model, data, labels, member_flags, device='cpu'):
    model.eval()  # Set model to evaluation mode
    dataset = HAR_Dataset(data, (np.min(data), np.max(data)), labels)  # Wrap data in custom PyTorch dataset
    loader = DataLoader(dataset, batch_size=32, shuffle=False)  # No shuffling for consistent order

    features, membership_labels = [], []  # Lists to store softmax outputs and membership labels
    i = 0  # Index to track position in member_flags

    with torch.no_grad():  # Disable gradient computation for inference
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)  # Move batch to device
            outputs = model(batch_x)  # Get model logits
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()  # Convert logits to softmax probabilities
            features.extend(probs.tolist())  # Store softmax vectors
            membership_labels.extend(member_flags[i:i + len(batch_x)])  # Store corresponding membership labels
            i += len(batch_x)  # Update index

    return np.array(features), np.array(membership_labels)  # Return as numpy arrays


SUBSET_FRAC = 0.03
NUM_SHADOWS = 2

def main():
    print("=== SHADOW MODEL FEATURE EXTRACTION ===")
    # Select device: use GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\u2713 Using device: {device}")

    # Load and preprocess data
    filtered_data, mapped_labels = load_raw_data()

    # Use only 3% of the total dataset for shadow model training and feature extraction
    total_samples = len(filtered_data)
    subset_size = int(SUBSET_FRAC * total_samples)
    np.random.seed(42)
    subset_indices = np.random.choice(total_samples, size=subset_size, replace=False)
    filtered_data = filtered_data[subset_indices]
    mapped_labels = mapped_labels[subset_indices]
    
    ## here we generate the noisy dataset, primed for the NUMSHADOWS
    noised_dataset, noised_dataset_labels = noisy_data_generation(filtered_data, mapped_labels, NUM_SHADOWS)

    # Train fewer shadow models (e.g., 1)
    model_packages = train_multiple_shadows(noised_dataset, noised_dataset_labels, NUM_SHADOWS, device, CNN_LSTM_Model) 

    # For each shadow model, extract features and membership labels
    all_features = []
    all_membership_labels = []
    for (model, train_loader, test_loader) in model_packages:
        # Get train and test data/labels for this shadow model
        s_train = train_loader.dataset.data.numpy()
        s_train_labels = train_loader.dataset.labels.numpy()
        s_test = test_loader.dataset.data.numpy()
        s_test_labels = test_loader.dataset.labels.numpy()

        member_flags = np.concatenate([
            np.ones(len(s_train), dtype=int),
            np.zeros(len(s_test), dtype=int)
        ])
        all_data = np.concatenate([s_train, s_test])
        all_labels = np.concatenate([s_train_labels, s_test_labels])

        features, membership_labels = extract_softmax_features(
            model, all_data, all_labels, member_flags, device=device
        )
        all_features.append(features)
        all_membership_labels.append(membership_labels)

    # Concatenate all features/labels from all shadow models
    all_features = np.concatenate(all_features, axis=0)
    all_membership_labels = np.concatenate(all_membership_labels, axis=0)

    # Save to file
    save_path = "shadow_dataset.npz"
    np.savez(save_path, features=all_features, labels=all_membership_labels)
    print(f"\n\u2713 Extracted features shape: {all_features.shape}")
    print(f"\u2713 Saved shadow dataset to: {save_path}")
    

if __name__ == "__main__":
    main()
