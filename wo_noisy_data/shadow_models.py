import torch
import numpy as np
import os
import json
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
# Assuming these modules are in the correct path
from mmWaveHAR4.data_loader import HAR_Dataset
from mmWaveHAR4.model_trainer import CNN_LSTM_Model, train_model


def load_raw_data(filepath='../data_sets/npy_0.npz'):
    """Loads a pre-processed dataset with numeric labels."""
    print("##################### LOADING RAW DATA #####################")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset '{filepath}' not found. Please ensure the path is correct.")

    # Load the data and numeric labels directly
    with np.load(filepath) as doc:
        data = doc['data']
        # Ensure you use the correct key for your labels, likely 'labels'
        labels = doc['labels']

    filtered_data = np.array(data)
    mapped_labels = np.array(labels)

    # Check that data was loaded successfully
    if len(filtered_data) == 0:
         raise ValueError(f"No data loaded from '{filepath}'. The file might be empty or structured incorrectly.")

    print(f"Data shape: {filtered_data.shape}, Labels shape: {mapped_labels.shape}")
    print("##################### RAW DATA LOADED #####################\n")
    return filtered_data, mapped_labels


def extract_softmax_features(model, data, labels, member_flags, device='cpu'):
    """
    Extracts softmax probability vectors from a model for a given dataset.

    Args:
        model (torch.nn.Module): The trained model to use for feature extraction.
        data (np.array): The input data samples.
        labels (np.array): The corresponding labels for the data.
        member_flags (np.array): An array indicating membership status (1 for member, 0 for non-member).
        device (torch.device): The device to run the model on.

    Returns:
        (np.array, np.array): A tuple of (softmax_features, membership_labels).
    """
    model.eval()
    # Create a dataset and loader for the provided data
    dataset = HAR_Dataset(data, (np.min(data), np.max(data)), labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    features = []
    membership_labels = []
    current_pos = 0
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()

            # Append the probabilities and corresponding membership flags
            features.extend(probs)
            batch_size = len(batch_x)
            membership_labels.extend(member_flags[current_pos: current_pos + batch_size])
            current_pos += batch_size

    return np.array(features), np.array(membership_labels)


def generate_shadow_attack_data(data_pool, labels_pool, num_shadows, device, model_architecture):
    """
    Trains multiple shadow models and generates a dataset for the membership inference attack.
    This version uses a clean split of data for members and non-members.
    """
    all_features = []
    all_membership_labels = []

    print(f"--- Generating Attack Data using {num_shadows} Shadow Models ---")
    for i in range(num_shadows):
        print(f"\n[Shadow Model {i + 1}/{num_shadows}]")

        # 1. Create a new, random split of the data pool for each shadow model.
        # This is the core of the corrected logic. We split the clean data first.
        # 'shadow_train_data' will be the "members" for this shadow model.
        # 'shadow_out_data' will be the "non-members".
        shadow_train_data, shadow_out_data, shadow_train_labels, shadow_out_labels = train_test_split(
            data_pool, labels_pool, test_size=0.5, stratify=labels_pool, random_state=42 + i
        )
        print(f"  - Member set size: {len(shadow_train_data)}, Non-member set size: {len(shadow_out_data)}")

        # 2. Prepare DataLoaders for the "member" set to train the shadow model.
        # A further split is needed for a validation set during the model training phase.
        x_train, x_val, y_train, y_val = train_test_split(
            shadow_train_data, shadow_train_labels, test_size=0.15, stratify=shadow_train_labels, random_state=42 + i
        )

        global_min, global_max = np.min(x_train), np.max(x_train)
        train_loader = DataLoader(HAR_Dataset(x_train, (global_min, global_max), y_train), batch_size=16, shuffle=True)
        val_loader = DataLoader(HAR_Dataset(x_val, (global_min, global_max), y_val), batch_size=16, shuffle=False)

        # 3. Train the shadow model on its "member" data.
        print("  - Training shadow model...")
        shadow_model = train_model(train_loader, val_loader, device, model_architecture)
        print("  - Training complete.")

        # 4. Generate attack features using both the "member" and "non-member" sets.
        print("  - Extracting features...")

        # Combine the member and non-member data to extract features in one pass
        combined_data = np.concatenate([shadow_train_data, shadow_out_data])
        combined_labels = np.concatenate([shadow_train_labels, shadow_out_labels])

        # Create flags to label the data: 1 for members, 0 for non-members
        member_flags = np.concatenate([
            np.ones(len(shadow_train_data), dtype=int),
            np.zeros(len(shadow_out_data), dtype=int)
        ])

        features, membership_labels = extract_softmax_features(
            shadow_model, combined_data, combined_labels, member_flags, device=device
        )

        all_features.append(features)
        all_membership_labels.append(membership_labels)
        print(f"  - Extracted {len(features)} feature vectors.")

    # Concatenate all features/labels from all shadow models into the final attack dataset.
    final_features = np.concatenate(all_features, axis=0)
    final_membership_labels = np.concatenate(all_membership_labels, axis=0)

    return final_features, final_membership_labels


def main():
    """
    Main execution script. Loads data, generates the attack dataset using shadow
    models, and saves it to a file.
    """
    print("=== SHADOW MODEL FEATURE EXTRACTION PIPELINE (Corrected) ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\u2713 Using device: {device}")

    # Load parameters from the JSON file
    params_path = 'params.json'
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Configuration file '{params_path}' not found.")

    with open(params_path, 'r') as f:
        params = json.load(f)

    subset_frac = params.get('subset_size', 0.8)
    num_shadows = params.get('num_shadows', 16)
    print(f"\u2713 Loaded parameters: {num_shadows} shadow models, {subset_frac * 100}% subset size.")

    # Load the full dataset
    filtered_data, mapped_labels = load_raw_data()

    # Create a smaller data pool to work with, if specified
    if subset_frac < 1.0:
        total_samples = len(filtered_data)
        subset_size = int(subset_frac * total_samples)

        # Use train_test_split to get a stratified random subset
        data_pool, _, labels_pool, _ = train_test_split(
            filtered_data, mapped_labels, train_size=subset_size, stratify=mapped_labels, random_state=42
        )
    else:
        data_pool = filtered_data
        labels_pool = mapped_labels

    print(f"\u2713 Created a data pool of {len(data_pool)} samples for shadow modeling.\n")

    # Generate the attack dataset by training shadow models
    features, labels = generate_shadow_attack_data(
        data_pool=data_pool,
        labels_pool=labels_pool,
        num_shadows=num_shadows,
        device=device,
        model_architecture=CNN_LSTM_Model  # Pass the model class itself
    )

    # Save the final dataset to a file, overwriting if it exists
    # Ensure the directory exists
    output_dir = "shadow_models"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = os.path.join(output_dir, "shadow_dataset.npz")

    np.savez(save_path, features=features, labels=labels)
    print(f"\n\u2713 Attack dataset generation complete.")
    print(f"\u2713 Final features shape: {features.shape}")
    print(f"\u2713 Final labels distribution: {np.bincount(labels)}")
    print(f"\u2713 Saved shadow attack dataset to: {save_path}")


if __name__ == "__main__":
    main()