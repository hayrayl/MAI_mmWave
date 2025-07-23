import torch
import numpy as np
import os
import json
from torch.utils.data import DataLoader
from mmWaveHAR4.data_loader import HAR_Dataset
from mmWaveHAR4.model_trainer import (CNN_LSTM_Model, train_model)
from sklearn.model_selection import train_test_split


# This function remains the same.
def load_raw_data(filepath='infocom24_dataset.npz'):
    """Loads and pre-processes the raw dataset from a given path."""
    print("##################### LOADING RAW DATA #####################")
    # Ensure the path to the dataset is correct
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset '{filepath}' not found. Make sure the path is correct.")

    doc = np.load(filepath)  # Use the provided filepath
    data, raw_labels = doc['data'], doc['label']

    label_map = {'push': 0, 'pull': 1, 'clockwise': 2, 'anticlockwise': 3}
    filtered_data, mapped_labels = [], []

    for d, l in zip(data, raw_labels):
        if l in label_map:
            filtered_data.append(d)
            mapped_labels.append(label_map[l])

    filtered_data = np.array(filtered_data)
    mapped_labels = np.array(mapped_labels)
    print("##################### RAW DATA LOADED #####################\n")
    return filtered_data, mapped_labels

# This function also remains the same.
def extract_softmax_features(model, data, labels, member_flags, device='cpu'):
    """Extracts softmax probability vectors from a model for a given dataset."""
    model.eval()
    dataset = HAR_Dataset(data, (np.min(data), np.max(data)), labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    features, membership_labels = [], []
    i = 0
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
            features.extend(probs.tolist())
            membership_labels.extend(member_flags[i:i + len(batch_x)])
            i += len(batch_x)
    return np.array(features), np.array(membership_labels)


def generate_shadow_attack_data(data_pool, labels_pool, num_shadows, perturb, noise, device, model_architecture):
    """
    Trains multiple shadow models and generates a dataset for the membership inference attack.

    For each shadow model, it:
    1. Splits the data pool into a 'member' set (for training) and a 'non-member' set.
    2. Trains a shadow model on the 'member' set.
    3. Uses the trained model to extract softmax features for both sets.
    4. Labels the features as member (1) or non-member (0).

    Args:
        data_pool (np.array): The pool of data samples available for training shadow models.
        labels_pool (np.array): The corresponding labels for the data_pool.
        num_shadows (int): The number of shadow models to train.
        device (torch.device): The device ('cpu' or 'cuda') to train on.
        model_architecture (torch.nn.Module): The class of the model to be trained (e.g., CNN_LSTM_Model).

    Returns:
        (np.array, np.array): A tuple containing the concatenated features and membership labels
                              from all shadow models.
    """
    all_features = []
    all_membership_labels = []

    dshape = data_pool.shape
    total_features = dshape[1] * dshape[2] * dshape[3]
    noise_features = int(total_features * perturb)
    datasize = len(data_pool)

    print(f"--- Generating Attack Data using {num_shadows} Shadow Models ---")
    for i in range(num_shadows):
        print(f"\n[Shadow Model {i + 1}/{num_shadows}]")

        data_copy = data_pool.copy()
        label_copy = labels_pool.copy()

        for i in range(datasize):
            flat_inds = np.random.choice(total_features, noise_features, replace=False)
            x, y, z = np.unravel_index(flat_inds, (dshape[1], dshape[2], dshape[3]))

            for j in range(noise_features):
                fudgefactor = np.random.uniform(-noise, noise)
                data_copy[i, x[j], y[j], z[j]] *= (1 + fudgefactor)

        # 1. Create a new, random split of the data pool for each shadow model.
        # This ensures each shadow model sees different "member" vs "non-member" data.
        shadow_train_data, shadow_out_data, shadow_train_labels, shadow_out_labels = train_test_split(
            data_copy, label_copy, test_size=0.5, stratify=label_copy, random_state=42 + i
        )
        print(f"  - Member set size: {len(shadow_train_data)}, Non-member set size: {len(shadow_out_data)}")

        # 2. Prepare DataLoaders for the "member" set to train the shadow model.
        # A further split is needed for a validation set during training.
        x_train, x_val, y_train, y_val = train_test_split(
            shadow_train_data, shadow_train_labels, test_size=0.15, stratify=shadow_train_labels, random_state=42 + i
        )

        global_min, global_max = np.min(x_train), np.max(x_train)
        train_loader = DataLoader(HAR_Dataset(x_train, (global_min, global_max), y_train), batch_size=16, shuffle=True)
        val_loader = DataLoader(HAR_Dataset(x_val, (global_min, global_max), y_val), batch_size=16, shuffle=False)

        # 3. Train the shadow model on the "member" data.
        print("  - Training shadow model...")
        shadow_model = train_model(train_loader, val_loader, device, model_architecture)
        print("  - Training complete.")

        # 4. Generate attack features using both the "member" and "non-member" sets.
        print(f"  - Extracting features...")
        member_data = shadow_train_data
        non_member_data = shadow_out_data

        combined_data = np.concatenate([member_data, non_member_data])
        combined_labels = np.concatenate([shadow_train_labels, shadow_out_labels])

        member_flags = np.concatenate([
            np.ones(len(member_data), dtype=int),
            np.zeros(len(non_member_data), dtype=int)
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
    print("=== SHADOW MODEL FEATURE EXTRACTION PIPELINE ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\u2713 Using device: {device}")

    params_path = 'params.json'
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Configuration file '{params_path}' not found.")

    with open(params_path, 'r') as f:
        params = json.load(f)

    subset_frac = params['subset_size']
    num_shadows = params['num_shadows']
    perturb_fraction = params['perturb_fraction']
    noise_fraction = params['noise_fraction']
    print(f"\u2713 Loaded parameters: {num_shadows} shadow models, {subset_frac * 100}% subset size.")

    # Load and create a smaller data pool to work with
    filtered_data, mapped_labels = load_raw_data("mmWaveHAR4/infocom24_dataset.npz")
    total_samples = len(filtered_data)
    subset_size = int(subset_frac * total_samples)

    ''' np.random.seed(42) '''
    subset_indices = np.random.choice(total_samples, size=subset_size, replace=False)
    data_pool = filtered_data[subset_indices]
    labels_pool = mapped_labels[subset_indices]
    print(f"\u2713 Created a data pool of {len(data_pool)} samples for shadow modeling.\n")

    # Generate the attack dataset by training shadow models
    features, labels = generate_shadow_attack_data(
        data_pool=data_pool,
        labels_pool=labels_pool,
        num_shadows=num_shadows,
        perturb=perturb_fraction,
        noise=noise_fraction,
        device=device,
        model_architecture=CNN_LSTM_Model
    )

    # Save the final dataset to a file, overwriting if it exists
    save_path = "shadow_dataset.npz"
    np.savez(save_path, features=features, labels=labels)
    print(f"\n\u2713 Attack dataset generation complete.")
    print(f"\u2713 Final features shape: {features.shape}")
    print(f"\u2713 Saved shadow attack dataset to: {save_path}")


if __name__ == "__main__":
    main()
