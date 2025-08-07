import numpy as np
import torch
from torch.utils.data import DataLoader
import os

# # Assuming the mmWaveHAR folder is at the same level as the MIA_Attack folder
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mmWaveHAR4.data_loader import HAR_Dataset, load_data as load_har_data
from mmWaveHAR4.model_trainer import train_model, CNN_LSTM_Model, CNN_LSTM_Model_Regularized


def extract_softmax_features(model, data, labels, member_flags, device='cpu'):
    """
    Extracts softmax probability vectors from a model for a given dataset.
    """
    model.eval()
    # The reference for normalization should be based on the training data range
    dataset = HAR_Dataset(data, (np.min(data), np.max(data)), labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    features, membership_labels = [], []
    current_pos = 0

    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
            features.extend(probs)
            batch_size = len(batch_x)
            membership_labels.extend(member_flags[current_pos : current_pos + batch_size])
            current_pos += batch_size

    return np.array(features), np.array(membership_labels)


def train_and_prepare_target(model_class, model_save_path, attack_dataset_save_path, train_loader, val_loader, x_train, y_train, non_member_data, non_member_labels, device):
    """
    A helper function to train a target model and prepare its corresponding attack dataset.
    """
    print("-" * 60)
    print(f"Processing Target Model: {model_class.__name__}")
    print("-" * 60)

    # --- 1. Train the target model ---
    print(f"\n[1] Training {model_class.__name__}...")
    target_model = train_model(train_loader, val_loader, device, model_class)
    torch.save(target_model.state_dict(), model_save_path)
    print(f"✓ Target model saved to: {model_save_path}")

    # --- 2. Prepare the final, combined dataset for the attack test ---
    print("\n[2] Preparing final dataset for attack evaluation...")
    final_test_data = np.concatenate([x_train, non_member_data])
    final_test_labels = np.concatenate([y_train, non_member_labels])

    member_flags = np.concatenate([
        np.ones(len(x_train), dtype=int),          # Members
        np.zeros(len(non_member_data), dtype=int) # Non-Members
    ])
    print(f"  - Final test set: {len(x_train)} members and {len(non_member_data)} non-members.")

    # --- 3. Extract features from the target model ---
    print("\n[3] Extracting features from target model...")
    features, labels = extract_softmax_features(
        target_model, final_test_data, final_test_labels, member_flags, device
    )

    np.savez_compressed(attack_dataset_save_path, features=features, labels=labels)
    print(f"\n✓ Extracted features shape: {features.shape}")
    print(f"✓ Saved final attack dataset to: {attack_dataset_save_path}")
    print("-" * 60)


def main():
    """
    Main script to train both target models (with and without regularization)
    and generate the necessary datasets for the membership inference attack.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load the primary dataset for the target models ---
    print("\n[+] Loading primary dataset for target model training...")
    # The `load_har_data` function splits data into train, val, and test
    train_loader, val_loader, _, x_train, _, y_train, _ = load_har_data([])
    print(f"  - Target models will be trained on {len(x_train)} samples.")

    # --- Load additional datasets to use as a common non-member pool ---
    print("\n[+] Loading additional datasets for the non-member pool...")
    additional_non_member_files = [
        './data_sets/npy_neg30.npz',
        './data_sets/npy_pos30.npz'
    ]
    additional_data_list = []
    additional_labels_list = []
    for f in additional_non_member_files:
        if os.path.exists(f):
            print(f"  - Loading {f}...")
            with np.load(f) as doc:
                additional_data_list.append(doc['data'])
                additional_labels_list.append(doc['labels'])
        else:
            print(f"  - WARNING: File not found: {f}. Skipping.")

    if not additional_data_list:
        print("FATAL ERROR: No non-member data files were found. Exiting.")
        return

    non_member_data = np.concatenate(additional_data_list, axis=0)
    non_member_labels = np.concatenate(additional_labels_list, axis=0)
    print(f"  - Loaded {len(non_member_data)} total non-member samples.")

    # --- Process the model WITHOUT regularization ---
    train_and_prepare_target(
        model_class=CNN_LSTM_Model,
        model_save_path='./target_model_no_regularization.pth',
        attack_dataset_save_path='./attack_on_target_no_regularization_dataset.npz',
        train_loader=train_loader,
        val_loader=val_loader,
        x_train=x_train,
        y_train=y_train,
        non_member_data=non_member_data,
        non_member_labels=non_member_labels,
        device=device
    )

    # --- Process the model WITH regularization ---
    train_and_prepare_target(
        model_class=CNN_LSTM_Model_Regularized,
        model_save_path='./target_model_with_regularization.pth',
        attack_dataset_save_path='./attack_on_target_with_regularization_dataset.npz',
        train_loader=train_loader,
        val_loader=val_loader,
        x_train=x_train,
        y_train=y_train,
        non_member_data=non_member_data,
        non_member_labels=non_member_labels,
        device=device
    )

    print("\n All target models trained and datasets prepared.")


if __name__ == "__main__":
    main()
