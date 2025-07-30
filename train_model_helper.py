
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from torch.utils.data import DataLoader

from mmWaveHAR4.data_loader import HAR_Dataset
from mmWaveHAR4.model_trainer import train_model, CNN_LSTM_Model
from sklearn.model_selection import train_test_split
import mmWaveHAR4.data_loader as loader
import joblib

def extract_softmax_features(model, data, labels, member_flags, device='cpu'):
    model.eval()  # Set model to evaluation mode
    dataset = HAR_Dataset(data, (np.min(data), np.max(data)), labels)  # Wrap data in custom PyTorch dataset
    loader = DataLoader(dataset, batch_size=16, shuffle=False)  # No shuffling for consistent order

    features, membership_labels = [], []  # Lists to store softmax outputs and membership labels
    i = 0  # Index to track position in member_flags

    with torch.no_grad():  # Disable gradient computation for inference
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)  # Move batch to device
            outputs = model(batch_x)  # Get model logits
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
            features.extend(probs.tolist())  # Store softmax vectors
            membership_labels.extend(member_flags[i:i + len(batch_x)])  # Store corresponding membership labels
            i += len(batch_x)  # Update index

    return np.array(features), np.array(membership_labels)  # Return as numpy arrays


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 1. Load the primary dataset for the victim model ---
    # Make sure your data_loader.py is pointing to the victim model's primary dataset (e.g., 'data_sets/npy_0.npz')
    print("[1] Loading primary dataset for victim model training...")
    train_loader, val_loader, test_loader, x_train, x_test, y_train, y_test = loader.load_data([])
    print(f"  - Victim model will be trained on {len(x_train)} samples.")
    print(f"  - Initial non-member test set contains {len(x_test)} samples.")

    # --- 2. Load the additional datasets to use as non-members ---
    print("\n[2] Loading additional datasets for the non-member pool...")
    additional_non_member_files = [
        'data_sets/npy_neg30.npz',
        'data_sets/npy_pos30.npz'
    ]

    additional_data_list = []
    additional_labels_list = []
    for f in additional_non_member_files:
        print(f"  - Loading {f}...")
        with np.load(f) as doc:
            additional_data_list.append(doc['data'])
            additional_labels_list.append(doc['labels'])

    additional_data = np.concatenate(additional_data_list, axis=0)
    additional_labels = np.concatenate(additional_labels_list, axis=0)
    print(f"  - Loaded {len(additional_data)} additional non-member samples.")

    # --- 3. Train the victim model (only on its member data) ---
    print("\n[3] Training the victim model...")
    victim_model = train_model(train_loader, val_loader, device, CNN_LSTM_Model)
    model_path = './victim_model.pth'
    torch.save(victim_model.state_dict(), model_path)
    print(f"✓ Victim model saved to: {model_path}")

    # --- 4. Prepare the final, combined dataset for the attack test ---
    print("\n[4] Preparing final dataset for attack evaluation...")

    # Combine the original test set with the additional data to form the full non-member pool
    #all_non_member_data = np.concatenate([additional_data])
    #all_non_member_labels = np.concatenate([additional_labels])

    # Combine members and the new, larger pool of non-members
    final_test_data = np.concatenate([x_train, additional_data])
    final_test_labels = np.concatenate([y_train, additional_labels])

    # Create the ground truth membership flags
    member_flags = np.concatenate([
        np.ones(len(x_train), dtype=int),  # Members
        np.zeros(len(additional_data), dtype=int)  # Non-Members
    ])
    print(f"  - Final test set: {len(x_train)} members and {len(additional_data)} non-members.")

    # --- 5. Extract features from the victim model ---
    print("\n[5] Extracting features from victim model...")
    features, labels = extract_softmax_features(
        victim_model, final_test_data, final_test_labels, member_flags, device
    )

    # The 'labels' returned here are the member_flags, which is what we want
    features = np.array(features)
    labels = np.array(labels)

    save_path = "attack_on_victim_dataset.npz"
    np.savez_compressed(save_path, features=features, labels=labels)
    print(f"\n\u2713 Extracted features shape: {features.shape}")
    print(f"\u2713 Saved final test dataset to: {save_path}")





# def main():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # DEBUG: Use only a small subset for fast training
#     # We can change this as needed later
#     #train_data = train_data[:1000]
#     #train_labels = train_labels[:1000]
#     #test_data = test_data[:1000]
#     #test_labels = test_labels[:1000]
#
#     # train_loader, val_loader, test_loader, x_train, x_test, y_train, y_test = loader.load_data(['left', 'right'])
#     train_loader, val_loader, test_loader, x_train, x_test, y_train, y_test = loader.load_data([])
#
#     target_model = train_model(train_loader, val_loader, device, CNN_LSTM_Model)
#     model_path = './victim_model.pth'
#     torch.save(target_model.state_dict(), model_path)
#     print(f"✓ Model saved to: {model_path}")
#
#     print("\n[2] Extracting features from target model...")
#     all_target_data = np.concatenate([x_train, x_test])
#     all_target_labels = np.concatenate([y_train, y_test])
#     member_flags = np.concatenate([
#         np.ones(len(x_train), dtype=int),  # training set = member
#         np.zeros(len(x_test), dtype=int)  # test set = non-member
#     ])
#
#     features, labels = extract_softmax_features(
#         target_model, all_target_data, all_target_labels, member_flags, device
#     )
#
#     features = np.array(features)
#     labels = np.array(labels)
#
#     save_path = "attack_on_victim_dataset.npz"
#     np.savez(save_path, features=features, labels=labels)
#     print(f"\n\u2713 Extracted features shape: {features.shape}")
#     print(f"\u2713 Saved target dataset to: {save_path}")

if __name__ == "__main__":
    main()