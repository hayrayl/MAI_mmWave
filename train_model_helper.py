
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from torch.utils.data import DataLoader

import utils
from mmWaveHAR4.data_loader import HAR_Dataset
from mmWaveHAR4.model_trainer import train_model, CNN_LSTM_Model
from shadow_models.SM_data_generation import extract_softmax_features, load_raw_data
from sklearn.model_selection import train_test_split
import joblib


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, labels = load_raw_data("./mmWaveHAR4/infocom24_dataset.npz")
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.5, stratify=labels, random_state=1337
    )

    # DEBUG: Use only a small subset for fast training
    # We can change this as needed later
    #train_data = train_data[:1000]
    #train_labels = train_labels[:1000]
    #test_data = test_data[:1000]
    #test_labels = test_labels[:1000]
    globalmax = np.max(train_data)
    globalmin = np.min(train_data)

    target_train_loader = DataLoader(HAR_Dataset(train_data, (globalmin, globalmax), train_labels), batch_size=16, shuffle=True)
    target_test_loader = DataLoader(HAR_Dataset(test_data, (globalmin, globalmax), test_labels), batch_size=16, shuffle=False)

    target_model = train_model(target_train_loader, target_test_loader, device, CNN_LSTM_Model)
    model_path = './target_model.pth'
    torch.save(target_model.state_dict(), model_path)
    print(f"✓ Model saved to: {model_path}")

    print("\n[2] Extracting features from target model...")
    all_target_data = np.concatenate([train_data, test_data])
    all_target_labels = np.concatenate([train_labels, test_labels])
    member_flags = np.concatenate([
        np.ones(len(train_data), dtype=int),  # training set = member
        np.zeros(len(test_data), dtype=int)  # test set = non-member
    ])

    features, labels = extract_softmax_features(
        target_model, all_target_data, all_target_labels, member_flags, device
    )

    features = np.array(features)
    labels = np.array(labels)

    save_path = "attack_on_target_dataset.npz"
    np.savez(save_path, features=features, labels=labels)
    print(f"\n\u2713 Extracted features shape: {features.shape}")
    print(f"\u2713 Saved target dataset to: {save_path}")

if __name__ == "__main__":
    main()