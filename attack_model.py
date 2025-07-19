"""
Trains an attack model using softmax outputs from shadow models (in shadow_dataset.npz),
then applies it to the target model's outputs to infer membership.

Expected:
- shadow_dataset.npz (from extract_shadow_features.py)
- target_model trained on real HAR data (from model_trainer.py)
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from torch.utils.data import DataLoader
from mmWaveHAR4.data_loader import HAR_Dataset
from mmWaveHAR4.model_trainer import train_model, CNN_LSTM_Model
from shadow_models.SM_data_generation import extract_softmax_features, load_raw_data
from sklearn.model_selection import train_test_split

def main():
    print("=== ATTACK MODEL ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === Step 1: Train the attack model from shadow dataset ===
    print("\n[1] Training attack model from shadow_dataset.npz...")
    shadow = np.load("./shadow_models/shadow_dataset.npz")
    X, y = shadow["features"], shadow["labels"]

    x_train, x_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    attack_model = LogisticRegression(max_iter=1000)
    attack_model.fit(x_train, y_train)

    y_pred = attack_model.predict(x_val)
    y_score = attack_model.predict_proba(x_val)[:, 1]

    print("Attack model trained.")
    print(classification_report(y_val, y_pred))
    print("AUC Score:", roc_auc_score(y_val, y_score))



    print("\n[2] Training target model...")

    data, labels = load_raw_data("./mmWaveHAR4/infocom24_dataset.npz")
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.5, stratify=labels, random_state=1337
    )

    # DEBUG: Use only a small subset for fast training 
    # We can change this as needed later
    train_data = train_data[:1000]
    train_labels = train_labels[:1000]
    test_data = test_data[:1000]
    test_labels = test_labels[:1000]
    globalmax = np.max(train_data)
    globalmin = np.min(train_data)

    target_train_loader = DataLoader(HAR_Dataset(train_data, (globalmin, globalmax), train_labels), batch_size=16, shuffle=True)
    target_test_loader = DataLoader(HAR_Dataset(test_data, (globalmin, globalmax), test_labels), batch_size=16, shuffle=False)

    target_model = train_model(target_train_loader, target_test_loader, device, CNN_LSTM_Model)

    # === Step 3: Extract softmax features from target model ===
    print("\n[3] Extracting features from target model...")
    all_target_data = np.concatenate([train_data, test_data])
    all_target_labels = np.concatenate([train_labels, test_labels])
    member_flags = np.concatenate([
        np.ones(len(train_data), dtype=int),   # training set = member
        np.zeros(len(test_data), dtype=int)    # test set = non-member
    ])

    features, labels = extract_softmax_features(
        target_model, all_target_data, all_target_labels, member_flags, device
    )

    # === Step 4: Run the attack on the target model ===
    print("\n[4] Running attack on target model outputs...")
    pred_membership = attack_model.predict_proba(features)
    print(pred_membership)
    scores = attack_model.score(features, labels)
    print(scores)


if __name__ == "__main__":
    main()
