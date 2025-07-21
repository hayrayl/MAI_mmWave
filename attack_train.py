"""
Trains an attack model using softmax outputs from shadow models (in shadow_dataset.npz),
then applies it to the target model's outputs to infer membership.
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
import joblib
import utils

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

    joblib.dump(attack_model, "attack_model.pkl")
    print("Attack model saved to 'attack_model.plk'")

if __name__ == "__main__":
    main()