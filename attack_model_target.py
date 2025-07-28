"""
applies trained attack model to the target model's outputs to infer membership.

Expected:
- target_model trained on real HAR data (from model_trainer.py)
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from torch.utils.data import DataLoader

import utils
from mmWaveHAR4.data_loader import HAR_Dataset
from mmWaveHAR4.model_trainer import train_model, CNN_LSTM_Model
from shadow_models.SM_data_generation_test import extract_softmax_features, load_raw_data
from sklearn.model_selection import train_test_split
import joblib

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    attack_model = joblib.load("attack_model.pkl")
    print("Attack model loaded in")


    target_model = CNN_LSTM_Model().to(device)
    target_model.load_state_dict(torch.load('target_model.pth', weights_only=True, map_location=device))
    print("target model loaded in")

    # Load the .npz file directly and use the correct keys
    print("Loading attack data from 'attack_on_target_dataset.npz'")
    attack_data = np.load('attack_on_target_dataset_no_softmax.npz')
    features = attack_data['features']
    labels = attack_data['labels']

    # === Step 4: Run the attack on the target model ===
    print("\n[3] Running attack on target model outputs...")
    pred_labels = attack_model.predict(features)
    pred_scores = attack_model.predict_proba(features)[:, 1]

    for i in range(1,4):
        print("train", features[i])
        print("test", features[-i])
        print("", labels[i])

    print(classification_report(labels, pred_labels))
    print("Attack AUC on target model:", roc_auc_score(labels, pred_scores))
    print("Attack accuracy on target model:", attack_model.score(features, labels))


if __name__ == "__main__":
    main()
