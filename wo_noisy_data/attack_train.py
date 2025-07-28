import numpy as np
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score


def get_attack_model(params):
    """
    Selects and instantiates the attack model based on parameters.
    """
    model_name = params.get('attack_architecture', 'RandomForest')

    print(f"  - Selected attack model: {model_name}")

    if model_name == 'RandomForest':
        # A powerful and robust default choice
        return RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1  # Use all available CPU cores
        )
    elif model_name == 'LogisticRegression':
        # A simple linear baseline model
        return LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000  # Increase iterations for convergence
        )
    elif model_name == 'MLP':
        # A neural network classifier
        epochs = params.get('attack_model_epochs', 500)
        print(f"  - MLP training with max_iter (epochs): {epochs}")
        return MLPClassifier(
            hidden_layer_sizes=(100, 50),  # A simple two-layer network
            max_iter=epochs,
            random_state=42,
            early_stopping=True,  # Helps prevent overfitting
            verbose=False  # Set to True to see training progress
        )
    else:
        raise ValueError(f"Unknown attack model '{model_name}' specified in params.json")


def main():
    """
    Trains a membership inference attack model based on settings in params.json.
    """
    print("=" * 25)
    print("=== ATTACK MODEL TRAINING ===")
    print("=" * 25)

    # --- 1. Load Parameters ---
    params_path = 'params.json'
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Configuration file '{params_path}' not found.")
    with open(params_path, 'r') as f:
        params = json.load(f)

    # --- 2. Load the Shadow Dataset ---
    shadow_dataset_path = os.path.join("shadow_models", "shadow_dataset.npz")
    if not os.path.exists(shadow_dataset_path):
        raise FileNotFoundError(f"Error: Shadow dataset not found at '{shadow_dataset_path}'.")

    print(f"\n[1] Loading attack training data from '{shadow_dataset_path}'...")
    with np.load(shadow_dataset_path) as shadow_data:
        X = shadow_data["features"]
        y = shadow_data["labels"]

    print(f"  - Loaded {len(X)} samples.")
    print(f"  - Label distribution: {np.bincount(y)} (0: Non-Member, 1: Member)")

    # --- 3. Split Data for Training and Validation ---
    print("\n[2] Splitting data into training and validation sets...")
    x_train, x_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  - Training set size: {len(x_train)}")
    print(f"  - Validation set size: {len(x_val)}")

    # --- 4. Get and Train the Attack Model ---
    print("\n[3] Initializing and training the attack model...")
    attack_model = get_attack_model(params)

    attack_model.fit(x_train, y_train)
    print("  - Attack model training complete.")

    # --- 5. Evaluate the Attack Model ---
    print("\n[4] Evaluating attack model performance on the validation set...")
    y_pred = attack_model.predict(x_val)
    y_score = attack_model.predict_proba(x_val)[:, 1]

    print("\n--- Validation Results ---")
    print(classification_report(y_val, y_pred, target_names=['Non-Member', 'Member']))
    auc_score = roc_auc_score(y_val, y_score)
    print(f"AUC Score: {auc_score:.4f}")
    print("--------------------------")

    # --- 6. Save the Trained Model ---
    model_save_path = "attack_model.pkl"
    joblib.dump(attack_model, model_save_path)
    print(f"\n[5] Attack model saved to '{model_save_path}'")
    print("\n=== TRAINING COMPLETE ===")


if __name__ == "__main__":
    main()
