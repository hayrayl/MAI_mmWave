import numpy as np
import joblib
import os
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    """
    Executes the final phase of the membership inference attack.

    This script loads the trained attack model and applies it to the outputs
    of the target model. It then evaluates the attack's performance by comparing
    its predictions to the true membership status of the data.
    """
    print("=" * 35)
    print("=== EXECUTING ATTACK ON TARGET MODEL ===")
    print("=" * 35)

    # --- 1. Define File Paths ---
    attack_model_path = "attack_model.pkl"
    target_dataset_path = "../attack_on_victim_dataset.npz"

    # --- 2. Load the Trained Attack Model ---
    if not os.path.exists(attack_model_path):
        raise FileNotFoundError(
            f"Error: Attack model not found at '{attack_model_path}'. "
            "Please run the attack model training script first."
        )

    print(f"\n[1] Loading trained attack model from '{attack_model_path}'...")
    attack_model = joblib.load(attack_model_path)
    print("  - Attack model loaded successfully.")

    # --- 3. Load the Target Model's Output Data ---
    # This file contains the softmax vectors from the target model when run on its
    # own training data (members) and test data (non-members).
    if not os.path.exists(target_dataset_path):
        raise FileNotFoundError(
            f"Error: Target dataset not found at '{target_dataset_path}'. "
            "Please run the 'train_model_helper.py' script to generate it."
        )

    print(f"\n[2] Loading target model's output data from '{target_dataset_path}'...")
    with np.load(target_dataset_path) as data:
        # 'features' are the softmax outputs from the target model
        # 'labels' are the true membership statuses (1 for member, 0 for non-member)
        features = data['features']
        true_labels = data['labels']

    print(f"  - Loaded {len(features)} data points to attack.")
    print(f"  - True label distribution: {np.bincount(true_labels)} (0: Non-Member, 1: Member)")

    # --- 4. Run the Inference Attack ---
    # The attack model predicts membership based on the target model's softmax outputs.
    print("\n[3] Predicting membership on the target model's outputs...")

    predicted_labels = attack_model.predict(features)
    # Get the probability scores for the 'member' class (class 1)
    predicted_scores = attack_model.predict_proba(features)[:, 1]

    print("  - Prediction complete.")

    # --- 5. Evaluate and Report Attack Performance ---
    print("\n[4] Evaluating attack performance...")

    accuracy = accuracy_score(true_labels, predicted_labels)
    auc_score = roc_auc_score(true_labels, predicted_scores)

    print("\n--- FINAL ATTACK RESULTS ---")
    print(f"Attack Accuracy: {accuracy:.4f}")
    print(f"Attack AUC Score: {auc_score:.4f}")
    print("----------------------------")

    # The classification report gives a detailed breakdown of performance.
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=['Non-Member', 'Member']))

    # Display a confusion matrix for a visual representation of the results.
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Non-Member', 'Predicted Member'],
                yticklabels=['Actual Non-Member', 'Actual Member'])
    plt.title('Attack Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')

    # In this environment, we can't show the plot directly, but this code would generate it.
    # To see the plot, you would typically call plt.show() or save the figure.
    plot_save_path = "attack_confusion_matrix.png"
    plt.savefig(plot_save_path)
    print(f"\nConfusion matrix saved to '{plot_save_path}'")
    print("\n=== ATTACK PIPELINE COMPLETE ===")

    return accuracy, auc_score

if __name__ == "__main__":
    main()