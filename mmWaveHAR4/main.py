import os
import torch
import numpy as np
import random

from torch.serialization import skip_data

import data_loader
import model_trainer
import model_tester
import time

from model_trainer import CNN_LSTM_Model


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    print("=" * 60)
    print("           HAR SYSTEM - CNN+LSTM MODEL TRAINING")
    print("=" * 60)

    set_seed(42)
    print("✓ Random seeds set for reproducibility")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")

    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    # Skip labels (excluding 'left' and 'right')
    # skip_list = ['left', 'right']
    skip_list = []
    print(f"✓ Excluding labels: {skip_list}")

    # folder = 'wo_' + '_'.join(skip_list)
    folder = "../victim_model"
    os.makedirs(folder, exist_ok=True)
    print(f"✓ Results will be saved to: {folder}/")

    print("\n" + "─" * 40)
    print("LOADING DATA")
    print("─" * 40)

    try:
        train_loader, val_loader, test_loader = data_loader.load_data(skip_list)
        print("✓ Data loaded successfully!")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        return
    except Exception as e:
        print(f"✗ Unexpected error loading data: {e}")
        return

    print("\n" + "─" * 40)
    print("TRAINING MODEL")
    print("─" * 40)

    start_time = time.time()
    model = model_trainer.train_model(train_loader, val_loader, device)
    training_time = time.time() - start_time

    print(f"✓ Training completed in {training_time / 60:.1f} minutes")

    model_path = f'{folder}/victim_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"✓ Model saved to: {model_path}")

    print("\n" + "─" * 40)
    print("MODEL EVALUATION")
    print("─" * 40)

    print("\n1. VALIDATION SET EVALUATION:")
    print("   (Data used during training for early stopping)")
    val_y_true, val_y_pred, val_accuracy = model_tester.test_model(model, val_loader, device)
    val_cm_path = model_tester.plot_confusion_matrix(val_y_true, val_y_pred, folder, 'validation')
    val_report_path = model_tester.detailed_classification_report(val_y_true, val_y_pred, folder, 'validation')
    model_tester.compare_predictions(val_y_true, val_y_pred, 'validation')

    print("\n2. TEST SET EVALUATION:")
    print("   (Truly unseen data - final performance metric)")
    test_y_true, test_y_pred, test_accuracy = model_tester.test_model(model, test_loader, device)
    test_cm_path = model_tester.plot_confusion_matrix(test_y_true, test_y_pred, folder, 'test')
    test_report_path = model_tester.detailed_classification_report(test_y_true, test_y_pred, folder, 'test')
    model_tester.compare_predictions(test_y_true, test_y_pred, 'test')

    print("\n3. TRAINING SET EVALUATION:")
    print("   (Training data - should have high accuracy)")
    train_y_true, train_y_pred, train_accuracy = model_tester.test_model(model, train_loader, device)
    train_cm_path = model_tester.plot_confusion_matrix(train_y_true, train_y_pred, folder, 'training')
    train_report_path = model_tester.detailed_classification_report(train_y_true, train_y_pred, folder, 'training')

    print("\n" + "=" * 60)
    print("                    FINAL RESULTS SUMMARY")
    print("=" * 60)

    print(f"Training Time: {training_time / 60:.1f} minutes")
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model Size: {os.path.getsize(model_path) / (1024 * 1024):.1f} MB")

    print(f"\nACCURACY RESULTS:")
    print(f"  Training Accuracy:   {train_accuracy:.2f}%")
    print(f"  Validation Accuracy: {val_accuracy:.2f}%")
    print(f"  Test Accuracy:       {test_accuracy:.2f}% ← MAIN METRIC")

    train_val_gap = train_accuracy - val_accuracy
    val_test_gap = val_accuracy - test_accuracy

    print(f"\nOVERFITTING ANALYSIS:")
    print(f"  Train-Validation gap: {train_val_gap:.2f}%")
    print(f"  Validation-Test gap:  {val_test_gap:.2f}%")

    if train_val_gap > 10:
        print("   Large train-validation gap suggests overfitting")
    elif train_val_gap > 5:
        print("  Moderate train-validation gap")
    else:
        print("  Good generalization from training to validation")

    if val_test_gap > 5:
        print("  Validation-test gap suggests validation set was used for model selection")
    else:
        print(" Good generalization from validation to test")

    print(f"\nSAVED FILES:")
    print(f"  Model: {model_path}")
    print(f"  Confusion Matrices:")
    print(f"    • Training: {train_cm_path}")
    print(f"    • Validation: {val_cm_path}")
    print(f"    • Test: {test_cm_path}")
    print(f"  Classification Reports:")
    print(f"    • Training: {train_report_path}")
    print(f"    • Validation: {val_report_path}")
    print(f"    • Test: {test_report_path}")




if __name__ == '__main__':
    main()