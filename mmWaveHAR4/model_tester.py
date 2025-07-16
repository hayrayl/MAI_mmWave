import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np


def test_model(model, data_loader, device):

    model.eval()
    y_true = []
    y_pred = []
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')

    return y_true, y_pred, accuracy


def plot_confusion_matrix(y_true, y_pred, folder, dataset_name):

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['push', 'pull', 'clockwise', 'anticlockwise'])


    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title(f'Confusion Matrix - {dataset_name.title()} Set')
    plt.tight_layout()


    save_path = f'{folder}/cm_{dataset_name}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def detailed_classification_report(y_true, y_pred, folder, dataset_name):

    class_names = ['push', 'pull', 'clockwise', 'anticlockwise']

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    print(f"\n=== {dataset_name.upper()} SET CLASSIFICATION REPORT ===")
    print(report)

    report_path = f'{folder}/classification_report_{dataset_name}.txt'
    with open(report_path, 'w') as f:
        f.write(f"Classification Report - {dataset_name.title()} Set\n")
        f.write("=" * 50 + "\n")
        f.write(report)

    return report_path


def compare_predictions(y_true, y_pred, dataset_name):

    class_names = ['push', 'pull', 'clockwise', 'anticlockwise']

    print(f"\n=== {dataset_name.upper()} SET PREDICTION ANALYSIS ===")


    total = len(y_true)
    correct = np.sum(np.array(y_true) == np.array(y_pred))
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {total - correct}")
    print(f"Accuracy: {100 * correct / total:.2f}%")


    print("\nPer-class accuracy:")
    for i, class_name in enumerate(class_names):
        class_mask = np.array(y_true) == i
        if np.sum(class_mask) > 0:
            class_correct = np.sum((np.array(y_true) == i) & (np.array(y_pred) == i))
            class_total = np.sum(class_mask)
            class_acc = 100 * class_correct / class_total
            print(f"  {class_name}: {class_acc:.2f}% ({class_correct}/{class_total})")

    return correct, total