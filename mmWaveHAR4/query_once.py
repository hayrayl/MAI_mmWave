import torch
import numpy as np
import argparse
from model_trainer import CNN_LSTM_Model


def query_model(model_path, data_path, sample_index):
    print("--- Initializing Query ---")

    # 1. Load Model
    device = torch.device("cpu")
    model = CNN_LSTM_Model()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print(f"✓ Model loaded successfully from '{model_path}'")

    # 2. Load and Prepare Data
    print(f"✓ Loading data from '{data_path}'...")
    doc = np.load(data_path)
    all_data, all_labels = doc['data'], doc['label']

    data_min = np.min(all_data)
    data_max = np.max(all_data)

    sample = all_data[sample_index]
    normalized_sample = (sample - data_min) / (data_max - data_min)
    label_str = all_labels[sample_index]

    sample_tensor = torch.from_numpy(normalized_sample).float()

    if sample_tensor.dim() == 3:
        sample_tensor = sample_tensor.unsqueeze(1)
        sample_tensor = sample_tensor.unsqueeze(0)
    else:
        print(f"✗ Error: The data sample has an unexpected shape: {sample_tensor.shape}")
        return

    sample_tensor = sample_tensor.to(device)
    print(f"✓ Prepared input data with shape: {sample_tensor.shape}")

    # 3. Perform Query and Get Confidence Scores
    with torch.no_grad():
        output = model(sample_tensor)

        # NEW: Apply the Softmax function to convert logits to probabilities
        probabilities = torch.nn.functional.softmax(output, dim=1)

    predicted_class_index = torch.argmax(probabilities, dim=1).item()
    # NEW: Get the confidence score for the winning class
    confidence = probabilities[0][predicted_class_index].item()

    # 4. Display the Result
    activity_labels = {0: 'push', 1: 'pull', 2: 'clockwise', 3: 'anticlockwise'}
    predicted_activity = activity_labels.get(predicted_class_index, "Unknown")

    print("\n--- ✅ QUERY RESULT ---")
    print(f"Sample Index:      {sample_index}")
    print(f"Actual Label was:  '{label_str}'")
    print("-" * 25)
    print(f"Model Predicted:   '{predicted_activity}' with {confidence:.2%} confidence")
    print("-" * 25)

    # NEW: Display the full confidence vector
    print("Confidence Vector:")
    for i in range(len(activity_labels)):
        class_name = activity_labels[i]
        class_confidence = probabilities[0][i].item()
        print(f"  - {class_name:<15}: {class_confidence:.2%}")
    print("----------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Query the trained HAR model.")
    parser.add_argument('--data_path', type=str, default='infocom24_dataset.npz',
                        help='Path to your .npz dataset file.')
    parser.add_argument('--index', type=int, default=0, help='Index of the sample to query.')
    args = parser.parse_args()

    query_model(
        model_path='wo_left_right/best_model.pth',
        data_path=args.data_path,
        sample_index=args.index
    )