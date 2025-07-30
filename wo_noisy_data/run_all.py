import os
import sys
import json
import csv
from datetime import datetime


# --- Helper function for user input ---
def get_user_input():
    """
    Prompts the user for attack parameters and validates the input.
    """
    attack_model_epochs = None  # Default to None

    # --- Get Attack Model Choice ---
    while True:
        print("\nChoose an attack model architecture:")
        print("  1: RandomForestClassifier (Default)")
        print("  2: LogisticRegression")
        print("  3: MLPClassifier (Simple Neural Network)")
        choice = input("Enter your choice (1, 2, or 3): ")
        if choice == '1':
            attack_model_name = 'RandomForest'
            break
        elif choice == '2':
            attack_model_name = 'LogisticRegression'
            break
        elif choice == '3':
            attack_model_name = 'MLP'
            # --- Conditionally ask for epochs ---
            while True:
                try:
                    epochs_str = input("Enter number of training epochs for MLP (e.g., 500): ")
                    attack_model_epochs = int(epochs_str)
                    if attack_model_epochs <= 0:
                        print("Error: Please enter a positive number of epochs.")
                        continue
                    break
                except ValueError:
                    print("Error: Invalid input. Please enter an integer.")
            break
        else:
            print("Error: Invalid choice. Please enter 1, 2, or 3.")

    # --- Get Subset Percentage ---
    while True:
        try:
            subset_percent_str = input(
                "Enter the percentage of the dataset to use for shadow models (e.g., 80 for 80%): ")
            subset_percent = float(subset_percent_str)
            if not 0 < subset_percent <= 100:
                print("Error: Please enter a number between 1 and 100.")
                continue
            subset_fraction = subset_percent / 100.0
            break
        except ValueError:
            print("Error: Invalid input. Please enter a number.")

    # --- Get Number of Shadow Models ---
    while True:
        try:
            num_shadows_str = input("Enter the number of shadow models to train (e.g., 16): ")
            num_shadows = int(num_shadows_str)
            if num_shadows <= 0:
                print("Error: Please enter a positive number of shadow models.")
                continue
            break
        except ValueError:
            print("Error: Invalid input. Please enter an integer.")

    notes = input("Any additional notes go here: ")


    return subset_fraction, num_shadows, attack_model_name, attack_model_epochs, notes


def check_required_files():
    """Checks if all necessary files for the attack exist in the current directory."""
    required_files = ['../victim_model.pth', '../attack_on_victim_dataset.npz']
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print("\n--- FATAL ERROR: Missing Required Files ---")
        for f in missing_files:
            print(f"  - Could not find '{f}' in the current directory.")
        print("\nPlease make sure these files are present before running the script.")
        return False

    print("\n\u2713 All required files found.")
    return True


def save_results_to_csv(params, results):
    """Saves the parameters and results of a run to a CSV file."""
    filepath = 'attack_results.csv'
    file_exists = os.path.isfile(filepath)

    fieldnames = ['timestamp', 'attack_model', 'attack_model_epochs', 'num_shadow_models', 'subset_fraction',
                  'attack_accuracy', 'attack_auc', 'notes']

    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        row_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'attack_model': params['attack_architecture'],
            'attack_model_epochs': params.get('attack_model_epochs', 'N/A'),  # Use .get for safety
            'num_shadow_models': params['num_shadows'],
            'subset_fraction': params['subset_fraction'],
            'attack_accuracy': results['accuracy'],
            'attack_auc': results['auc'],
            'notes': params['notes']
        }
        writer.writerow(row_data)

    print(f"\n\u2713 Results saved to '{filepath}'")


# --- Main Pipeline Execution ---
def run_attack_pipeline():
    """Executes the attack phase of the membership inference pipeline."""
    print("========================================================")
    print("=== MEMBERSHIP INFERENCE ATTACK RUNNER               ===")
    print("========================================================")

    if not check_required_files():
        sys.exit(1)

    subset_fraction, num_shadows, attack_model_name, attack_model_epochs, notes = get_user_input()

    params_path = 'params.json'
    print(f"\nUpdating '{params_path}' with your configuration...")
    try:
        with open(params_path, 'r') as f:
            params = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        params = {"shadow_architecture": "CNN_LSTM_Model"}

    params['subset_size'] = subset_fraction
    params['num_shadows'] = num_shadows
    params['attack_architecture'] = attack_model_name
    if attack_model_epochs is not None:
        params['attack_model_epochs'] = attack_model_epochs
    elif 'attack_model_epochs' in params:
        del params['attack_model_epochs']  # Clean up old param if not used

    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"  - Set attack model to: {attack_model_name}")
    if attack_model_epochs is not None:
        print(f"  - Set MLP epochs to: {attack_model_epochs}")
    print(f"  - Set dataset subset to: {subset_fraction * 100:.1f}%")
    print(f"  - Set number of shadow models to: {num_shadows}")

    try:
        import shadow_models as shadow_model_script
        import attack_train
        import attack_execution as attack_model_target
    except ImportError as e:
        print(f"\n--- FATAL ERROR: Could not import a required script: {e} ---")
        sys.exit(1)

    print("\n--- [PHASE 1/3] Generating Shadow Model Attack Data ---")
    try:
        shadow_model_script.main()
        print("\n--- [PHASE 1/3] Shadow Data Generation COMPLETE ---")
    except Exception as e:
        print(f"\n--- ERROR in Phase 1 (Shadow Data Generation): {e} ---")
        sys.exit(1)

    print("\n--- [PHASE 2/3] Training the Attack Model ---")
    try:
        attack_train.main()
        print("\n--- [PHASE 2/3] Attack Model Training COMPLETE ---")
    except Exception as e:
        print(f"\n--- ERROR in Phase 2 (Attack Model Training): {e} ---")
        sys.exit(1)

    final_results = None
    print("\n--- [PHASE 3/3] Executing Attack on Target Model ---")
    try:
        accuracy, auc = attack_model_target.main()
        final_results = {'accuracy': accuracy, 'auc': auc}
        print("\n--- [PHASE 3/3] Attack Execution COMPLETE ---")
    except Exception as e:
        print(f"\n--- ERROR in Phase 3 (Attack Execution): {e} ---")
        sys.exit(1)

    if final_results:
        run_params = {
            'num_shadows': num_shadows,
            'subset_fraction': subset_fraction,
            'attack_architecture': attack_model_name,
            'attack_model_epochs': attack_model_epochs,
            'notes':notes
        }
        save_results_to_csv(run_params, final_results)

    print("\n========================================================")
    print("=== PIPELINE FINISHED SUCCESSFULLY                   ===")
    print("========================================================")


if __name__ == "__main__":
    run_attack_pipeline()
