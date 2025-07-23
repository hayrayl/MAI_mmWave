import os
import sys

# --- SCRIPT CONFIGURATION ---
# Add the necessary directories to the Python path to ensure modules can be found.
# This is important if your scripts are in subdirectories.
# You may need to adjust these paths based on your project structure.
# For example, if 'mmWaveHAR4' and 'shadow_models' are in the same parent directory.
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    # Import the main functions from your existing scripts
    import train_model_helper
    # Note: The refactored script is assumed to be named SM_data_generation.py
    import shadow_models.SM_data_generation_test as SM_data
    import attack_train
    import attack_model_target
except ImportError as e:
    print(f"Error: Could not import a required module: {e}")
    print("Please ensure that all required scripts (train_model_helper.py, SM_data_generation.py, etc.)")
    print("are in the correct directory and that your Python path is set up properly.")
    sys.exit(1)


def run_pipeline():
    """
    Executes the full membership inference attack pipeline step-by-step.
    """
    print("========================================================")
    print("=== STARTING MEMBERSHIP INFERENCE ATTACK PIPELINE    ===")
    print("========================================================")

    # --- Step 1: Train the Target Model (or skip if already trained) ---
    # This simulates the victim model being trained on its private data.
    # It produces 'target_model.pth' and 'attack_on_target_dataset.npz'.
    target_model_path = 'target_model.pth'
    target_dataset_path = 'attack_on_target_dataset.npz'

    print("\n--- [PHASE 1/4] Preparing Target Model ---")
    if os.path.exists(target_model_path) and os.path.exists(target_dataset_path):
        print(f"✓ Found existing target model ('{target_model_path}') and dataset.")
        print("✓ Skipping Phase 1.")
    else:
        print("Target model not found. Starting training...")
        try:
            train_model_helper.main()
            print("\n--- [PHASE 1/4] Target Model Training COMPLETE ---")
        except Exception as e:
            print(f"\n--- ERROR in Phase 1 (Target Model Training): {e} ---")
            sys.exit(1)


    # --- Step 2: Generate Shadow Model Data ---
    # This trains several "shadow" models to learn the general behavior of
    # members vs. non-members. It produces 'shadow_dataset.npz'.
    print("\n--- [PHASE 2/4] Generating Shadow Model Attack Data ---")


    try:
        # Assuming the refactored script is named SM_data_generation.py
        SM_data.main()
        print("\n--- [PHASE 2/4] Shadow Data Generation COMPLETE ---")
    except Exception as e:
        print(f"\n--- ERROR in Phase 2 (Shadow Data Generation): {e} ---")
        sys.exit(1)

    # --- Step 3: Train the Attack Model ---
    # This uses the shadow data to train a classifier that can distinguish
    # between member and non-member softmax outputs. It produces 'attack_model.pkl'.
    print("\n--- [PHASE 3/4] Training the Attack Model ---")
    try:
        attack_train.main()
        print("\n--- [PHASE 3/4] Attack Model Training COMPLETE ---")
    except Exception as e:
        print(f"\n--- ERROR in Phase 3 (Attack Model Training): {e} ---")
        sys.exit(1)

    # --- Step 4: Execute the Attack on the Target Model ---
    # This uses the trained attack model to predict membership on the
    # original target model's outputs and reports the final score.
    print("\n--- [PHASE 4/4] Executing Attack on Target Model ---")
    try:
        attack_model_target.main()
        print("\n--- [PHASE 4/4] Attack Execution COMPLETE ---")
    except Exception as e:
        print(f"\n--- ERROR in Phase 4 (Attack Execution): {e} ---")
        sys.exit(1)

    print("\n========================================================")
    print("=== PIPELINE FINISHED SUCCESSFULLY                   ===")
    print("========================================================")


if __name__ == "__main__":
    run_pipeline()
