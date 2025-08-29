import os
import sys

import torch
from ultralytics.utils import checks
# No need to import BaseTrainer directly if we are just checking the path
# from ultralytics.engine.trainer import BaseTrainer 

# Our custom modules
from multi_task_trainer import MultiTaskTrainer

def debug_pipeline():
    """
    Debugs the entire pipeline by initializing the trainer and running the training.
    """
    print("--- Starting Debug Session for the Training Pipeline ---")
    checks.check_requirements("ultralytics")

    # Removed: print(f"DEBUG: BaseTrainer loaded from: {BaseTrainer.__file__}") # Print BaseTrainer path

    # 1. Define arguments for the trainer
    # These arguments are normally passed via CLI.
    # The 'model' and 'data' keys are essential for the trainer to initialize.
    override_args = {
        'model': '/home/user/fdm/halibut-mtl/halibut-mtl.yaml',
        'data': '/home/user/fdm/halibut-mtl/data.yaml',
        'epochs': 1,       # Run for a few epochs for testing
        'batch': 64,        # Use a small batch size
        'imgsz': 640,       # Use a small image size for faster debugging
        'workers': 0,      # Disable multiprocessing for data loading to debug
        'patience': 100,   # Disable early stopping for this debug run
        'plots': False,    # Disable plotting to avoid plotting errors and focus on core issues
        'device': 0,  # Use CPU for debugging
    }

    # 2. Initialize our custom trainer 
    print("\nStep 1: Initializing MultiTaskTrainer...")
    try:
        # The BaseTrainer will use the 'model' and 'data' args from overrides
        # to set up the model and dataset paths.
        trainer = MultiTaskTrainer(overrides=override_args)
        print("✅ Trainer initialized successfully.")
    except Exception as e:
        print(f"❌ Error during trainer initialization: {e}")
        raise

    # 3. Start the training process
    # The trainer.train() method will orchestrate everything based on our overrides.
    print("\nStep 2: Starting the training process...")
    try:
        trainer.train()
        print("\n✅ Training process completed. Exiting before validation.")
        sys.exit()  # Exit after training to focus on loss debugging
        print("\n✅ Training process completed a few epochs without critical errors.")
    except Exception as e:
        print(f"❌ Error during the training process: {e}")
        raise
        
    print("\n--- Debug Session Finished ---")


if __name__ == '__main__':
    debug_pipeline()
