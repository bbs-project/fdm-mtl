import argparse
import sys
import os
from pathlib import Path

# Add project root to Python path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # halibut-mtl directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ultralytics.utils import SETTINGS, checks
from mtl.multi_task_trainer import MultiTaskTrainer

def main():
    """
    Main function to start the multi-task training process.
    Parses command-line arguments and initializes and runs the MultiTaskTrainer.
    """
    # --- Set PYTHONPATH for DDP ---
    # This ensures that subprocesses spawned by DDP can find our custom modules.
    python_path = os.environ.get('PYTHONPATH', '')
    if str(ROOT) not in python_path:
        os.environ['PYTHONPATH'] = f"{str(ROOT)}:{python_path}"

    # --- Argument Parser ---
    parser = argparse.ArgumentParser(description="Train a multi-task YOLO model.")
    parser.add_argument('--model', type=str, default='./yaml/model/original.yaml', help='Path to the model configuration file (.yaml).')
    parser.add_argument('--weights', type=str, default='', help='Path to the pretrained weights file (.pt). Leave empty to train from scratch.')
    parser.add_argument('--data', type=str, default='./yaml/data/original.yaml', help='Path to the data configuration file.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch', type=int, default=64, help='Batch size.')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size.')
    parser.add_argument('--device', type=str, default='0,1', help="Device to use for training, e.g., '0' or '0,1,2,3' or 'cpu'.")
    parser.add_argument('--patience', type=int, default=100, help='Patience for early stopping.')
    parser.add_argument('--plots', default=True, help='Enable plotting of training results.')
    parser.add_argument('--name', type=str, default='mtl', help='Name of the training run for logging.')
    
    args = parser.parse_args()

    print("--- Starting Multi-Task Training Session ---")
    checks.check_requirements("ultralytics")

    # --- Prepare Overrides ---
    # Convert argparse.Namespace to a dictionary for the trainer
    override_args = vars(args)
    override_args['project'] = 'halibut'

    # --- Initialize and Run Trainer ---
    print("\nStep 1: Initializing MultiTaskTrainer...")
    try:
        trainer = MultiTaskTrainer(overrides=override_args)
        print("✅ Trainer initialized successfully.")
    except Exception as e:
        print(f"❌ Error during trainer initialization: {e}")
        raise

    print("\nStep 2: Starting the training process...")
    try:
        trainer.train()
        print("\n✅ Training process completed successfully.")
    except Exception as e:
        print(f"❌ Error during the training process: {e}")
        raise
        
    print("\n--- Training Session Finished ---")


if __name__ == '__main__':
    main()