import os
from pathlib import Path
import yaml
from tqdm import tqdm

def clean_unmatched_files():
    """
    Cleans the dataset by ensuring that for every image file in the train/val
    directories, a corresponding detection label and classification label exist.
    If either label is missing, it deletes all files associated with that stem.
    """
    print("--- Starting Dataset Cleaning Script ---")

    # 1. Load data configuration
    data_yaml_path = 'halibut-mtl/data.yaml'
    try:
        with open(data_yaml_path, 'r') as f:
            data_cfg = yaml.safe_load(f)
        
        base_path = Path(data_cfg['path'])
        img_dir_name = 'images'
        det_label_dir_name = data_cfg.get('det_label_dir')
        cls_label_dir_name = data_cfg.get('cls_label_dir')

        if not all([det_label_dir_name, cls_label_dir_name]):
            print("❌ 'det_label_dir' or 'cls_label_dir' not found in data.yaml")
            return

    except Exception as e:
        print(f"❌ Error loading {data_yaml_path} or parsing paths: {e}")
        return

    # 2. Iterate through train and val splits
    for split in ['train', 'val']:
        print(f"\n--- Processing '{split}' split ---")
        
        img_path = base_path / img_dir_name / split
        det_path = base_path / det_label_dir_name / split
        cls_path = base_path / cls_label_dir_name / split

        if not img_path.exists():
            print(f"  - Image directory not found: {img_path}")
            continue

        image_files = list(img_path.glob('*.*')) # Get all files regardless of extension
        
        if not image_files:
            print(f"  - No images found in {img_path}")
            continue
            
        print(f"🔍 Found {len(image_files)} images. Checking for matching labels...")
        
        files_to_delete = []

        for img_file in tqdm(image_files, desc=f"Verifying {split} files"):
            file_stem = img_file.stem
            
            det_label_file = det_path / f"{file_stem}.txt"
            cls_label_file = cls_path / f"{file_stem}.txt"

            # Check for missing label files
            if not det_label_file.exists() or not cls_label_file.exists():
                print(f"\n  - Mismatch found for stem: {file_stem}")
                if not det_label_file.exists():
                    print(f"    - Missing detection label: {det_label_file}")
                if not cls_label_file.exists():
                    print(f"    - Missing classification label: {cls_label_file}")
                
                # Add all associated files to the deletion list
                files_to_delete.append(img_file)
                if det_label_file.exists():
                    files_to_delete.append(det_label_file)
                if cls_label_file.exists():
                    files_to_delete.append(cls_label_file)

        if not files_to_delete:
            print("✅ No unmatched files found. Dataset is clean.")
            continue

        print(f"\n--- Deleting {len(files_to_delete)} Unmatched Files in '{split}' split ---")
        
        # Perform deletion
        for f in tqdm(files_to_delete, desc=f"Deleting {split} files"):
            try:
                f.unlink()
            except Exception as e:
                print(f"  - ❌ Error deleting {f}: {e}")

    print("\n--- Dataset Cleaning Finished ---")
    print("💡 It's recommended to delete the old .cache files before re-training.")

if __name__ == '__main__':
    clean_unmatched_files()
