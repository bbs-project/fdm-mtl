import os
import shutil
import random
from pathlib import Path
from collections import Counter
import yaml
import math
from tqdm import tqdm

def balance_dataset():
    """
    Balances the train/validation split for classes with a validation ratio
    less than 20%. It moves a calculated number of files (image, det_label, cls_label)
    from the train set to the validation set to achieve an 8:2 ratio.
    """
    print("--- Starting Dataset Balancing Script ---")

    # 1. Load data configuration
    data_yaml_path = 'halibut-mtl/data.yaml'
    try:
        with open(data_yaml_path, 'r') as f:
            data_cfg = yaml.safe_load(f)
        
        base_path = Path(data_cfg['path'])
        img_dir_name = 'images'
        det_label_dir_name = data_cfg.get('det_label_dir')
        cls_label_dir_name = data_cfg.get('cls_label_dir')
        class_names = data_cfg.get('names_cls', {})

        if not all([det_label_dir_name, cls_label_dir_name]):
            print("❌ 'det_label_dir' or 'cls_label_dir' not found in data.yaml")
            return

        # Define all necessary paths
        paths = {
            'img_train': base_path / img_dir_name / 'train',
            'img_val': base_path / img_dir_name / 'val',
            'det_train': base_path / det_label_dir_name / 'train',
            'det_val': base_path / det_label_dir_name / 'val',
            'cls_train': base_path / cls_label_dir_name / 'train',
            'cls_val': base_path / cls_label_dir_name / 'val',
        }

        for p in paths.values():
            p.mkdir(parents=True, exist_ok=True)

    except Exception as e:
        print(f"❌ Error loading {data_yaml_path} or parsing paths: {e}")
        return

    # 2. Get all label files and their class IDs for both splits
    def get_class_files(directory: Path):
        class_files = {}
        files = list(directory.glob('*.txt'))
        print(f"🔍 Found {len(files)} label files in {directory}")
        for label_file in tqdm(files, desc=f"Reading labels from {directory.name}"):
            try:
                with open(label_file, 'r') as f:
                    class_id = int(f.read().strip())
                    if class_id not in class_files:
                        class_files[class_id] = []
                    class_files[class_id].append(label_file)
            except Exception:
                continue # Ignore parsing errors
        return class_files

    train_class_files = get_class_files(paths['cls_train'])
    val_class_files = get_class_files(paths['cls_val'])
    
    all_class_ids = sorted(list(set(train_class_files.keys()) | set(val_class_files.keys())))

    print("\n--- Analyzing Ratios and Planning Moves ---")
    
    files_to_move = []

    for cid in all_class_ids:
        train_files = train_class_files.get(cid, [])
        val_files = val_class_files.get(cid, [])
        train_count = len(train_files)
        val_count = len(val_files)
        total_count = train_count + val_count

        if total_count == 0:
            continue

        current_val_ratio = val_count / total_count
        target_val_ratio = 0.2

        # If validation set is under-represented
        if current_val_ratio < target_val_ratio:
            target_val_count = math.ceil(total_count * target_val_ratio)
            num_to_move = target_val_count - val_count
            
            if num_to_move > 0 and train_count >= num_to_move:
                class_name = class_names.get(cid, 'Unknown')
                print(f"  - Class {cid} ({class_name}): Needs balancing. Moving {num_to_move} files from train to val.")
                
                # Randomly select files to move
                selected_files = random.sample(train_files, num_to_move)
                files_to_move.extend(selected_files)
            elif train_count < num_to_move:
                 print(f"  - Class {cid}: Not enough train files ({train_count}) to move {num_to_move}.")

    if not files_to_move:
        print("\n✅ All classes are already balanced. No files to move.")
        return

    print(f"\n--- Moving {len(files_to_move)} File Sets ---")

    for cls_label_file in tqdm(files_to_move, desc="Moving files"):
        file_stem = cls_label_file.stem
        
        # Define source and destination for all three file types
        sources = {
            'img': paths['img_train'] / f"{file_stem}.jpg", # Assuming .jpg, might need to be more robust
            'det': paths['det_train'] / f"{file_stem}.txt",
            'cls': cls_label_file
        }
        dests = {
            'img': paths['img_val'] / f"{file_stem}.jpg",
            'det': paths['det_val'] / f"{file_stem}.txt",
            'cls': paths['cls_val'] / cls_label_file.name
        }

        # Move all three files
        moved_all = True
        for key in ['cls', 'det', 'img']:
            # Special check for image extension
            if key == 'img':
                # Find the correct image extension (.png, .jpeg, etc.)
                potential_img_files = list(paths['img_train'].glob(f"{file_stem}.*"))
                if not potential_img_files:
                    print(f"  - ⚠️ WARNING: Image for {file_stem} not found. Skipping this set.")
                    moved_all = False
                    break
                sources['img'] = potential_img_files[0]
                dests['img'] = paths['img_val'] / sources['img'].name

            if sources[key].exists():
                shutil.move(str(sources[key]), str(dests[key]))
            else:
                print(f"  - ⚠️ WARNING: Source file {sources[key]} not found. Skipping.")
                moved_all = False
                break
    
    print("\n--- Dataset Balancing Finished ---")
    print("💡 It's recommended to delete the old .cache files in your dataset directories before re-training.")


if __name__ == '__main__':
    balance_dataset()
