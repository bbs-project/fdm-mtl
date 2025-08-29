import os
from pathlib import Path
from collections import Counter
import yaml

def check_class_distribution():
    """
    Analyzes the distribution of classes in the classification dataset.
    It reads all label files from the train and validation cls_labels directories
    and prints the count for each class.
    """
    input_dir = input("Enter the yaml file include dataset info to analyze (default: data.yaml): ")
    print("--- Analyzing Class Distribution ---")

    # 1. Load data configuration to get paths
    data_yaml_path = '/home/user/fdm/halibut-mtl/yaml/data/' + input_dir
    try:
        with open(data_yaml_path, 'r') as f:
            data_cfg = yaml.safe_load(f)
        
        base_path = Path(data_cfg['path'])
        cls_label_dir_name = data_cfg.get('cls_label_dir')
        if not cls_label_dir_name:
            print(f"❌ 'cls_label_dir' not found in {input_dir}. Please check the YAML file.")
            return
            
        cls_label_path = base_path / cls_label_dir_name
        train_path = cls_label_path / 'train'
        val_path = cls_label_path / 'val'
        class_names = data_cfg.get('names_cls', {})

    except Exception as e:
        print(f"❌ Error loading {data_yaml_path} or parsing paths: {e}")
        return

    # 2. Function to count labels in a directory
    def count_labels(directory: Path, split_name: str):
        print(f"\n🔍 Counting labels in: {directory} ({split_name} set)")
        if not directory.exists():
            print(f"  - Directory does not exist.")
            return None
        
        label_files = list(directory.glob('*.txt'))
        if not label_files:
            print("  - No label files found.")
            return None

        counts = Counter()
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    class_index = int(f.read().strip())
                    counts[class_index] += 1
            except Exception as e:
                print(f"  - Warning: Could not read or parse {label_file}: {e}")
        
        return counts

    # 3. Count for both train and validation sets
    train_counts = count_labels(train_path, 'train')
    val_counts = count_labels(val_path, 'val')

    # 4. Print results in a sorted table
    print("\n--- Class Distribution Summary ---")
    
    all_classes = sorted(list(set(train_counts.keys() if train_counts else []) | set(val_counts.keys() if val_counts else [])))

    if not all_classes:
        print("No class data found to summarize.")
        return

    # Header
    print(f"{'Class ID':<10} {'Class Name':<55} {'Train Count':<15} {'Val Count':<15} {'Proportion':<15}")
    print("-" * 105)

    train_sum = 0

    for class_id in all_classes:
        train_count = train_counts.get(class_id, 0) if train_counts else 0
        train_sum += train_count

    for class_id in all_classes:
        class_name = class_names.get(class_id, 'Unknown')
        train_count = train_counts.get(class_id, 0) if train_counts else 0
        val_count = val_counts.get(class_id, 0) if val_counts else 0
        prop = train_count / train_sum if train_sum > 0 else 0
        print(f"{class_id:<10} {class_name:<55} {train_count:<15} {val_count:<15} {prop:<15.2f}")
        
    print("-" * 105)
    
    if train_counts:
        print(f"\nTotal Train Labels: {sum(train_counts.values())}")
    if val_counts:
        print(f"Total Val Labels:   {sum(val_counts.values())}")


if __name__ == '__main__':
    check_class_distribution()
