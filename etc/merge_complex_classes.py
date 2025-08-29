import os
import yaml
from pathlib import Path
import shutil
from collections import Counter

def merge_complex_classes():
    """
    Merges classification labels to reduce class imbalance.

    This script performs the following actions:
    1.  Defines a new, simplified class structure where complex diseases
        (containing '_') are merged into a single 'complex' class.
    2.  Creates a new label directory `cls_labels_merged` to store the updated labels,
        preserving the original data.
    3.  Iterates through the original `train` and `val` label files.
    4.  For each label, it checks if the original class name corresponds to a
        simple disease or a complex one.
    5.  It writes a new label file in `cls_labels_merged` with the updated class ID.
    6.  Generates a `data_merged.yaml` file reflecting the new class structure,
        including the updated class count (`nc_cls`) and names (`names_cls`).
    7.  Prints a summary of the class distribution before and after the merge.
    """
    print("--- Starting Class Merge Process ---")

    # 1. Load original data configuration
    original_data_yaml = 'halibut-mtl/data.yaml'
    try:
        with open(original_data_yaml, 'r') as f:
            data_cfg = yaml.safe_load(f)
        original_names_cls = data_cfg.get('names_cls', {})
        base_path = Path(data_cfg['path'])
        cls_label_dir = base_path / data_cfg.get('cls_label_dir', 'cls_labels')
    except Exception as e:
        print(f"❌ Error: Could not read or parse {original_data_yaml}. {e}")
        return

    # 2. Define the new class structure
    simple_diseases = [
        'normal', 'etc', 'viral_hemorrhagic_septicemia', 'emaciation', 
        'vibrio', 'scuticociliatosis', 'kudoa', 'edwardsiella', 
        'lymphocystis', 'streptococcus'
    ]
    
    # Create new class mapping
    new_names_cls = {i: name for i, name in enumerate(simple_diseases)}
    complex_class_id = len(simple_diseases)
    new_names_cls[complex_class_id] = 'complex'
    
    print("\n✨ New Class Structure Defined:")
    for i, name in new_names_cls.items():
        print(f"  - ID {i}: {name}")

    # Create a reverse map for old names to new IDs
    old_name_to_new_id = {}
    for old_id, old_name in original_names_cls.items():
        if old_name in simple_diseases:
            # Find the new ID for this simple disease
            for new_id, new_name in new_names_cls.items():
                if old_name == new_name:
                    old_name_to_new_id[old_id] = new_id
                    break
        else:
            # It's a complex disease
            old_name_to_new_id[old_id] = complex_class_id

    # 3. Set up new directory for merged labels
    merged_cls_label_dir = base_path / 'cls_labels_merged'
    if merged_cls_label_dir.exists():
        print(f"\n⚠️  '{merged_cls_label_dir}' already exists. Deleting it to ensure a fresh start.")
        shutil.rmtree(merged_cls_label_dir)
    
    print(f"Creating new directory for merged labels at: {merged_cls_label_dir}")
    merged_cls_label_dir.mkdir(parents=True, exist_ok=True)

    # 4. Process labels and print distribution
    original_distribution = {}
    new_distribution = {}

    for split in ['train', 'val']:
        print(f"\n🔄 Processing '{split}' split...")
        original_split_path = cls_label_dir / split
        merged_split_path = merged_cls_label_dir / split
        merged_split_path.mkdir(exist_ok=True)

        if not original_split_path.exists():
            print(f"  - Skipping '{split}': Directory not found at {original_split_path}")
            continue

        label_files = list(original_split_path.glob('*.txt'))
        
        original_counts = Counter()
        new_counts = Counter()

        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    original_class_id = int(f.read().strip())
                
                original_counts[original_class_id] += 1
                
                new_class_id = old_name_to_new_id.get(original_class_id)

                if new_class_id is not None:
                    new_counts[new_class_id] += 1
                    # Write the new label file
                    new_label_path = merged_split_path / label_file.name
                    with open(new_label_path, 'w') as f_out:
                        f_out.write(str(new_class_id))
                else:
                    print(f"  - Warning: Could not find a mapping for original class ID {original_class_id}. Skipping file {label_file.name}")

            except Exception as e:
                print(f"  - Warning: Could not process file {label_file}: {e}")
        
        original_distribution[split] = original_counts
        new_distribution[split] = new_counts
        print(f"  - Successfully processed {len(label_files)} files.")

    # 5. Print distribution comparison
    print("\n--- Class Distribution Comparison ---")
    print("\n--- BEFORE MERGE ---")
    for split, counts in original_distribution.items():
        print(f"\n[{split.upper()} SET]")
        print(f"{'ID':<5} {'Name':<45} {'Count':<10}")
        print("-" * 60)
        for class_id, count in sorted(counts.items()):
            print(f"{class_id:<5} {original_names_cls.get(class_id, 'Unknown'):<45} {count:<10}")

    print("\n--- AFTER MERGE ---")
    for split, counts in new_distribution.items():
        print(f"\n[{split.upper()} SET]")
        print(f"{'ID':<5} {'Name':<45} {'Count':<10}")
        print("-" * 60)
        for class_id, count in sorted(counts.items()):
            print(f"{class_id:<5} {new_names_cls.get(class_id, 'Unknown'):<45} {count:<10}")

    # 6. Create new data.yaml
    new_data_yaml_path = 'halibut-mtl/data_merged.yaml'
    data_cfg['cls_label_dir'] = 'cls_labels_merged' # Update the label directory
    data_cfg['names_cls'] = new_names_cls
    # The model's nc should be updated in the model yaml, but we set it here for consistency
    # data_cfg['nc_cls'] = len(new_names_cls) 

    try:
        with open(new_data_yaml_path, 'w') as f:
            yaml.dump(data_cfg, f, sort_keys=False, indent=2)
        print(f"\n✅ Successfully created merged data configuration at: {new_data_yaml_path}")
        print("You should now use this YAML file for training with the merged dataset.")
    except Exception as e:
        print(f"❌ Error: Could not write {new_data_yaml_path}. {e}")

    print("\n--- Process Complete ---")


if __name__ == '__main__':
    merge_complex_classes()
