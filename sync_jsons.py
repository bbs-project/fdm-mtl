import os
import shutil
from pathlib import Path

def sync_and_verify_json_files():
    """
    Synchronizes JSON files to match image files, moves orphaned JSONs, and verifies the final state.
    1. Moves orphaned JSONs (without a matching image) to a backup directory.
    2. Synchronizes remaining JSON files to the correct train/val directory.
    3. Verifies that the JSON and image splits are perfectly matched.
    """
    base_dir = Path("/home/user/fdm/halibut-mtl/datasets")
    images_dir = base_dir / "images"
    jsons_dir = base_dir / "jsons"
    orphaned_dir = base_dir / "jsons_orphaned"

    # --- 0. Setup ---
    # Create the directory for orphaned JSONs if it doesn't exist
    orphaned_dir.mkdir(exist_ok=True)

    # Check if essential directories exist
    if not all([images_dir.exists(), jsons_dir.exists(), (images_dir / "train").exists(),
                (images_dir / "val").exists(), (jsons_dir / "train").exists(), (jsons_dir / "val").exists()]):
        print("Error: One or more required directories do not exist.")
        return

    print("Starting synchronization and verification...")
    print(f"Orphaned JSONs will be moved to: {orphaned_dir}\n")

    # --- 1. Identify all image basenames ---
    image_train_basenames = {f.stem for f in (images_dir / "train").iterdir() if f.is_file()}
    image_val_basenames = {f.stem for f in (images_dir / "val").iterdir() if f.is_file()}
    all_image_basenames = image_train_basenames.union(image_val_basenames)

    print(f"Found {len(image_train_basenames)} images in train set.")
    print(f"Found {len(image_val_basenames)} images in val set.")
    print(f"Total unique images: {len(all_image_basenames)}\n")

    # --- 2. Find and move orphaned JSONs ---
    moved_orphans = 0
    print("Searching for orphaned JSON files...")
    all_json_files = list((jsons_dir / "train").glob("*.json")) + list((jsons_dir / "val").glob("*.json"))

    for json_file in all_json_files:
        if json_file.stem not in all_image_basenames:
            destination_path = orphaned_dir / json_file.name
            print(f"  - Moving orphan '{json_file.name}' to {orphaned_dir}")
            shutil.move(json_file, destination_path)
            moved_orphans += 1
    
    print(f"Moved {moved_orphans} orphaned JSON files.\n")

    # --- 3. Synchronize remaining JSONs between train and val ---
    moved_from_train_to_val = 0
    moved_from_val_to_train = 0
    
    # Sync jsons/train
    print("Checking 'jsons/train' for synchronization...")
    json_train_dir = jsons_dir / "train"
    json_val_dir = jsons_dir / "val"

    for json_file in list(json_train_dir.glob("*.json")): # Use list to prevent issues while iterating and moving
        basename = json_file.stem
        if basename not in image_train_basenames and basename in image_val_basenames:
            destination_path = json_val_dir / json_file.name
            print(f"  - Moving '{json_file.name}' from train to val...")
            shutil.move(json_file, destination_path)
            moved_from_train_to_val += 1

    # Sync jsons/val
    print("\nChecking 'jsons/val' for synchronization...")
    for json_file in list(json_val_dir.glob("*.json")): # Use list here as well
        basename = json_file.stem
        if basename not in image_val_basenames and basename in image_train_basenames:
            destination_path = json_train_dir / json_file.name
            print(f"  - Moving '{json_file.name}' from val to train...")
            shutil.move(json_file, destination_path)
            moved_from_val_to_train += 1
            
    print("\n--- Synchronization Summary ---")
    print(f"Moved from train to val: {moved_from_train_to_val} files.")
    print(f"Moved from val to train: {moved_from_val_to_train} files.")
    print(f"Moved orphaned files: {moved_orphans} files.\n")

    # --- 4. Verification Step ---
    print("--- Starting Verification ---")
    final_json_train_basenames = {f.stem for f in (jsons_dir / "train").iterdir()}
    final_json_val_basenames = {f.stem for f in (jsons_dir / "val").iterdir()}

    train_mismatch = final_json_train_basenames.symmetric_difference(image_train_basenames)
    val_mismatch = final_json_val_basenames.symmetric_difference(image_val_basenames)

    if not train_mismatch and not val_mismatch:
        print("✅ Verification successful! All JSON and image files are perfectly matched.")
        print(f"  - Train sets match: {len(final_json_train_basenames)} JSONs and {len(image_train_basenames)} images.")
        print(f"  - Val sets match: {len(final_json_val_basenames)} JSONs and {len(image_val_basenames)} images.")
    else:
        print("❌ Verification failed! Mismatches found.")
        if train_mismatch:
            print(f"  - Mismatches in train set ({len(train_mismatch)} files): {train_mismatch}")
        if val_mismatch:
            print(f"  - Mismatches in val set ({len(val_mismatch)} files): {val_mismatch}")
            
    print("\nProcess complete.")

if __name__ == "__main__":
    sync_and_verify_json_files()