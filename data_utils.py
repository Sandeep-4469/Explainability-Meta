# data_utils.py
import os
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from tqdm.notebook import tqdm # Use standard tqdm if not in notebook

import config # Import configuration

def parse_cub_metadata(data_dir):
    """Reads CUB metadata files into pandas DataFrames."""
    images_path = os.path.join(data_dir, 'images.txt')
    labels_path = os.path.join(data_dir, 'image_class_labels.txt')
    split_path = os.path.join(data_dir, 'train_test_split.txt')
    bbox_path = os.path.join(data_dir, 'bounding_boxes.txt')
    classes_path = os.path.join(data_dir, 'classes.txt')

    required_files = [images_path, labels_path, split_path, bbox_path, classes_path]
    for f_path in required_files:
        if not os.path.exists(f_path):
            raise FileNotFoundError(f"Metadata file not found: {f_path}. Check DATA_DIR in config.py.")

    df_images = pd.read_csv(images_path, sep=' ', names=['img_id', 'filepath'])
    df_labels = pd.read_csv(labels_path, sep=' ', names=['img_id', 'class_id'])
    df_split = pd.read_csv(split_path, sep=' ', names=['img_id', 'is_training'])
    df_bboxes = pd.read_csv(bbox_path, sep=' ', names=['img_id', 'x', 'y', 'width', 'height'])
    df_classes = pd.read_csv(classes_path, sep=' ', names=['class_id', 'class_name'])

    df_labels['class_id'] = df_labels['class_id'] - 1 # 0-based indexing for class IDs
    df = df_images.merge(df_labels, on='img_id')
    df = df.merge(df_split, on='img_id')
    df = df.merge(df_bboxes, on='img_id')
    # Use IMAGES_DIR from config
    df['full_path'] = df['filepath'].apply(lambda x: os.path.join(config.IMAGES_DIR, x))
    class_id_to_name = df_classes.set_index('class_id')['class_name'].to_dict()
    # Adjust keys in map to be 0-based
    class_id_to_name = {(k - 1): v for k, v in class_id_to_name.items()}
    print(f"Parsed metadata for {len(df)} total image entries across {df['class_id'].nunique()} original classes.")
    return df, class_id_to_name

class CubDataset(Dataset):
    """Custom Dataset for CUB-200-2011 with cropping."""
    def __init__(self, df_subset, transform=None):
        self.df = df_subset
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_info = self.df.iloc[idx]
        img_path = img_info['full_path']
        label = img_info['subset_class_id'] # Use 0-based subset ID
        bbox = (img_info['x'], img_info['y'], img_info['width'], img_info['height'])

        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: Image file not found {img_path}. Skipping.")
            return None, -1 # Return None to indicate failure
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, -1

        x, y, w, h = bbox
        left, upper = int(np.floor(x)), int(np.floor(y))
        right, lower = int(np.ceil(x + w)), int(np.ceil(y + h))
        img_width, img_height = image.size

        # Clamp bounding box coordinates to image dimensions
        left, upper = max(0, left), max(0, upper)
        right, lower = min(img_width, right), min(img_height, lower)

        # Crop only if the bounding box is valid
        if right > left and lower > upper:
             image = image.crop((left, upper, right, lower))
        # else: print(f"Warning: Invalid bbox for {img_path}, using full image.") # Optional warning

        if self.transform:
            image = self.transform(image)

        return image, label


def prepare_cub_data_splits(data_dir, images_dir, image_size, n_way, k_shot, n_query, n_meta_train_ratio=0.7):
    """
    Loads CUB data, detects available classes, prepares meta-splits,
    and creates data lists for samplers.
    Adjusts n_way if not enough classes are available.
    Returns meta_train_data, meta_test_data, and the *potentially adjusted* n_way.
    """
    if not os.path.isdir(images_dir):
         raise FileNotFoundError(f"Images directory not found: {images_dir}. Check config.py (IMAGES_DIR).")

    df_all, class_id_to_name_map = parse_cub_metadata(data_dir)

    # Detect classes based on folders present in the images directory
    available_folders = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]
    available_original_class_ids = []
    for folder_name in available_folders:
        try:
            # Assuming folder names like '001.Black_footed_Albatross'
            class_num_str = folder_name.split('.')[0]
            original_id_one_based = int(class_num_str)
            # Store 0-based original ID
            available_original_class_ids.append(original_id_one_based - 1)
        except ValueError:
            print(f"Warning: Could not parse class ID from folder name: {folder_name}")
            continue # Ignore invalid folders

    available_original_class_ids = sorted(list(set(available_original_class_ids)))

    if not available_original_class_ids:
        raise ValueError("No valid class folders found in the images directory.")

    N_CLASSES_TOTAL_DETECTED = len(available_original_class_ids)
    print(f"\nDetected {N_CLASSES_TOTAL_DETECTED} classes based on folders in {images_dir}.")
    # Display first few detected classes for verification
    print("Sample Detected Classes (Original IDs):")
    for i, cid in enumerate(available_original_class_ids[:min(5, N_CLASSES_TOTAL_DETECTED)]):
         print(f"  ID {cid}: {class_id_to_name_map.get(cid, 'Unknown Name')}")
    if N_CLASSES_TOTAL_DETECTED > 5: print("  ...")

    # Filter the main DataFrame to include only detected classes
    df_subset = df_all[df_all['class_id'].isin(available_original_class_ids)].copy()

    # Create a new 0-based 'subset_class_id' for the detected classes
    subset_class_map = {orig_id: new_id for new_id, orig_id in enumerate(available_original_class_ids)}
    df_subset['subset_class_id'] = df_subset['class_id'].map(subset_class_map)

    # Define image transformations
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    # Split detected classes into meta-train and meta-test sets
    n_meta_train_actual = int(N_CLASSES_TOTAL_DETECTED * n_meta_train_ratio)
    n_meta_test_actual = N_CLASSES_TOTAL_DETECTED - n_meta_train_actual

    if n_meta_train_actual < n_way or n_meta_test_actual < n_way:
        print(f"Warning: Requested N_WAY ({n_way}) is too large for the detected classes split.")
        print(f"  Available for meta-train: {n_meta_train_actual}")
        print(f"  Available for meta-test: {n_meta_test_actual}")
        n_way = min(n_meta_train_actual, n_meta_test_actual)
        if n_way <= 0:
             raise ValueError(f"Cannot proceed with N_WAY <= 0 after adjustment. Need more classes or lower N_WAY.")
        print(f"Reducing N_WAY to {n_way}.")
    elif n_meta_train_actual == 0 or n_meta_test_actual == 0:
        raise ValueError(f"Cannot split {N_CLASSES_TOTAL_DETECTED} classes into train/test with ratio {n_meta_train_ratio}. Need more classes or adjust ratio.")

    print(f"Splitting detected classes: {n_meta_train_actual} for meta-train, {n_meta_test_actual} for meta-test.")

    # Get the list of 0-based subset indices
    all_subset_indices = list(range(N_CLASSES_TOTAL_DETECTED))
    random.shuffle(all_subset_indices)
    meta_train_subset_indices = sorted(all_subset_indices[:n_meta_train_actual])
    meta_test_subset_indices = sorted(all_subset_indices[n_meta_train_actual:])

    # Map subset indices back to original class IDs for printing clarity
    meta_train_classes_orig_ids = [available_original_class_ids[i] for i in meta_train_subset_indices]
    meta_test_classes_orig_ids = [available_original_class_ids[i] for i in meta_test_subset_indices]

    print(f"Using {len(meta_train_classes_orig_ids)} classes for meta-train (Orig IDs e.g., {meta_train_classes_orig_ids[:3]}...).")
    print(f"Using {len(meta_test_classes_orig_ids)} classes for meta-test (Orig IDs e.g., {meta_test_classes_orig_ids[:3]}...).")

    # Use CUB's predefined train/test split flags to assign images
    df_subset_train_cub = df_subset[df_subset['is_training'] == 1]
    df_subset_test_cub = df_subset[df_subset['is_training'] == 0]

    # --- Helper function to load images for a specific meta-split ---
    def load_data_for_split(target_subset_indices, df_source, split_name):
        data = [] # List of tuples: (subset_class_id, [list_of_images])
        pbar = tqdm(target_subset_indices, desc=f"Loading Meta-{split_name} Images")
        min_samples_needed = k_shot + n_query
        valid_classes_count = 0
        classes_with_insufficient_samples = []

        for subset_class_id in pbar: # Iterate over the 0..N_split-1 indices
            # Filter the source dataframe (e.g., df_subset_train_cub) by the SUBSET class ID
            class_df = df_source[df_source['subset_class_id'] == subset_class_id]

            if not class_df.empty:
                # Use CubDataset to handle loading and transformation for this class
                temp_dataset = CubDataset(class_df, transform=transform)
                # Collect only successfully loaded images
                images = [img for img, lbl in (temp_dataset[i] for i in range(len(temp_dataset))) if img is not None]

                if images:
                    if len(images) >= min_samples_needed:
                        data.append((subset_class_id, images)) # Store with SUBSET class ID
                        valid_classes_count += 1
                    else:
                        # Only add if replacement is acceptable, otherwise skip
                        # For now, we allow replacement in the sampler, so we *could* include them
                        # but it's better to warn and potentially filter here if strictness is needed.
                        # data.append((subset_class_id, images)) # Add even if insufficient
                        classes_with_insufficient_samples.append(subset_class_id)

            # else: Class not found in this CUB split (e.g., a meta-train class only in CUB's test set)

        if classes_with_insufficient_samples:
             print(f"\n*** WARNING ({split_name}): {len(classes_with_insufficient_samples)} classes have < {min_samples_needed} samples (needed for K={k_shot}, Q={n_query}).")
             print(f"   Example insufficient class subset IDs: {classes_with_insufficient_samples[:5]}")
             print(f"   Sampler will use replacement for these classes.")

        if valid_classes_count < n_way:
             print(f"*** CRITICAL WARNING ({split_name}): Only {valid_classes_count} classes have enough samples ({min_samples_needed}).")
             print(f"   The effective N_WAY for this split might be lower than {n_way} if insufficient classes are consistently sampled.")

        # Filter data to only include classes with enough samples if strictness is desired
        # data = [d for d in data if len(d[1]) >= min_samples_needed]
        # print(f"Filtered {split_name} data to {len(data)} classes with >= {min_samples_needed} samples.")

        return data

    print("\nBuilding meta-train data (from CUB train split)...")
    meta_train_data = load_data_for_split(meta_train_subset_indices, df_subset_train_cub, "Train")
    print(f"Meta-train data loaded for {len(meta_train_data)} classes.")

    print("\nBuilding meta-test data (from CUB test split)...")
    meta_test_data = load_data_for_split(meta_test_subset_indices, df_subset_test_cub, "Test")
    print(f"Meta-test data loaded for {len(meta_test_data)} classes.")

    # Final check if enough classes remain for the adjusted n_way
    if len(meta_train_data) < n_way or len(meta_test_data) < n_way:
         print("\n*** ERROR: Not enough classes with sufficient samples remain after loading.")
         print(f"Meta-Train classes available: {len(meta_train_data)}")
         print(f"Meta-Test classes available: {len(meta_test_data)}")
         print(f"Required N_WAY: {n_way}")
         print("Consider reducing K_SHOT, N_QUERY, or N_WAY, or check data integrity.")
         return None, None, n_way # Return None to signal failure

    print(f"\nFinal classes available for meta-train sampler: {len(meta_train_data)}")
    print(f"Final classes available for meta-test sampler: {len(meta_test_data)}")

    return meta_train_data, meta_test_data, n_way # Return the potentially adjusted n_way