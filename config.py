# config.py
import torch
import os

# -------------------------------------
# --- Configuration ---
# -------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset params
# IMPORTANT: Set this path correctly for your environment
# Example for Google Colab: '/content/drive/MyDrive/CUB_200_2011'
# Example for local machine: 'path/to/your/CUB_200_2011'
DATA_DIR = '/content/drive/MyDrive/CUB_200_2011' #<---- CHECK YOUR PATH
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
IMAGE_SIZE = 224
N_META_TRAIN_RATIO = 0.7 # Ratio of classes for meta-training

# Episode params
N_WAY = 5          # Number of classes per episode (will be checked against available classes)
K_SHOT = 5         # Number of support examples per class
N_QUERY = 10       # Number of query examples per class
N_TRAIN_EPISODES = 5000 # Number of episodes for meta-training
N_TEST_EPISODES = 600   # Number of episodes for meta-testing

# Model params
EMBEDDING_DIM = 256 # Feature embedding dimension
PRETRAINED = True
FREEZE_UNTIL_LAYER = "layer3" # Options: None, "stem", "layer1", "layer2", "layer3"
DROPOUT_RATE = 0.5 # Dropout rate in the encoder head

# Training params
LR_BACKBONE = 1e-5      # Lower LR for frozen/finetuned backbone
LR_HEAD = 1e-4          # Higher LR for the new embedding layer
WEIGHT_DECAY = 5e-4      # Increased weight decay
LABEL_SMOOTHING = 0.1    # Use label smoothing
GRADIENT_CLIP_NORM = 1.0 # Use gradient clipping
TEST_EVAL_INTERVAL = 500 # How often to evaluate on test set during training
LOG_INTERVAL = 100       # How often to log training progress

# Visualization Params
PATCH_SIZE_VIS = 28 # Approximate size of patch to draw rectangle for visualization
NUM_VISUALIZATIONS = 5 # Number of explanation examples to generate

# --- Derived/Checked Parameters ---
# Ensure IMAGES_DIR is correctly derived from DATA_DIR
if not os.path.isdir(DATA_DIR):
    print(f"WARNING: DATA_DIR specified does not exist: {DATA_DIR}")
    # Optionally raise an error: raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR}")
elif not os.path.isdir(IMAGES_DIR):
     print(f"WARNING: Default IMAGES_DIR does not exist: {IMAGES_DIR}")
     # Optionally try alternative common structure 'CUB_200_2011/images' if root is given
     alt_images_dir = os.path.join(DATA_DIR, 'CUB_200_2011', 'images')
     if os.path.isdir(alt_images_dir):
         IMAGES_DIR = alt_images_dir
         print(f"Found images at alternative path: {IMAGES_DIR}")
     # else: raise FileNotFoundError(f"Cannot find images directory within {DATA_DIR}")