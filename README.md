# Explainable Prototypical Networks for Few-Shot Learning on CUB-200-2011

## Overview

This project implements an Explainable Prototypical Network for few-shot image classification on the Caltech-UCSD Birds-200-2011 (CUB-200-2011) dataset. Prototypical Networks are a popular meta-learning approach for few-shot learning, where classification is performed by finding the nearest class prototype in an embedding space.

This implementation extends the basic Prototypical Network by incorporating an attention-based explanation mechanism. After predicting the class of a query image (Level 1), the model provides a visual explanation (Level 2) by identifying the most relevant patch in the query image and its corresponding matching patch in a support image of the predicted class.

## Features

*   **Few-Shot Classification:** Uses Prototypical Networks for N-Way K-Shot classification.
*   **Explainability:** Implements a Level 2 attention mechanism to find corresponding patches between query and support images for the predicted class.
*   **Backbone:** Utilizes a pre-trained ResNet18 model as the feature encoder.
*   **Fine-tuning:** Allows freezing early layers of the ResNet backbone (`FREEZE_UNTIL_LAYER` in `config.py`).
*   **Regularization:** Includes Dropout in the embedding head, Weight Decay (AdamW), Label Smoothing, and Gradient Clipping.
*   **Differential Learning Rates:** Applies different learning rates for the backbone and the newly added embedding head.
*   **CUB-200-2011 Data Handling:**
    *   Parses standard CUB metadata files.
    *   Uses bounding boxes to crop images, focusing on the bird.
    *   Splits classes into meta-train and meta-test sets.
    *   Handles data loading and episode sampling efficiently.
*   **Visualization:** Generates plots showing training progress (loss, accuracy) and visualization of the patch-level explanations.
*   **Modular Code:** Organized into separate Python files for configuration, data utilities, model definition, training logic, and visualization.

## File Structure
├── config.py # All configuration parameters and constants\
├── data_utils.py # CUB data parsing, dataset class, meta-split logic\
├── sampler.py # EpisodeSampler class for few-shot batches\
├── model.py # ResNetEncoderWithDropout and ExplainablePrototypicalNet classes\
├── training.py # Training loop, evaluation functions, optimizer setup, loss class\
├── visualization.py # Explanation visualization logic\
├── main.py # Main script to run the entire workflow\
├── requirements.txt # Python package dependencies\
└── README.md 



## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    https://github.com/Sandeep-4469/Explainability-Meta.git
    cd Explainability-Meta
    ```

2.  **Download CUB-200-2011 Dataset:**
    *   Download the dataset from the [official website](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html). You'll need `CUB_200_2011.tgz`.
    *   Extract the dataset. You should have a directory named `CUB_200_2011` containing:
        *   `images/` (subdirectory with images organized by class)
        *   `images.txt`
        *   `image_class_labels.txt`
        *   `train_test_split.txt`
        *   `bounding_boxes.txt`
        *   `classes.txt`
        *   ... and other metadata files.

3.  **Configure Data Path:**
    *   **CRITICAL:** Open the `config.py` file.
    *   Modify the `DATA_DIR` variable to point to the **absolute or relative path** of your extracted `CUB_200_2011` directory.
        ```python
        # Example:
        DATA_DIR = '/path/to/your/datasets/CUB_200_2011'
        # Or for Google Colab:
        # DATA_DIR = '/content/drive/MyDrive/CUB_200_2011'
        ```
    *   Ensure the `IMAGES_DIR` derived from `DATA_DIR` correctly points to the `images` subdirectory within your `CUB_200_2011` folder. The script has a basic check, but verify this path.

4.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

5.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    This will install PyTorch, Torchvision, Pandas, NumPy, Matplotlib, TQDM, and Pillow. Ensure your PyTorch version is compatible with your CUDA version if using a GPU.

## Running the Code

Execute the main script from the project's root directory:

```bash
python main.py