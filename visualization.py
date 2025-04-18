# visualization.py
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random

import config # Import DEVICE, IMAGE_SIZE, PATCH_SIZE_VIS

# Inverse normalization transform for visualizing ImageNet-normalized images
inv_normalize_imagenet = transforms.Normalize(
   mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
   std=[1/0.229, 1/0.224, 1/0.225]
)

def tensor_to_numpy_img(tensor_img):
    """Safely convert a normalized tensor image to a numpy array for display."""
    if tensor_img is None:
        print("Warning: Received None tensor in tensor_to_numpy_img.")
        return np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3)) # Return black image

    try:
        # Ensure tensor is on CPU and detached from computation graph
        img = tensor_img.detach().cpu()
        # Apply inverse normalization
        img = inv_normalize_imagenet(img)
        # Permute dimensions from (C, H, W) to (H, W, C) for plotting
        img = img.permute(1, 2, 0)
        # Clamp values to [0, 1] range to prevent issues with minor float inaccuracies
        img = img.clamp(0, 1)
        return img.numpy()
    except Exception as e:
        print(f"Error during tensor_to_numpy_img conversion: {e}")
        # Return a placeholder image on error
        return np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3))

def map_patch_coords_to_image(flat_idx, patch_h, patch_w, total_stride, img_size=config.IMAGE_SIZE, patch_vis_size=config.PATCH_SIZE_VIS):
    """
    Maps a flat index from a feature map patch to the top-left corner
    coordinates for drawing a bounding box on the original image.

    Args:
        flat_idx (int): The flattened index of the patch (0 to H*W - 1).
        patch_h (int): Height of the feature map.
        patch_w (int): Width of the feature map.
        total_stride (int): The total downsampling stride from image to feature map.
        img_size (int): The size of the input image (assumed square).
        patch_vis_size (int): The desired size of the visualization bounding box.

    Returns:
        tuple: (top_left_x, top_left_y) coordinates for the bounding box.
               Returns (0, 0) if dimensions are invalid.
    """
    if patch_h <= 0 or patch_w <= 0 or total_stride <= 0:
        # print(f"Warning: Invalid dimensions for patch mapping (H={patch_h}, W={patch_w}, Stride={total_stride}).")
        return 0, 0 # Default to top-left corner

    # Calculate the row (y) and column (x) of the patch in the feature map
    patch_y = flat_idx // patch_w
    patch_x = flat_idx % patch_w

    # Calculate the center coordinates of the receptive field in the original image
    # corresponding to the center of this patch
    center_y = (patch_y + 0.5) * total_stride
    center_x = (patch_x + 0.5) * total_stride

    # Calculate the top-left corner of the visualization box based on its desired size
    half_patch_vis = patch_vis_size // 2
    top_left_y = int(center_y - half_patch_vis)
    top_left_x = int(center_x - half_patch_vis)

    # Clamp coordinates to ensure the box stays within image bounds
    top_left_x = max(0, min(top_left_x, img_size - patch_vis_size))
    top_left_y = max(0, min(top_left_y, img_size - patch_vis_size))

    return top_left_x, top_left_y

def visualize_explanation(model, test_sampler, num_explanations=config.NUM_VISUALIZATIONS):
    """
    Visualizes the explanation for a few random query images from test episodes.
    Draws boxes around the most attentive query patch and its corresponding support patch.
    """
    if num_explanations <= 0: return
    print(f"\n--- Visualizing Explanations for {num_explanations} Test Episodes ---")
    model.eval() # Ensure model is in evaluation mode

    for vis_idx in range(num_explanations):
        print(f"\n--- Explanation Example {vis_idx + 1}/{num_explanations} ---")
        try:
            # Sample a test episode
            support_imgs, support_lbls, query_imgs, query_lbls = test_sampler.sample()
            n_way = test_sampler.n_way
            k_shot = test_sampler.k_shot # Used for plot layout
        except ValueError as e:
            print(f"Skipping visualization {vis_idx+1} due to sampler error: {e}")
            continue
        except Exception as e:
             print(f"An unexpected error occurred during sampling for visualization {vis_idx+1}: {e}")
             continue

        # Ensure there are query images to explain
        if query_imgs.size(0) == 0:
            print("Sampled query set is empty, cannot generate explanation.")
            continue

        # Select a random query image index from the episode
        query_idx_to_explain = random.randrange(query_imgs.size(0))
        query_image_single = query_imgs[query_idx_to_explain].unsqueeze(0) # Add batch dim
        true_label = query_lbls[query_idx_to_explain].item() # Get scalar label

        # --- Move ALL necessary tensors to the configured device ---
        support_images_dev = support_imgs.to(config.DEVICE)
        support_labels_dev = support_lbls.to(config.DEVICE) # Labels needed for prototype calc
        query_image_single_dev = query_image_single.to(config.DEVICE)

        # --- Perform Inference and Explanation Generation ---
        query_conv_feature_single = None
        best_query_patch_flat_idx, best_support_patch_in_image_flat_idx, best_support_global_idx = -1, -1, -1
        predicted_label = -1

        with torch.no_grad():
            try:
                # 1. Get features and embeddings for ALL support images
                support_conv_features_all, support_embeddings_all = model.encoder(support_images_dev)

                # 2. Get features and embedding for the SINGLE query image
                query_conv_feature_single, query_embedding_single = model.encoder(query_image_single_dev)

                # 3. Perform Level 1 Classification for the single query image
                # Use the main model's classification logic which utilizes the encoder outputs
                logits_single = model.level1_classify(support_embeddings_all, support_labels_dev,
                                                      query_embedding_single, n_way)
                predicted_label = torch.argmax(logits_single, dim=1).item()

                # 4. Prepare for Level 2 Explanation
                # Find indices of support images belonging to the *predicted* class
                predicted_class_support_indices = [
                    i for i, lbl in enumerate(support_labels_dev.cpu().numpy()) # Compare labels on CPU
                    if lbl == predicted_label
                ]

                # 5. Generate the patch-level explanation if possible
                if predicted_class_support_indices and query_conv_feature_single is not None:
                    best_query_patch_flat_idx, best_support_patch_in_image_flat_idx, best_support_global_idx = \
                        model.level2_explain_with_attention(
                            query_conv_feature_single,          # Query feature map [1, D, H, W]
                            support_conv_features_all,        # ALL support feature maps [N_support, D, H, W]
                            predicted_class_support_indices   # List of relevant support image indices
                        )
                elif query_conv_feature_single is None:
                     print("Query feature map is None, cannot generate explanation.")
                elif not predicted_class_support_indices:
                     print(f"No support images found for predicted class {predicted_label}. Cannot generate explanation.")

            except Exception as e:
                print(f"Error during inference or explanation generation: {e}")
                # Continue to next visualization attempt

        # --- Print Explanation Info ---
        print(f"Query Image Index in Episode: {query_idx_to_explain}")
        print(f"Predicted Label (local 0-{n_way-1}): {predicted_label}")
        print(f"True Label (local 0-{n_way-1}): {true_label}")
        print(f"Correct Prediction: {'Yes' if predicted_label == true_label else 'No'}")

        if best_query_patch_flat_idx != -1:
            print(f"Explanation based on Predicted Class {predicted_label}:")
            print(f" -> Most Attentive Query Patch (flat index): {best_query_patch_flat_idx}")
            print(f" -> Matched Support Patch (flat index in its image): {best_support_patch_in_image_flat_idx}")
            print(f" -> Matched Support Image Index (0 to {support_imgs.size(0)-1}): {best_support_global_idx}")
        elif predicted_label != -1: # Only print failure if prediction happened
            print(" -> Patch-level explanation failed or could not be generated.")

        # --- Create Visualization Plot ---
        # Determine layout (Query on top, relevant support below)
        num_support_to_show = k_shot # Show all support images for the predicted class
        cols_support = max(num_support_to_show, 3) # Min 3 columns for layout
        fig, axes = plt.subplots(2, cols_support, figsize=(max(15, 3 * cols_support), 8)) # Adjust figsize dynamically
        fig.suptitle(f"Explanation {vis_idx+1}: Query Idx {query_idx_to_explain} (Pred: {predicted_label}, True: {true_label})", fontsize=14, y=0.99)

        # --- Display Query Image ---
        # Place query image spanning top row or in first cell
        ax_query = plt.subplot2grid((2, cols_support), (0, 0), colspan=cols_support) # Span top row
        query_img_np = tensor_to_numpy_img(query_image_single.squeeze(0)) # Remove batch dim for conversion
        ax_query.imshow(query_img_np)
        ax_query.set_title(f"Query Image (True Label: {true_label})")
        ax_query.axis('off')

        # Draw rectangle on query image if explanation was successful
        if best_query_patch_flat_idx != -1 and query_conv_feature_single is not None and query_conv_feature_single.numel() > 0:
            try:
                B, D, H_feat, W_feat = query_conv_feature_single.shape
                if H_feat > 0 and W_feat > 0:
                    total_stride = model.encoder.total_stride
                    q_patch_x, q_patch_y = map_patch_coords_to_image(
                        best_query_patch_flat_idx, H_feat, W_feat, total_stride,
                        config.IMAGE_SIZE, config.PATCH_SIZE_VIS
                    )
                    q_rect = Rectangle((q_patch_x, q_patch_y), config.PATCH_SIZE_VIS, config.PATCH_SIZE_VIS,
                                       linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
                    ax_query.add_patch(q_rect)
                    ax_query.set_title(f"Query Image (True: {true_label}, Pred: {predicted_label})\nAttentive Patch (Red)")
            except Exception as e:
                 print(f"Error drawing query patch rectangle: {e}")


        # --- Display Relevant Support Images ---
        support_indices_pred_class = [
            i for i, lbl in enumerate(support_lbls.numpy()) # Use original CPU labels
            if lbl == predicted_label
        ]

        for i, support_global_idx in enumerate(support_indices_pred_class):
            if i >= cols_support: break # Don't exceed allocated columns
            ax_supp = axes[1, i] # Use second row axes
            support_img_np = tensor_to_numpy_img(support_imgs[support_global_idx])
            ax_supp.imshow(support_img_np)
            ax_supp.set_title(f"Support Img {support_global_idx}\n(Class {predicted_label})")
            ax_supp.axis('off')

            # Highlight the best matching support patch if it's in this image
            if best_support_global_idx == support_global_idx and best_support_patch_in_image_flat_idx != -1:
                try:
                    # Need the shape of the specific support feature map
                    if support_conv_features_all is not None and support_conv_features_all.numel() > 0:
                         _, _, H_feat_s, W_feat_s = support_conv_features_all[support_global_idx].shape # Get shape directly
                         if H_feat_s > 0 and W_feat_s > 0:
                             total_stride = model.encoder.total_stride
                             s_patch_x, s_patch_y = map_patch_coords_to_image(
                                 best_support_patch_in_image_flat_idx, H_feat_s, W_feat_s, total_stride,
                                 config.IMAGE_SIZE, config.PATCH_SIZE_VIS
                             )
                             s_rect = Rectangle((s_patch_x, s_patch_y), config.PATCH_SIZE_VIS, config.PATCH_SIZE_VIS,
                                                linewidth=2.5, edgecolor='lime', facecolor='none')
                             ax_supp.add_patch(s_rect)
                             ax_supp.set_title(f"Support Img {support_global_idx}\n(Matching Patch - Lime)")
                except Exception as e:
                     print(f"Error drawing support patch rectangle for image {support_global_idx}: {e}")


        # Turn off axes for any unused subplots in the second row
        for j in range(len(support_indices_pred_class), cols_support):
            axes[1, j].axis('off')

        plt.tight_layout(rect=[0, 0.02, 1, 0.95]) # Adjust rect to prevent title overlap
        plt.show()