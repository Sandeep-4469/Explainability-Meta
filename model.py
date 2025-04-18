# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

# No dependency on config here, parameters passed during init

class ResNetEncoderWithDropout(nn.Module):
    """
    Encoder based on pretrained ResNet18 with layer freezing and dropout.
    Provides access to both convolutional features and final embeddings.
    """
    def __init__(self, embedding_dim, pretrained=True, freeze_until=None, dropout_rate=0.5):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        # Backbone Feature Extraction Layers
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Feature extractor combines these layers
        self.feature_extractor = nn.Sequential(
            self.stem, self.layer1, self.layer2, self.layer3, self.layer4
        )

        # Freeze layers based on configuration
        if freeze_until:
            self._freeze_layers(freeze_until)

        # Embedding Head
        resnet_out_dim = resnet.fc.in_features # 512 for ResNet18
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_rate) # Add dropout before final layer
        self.embedding_layer = nn.Linear(resnet_out_dim, embedding_dim)

        self.embedding_head = nn.Sequential(
            self.global_pool,
            self.flatten,
            self.dropout,
            self.embedding_layer
        )

        # Calculate total stride (typically 32 for ResNet18)
        # Assuming standard ResNet strides: conv1 (2), maxpool (2), layer1(1), layer2(2), layer3(2), layer4(2)
        # Total = 2 * 2 * 1 * 2 * 2 * 2 = 32 (Check architecture if modified)
        self.total_stride = 32

    def _freeze_layers(self, freeze_until):
        print(f"Freezing ResNet layers up to and including: {freeze_until}")
        target_modules_map = {
            "stem": [self.stem],
            "layer1": [self.stem, self.layer1],
            "layer2": [self.stem, self.layer1, self.layer2],
            "layer3": [self.stem, self.layer1, self.layer2, self.layer3],
            # Add "layer4" if you want to freeze it too
        }
        if freeze_until not in target_modules_map:
             valid_options = list(target_modules_map.keys())
             print(f"Warning: Invalid freeze_until value '{freeze_until}'. Valid options are: {valid_options}. No layers frozen.")
             return

        target_modules = target_modules_map[freeze_until]

        for module in target_modules:
            for param in module.parameters():
                param.requires_grad = False

        # --- Verification ---
        print("Trainable status check (after freezing):")
        all_modules_to_check = [
            ('stem', self.stem), ('layer1', self.layer1), ('layer2', self.layer2),
            ('layer3', self.layer3), ('layer4', self.layer4),
            ('embedding_layer', self.embedding_layer) # Check head too
        ]
        for name, module in all_modules_to_check:
             try:
                 is_trainable = any(p.requires_grad for p in module.parameters())
                 print(f"  {name}: {'Trainable' if is_trainable else 'Frozen'}")
             except AttributeError:
                 print(f"  {name}: Error checking parameters (module might be empty or Sequential)")
             except Exception as e:
                 print(f"  {name}: Error checking - {e}")


    def forward(self, x):
        conv_features = self.feature_extractor(x) # e.g., [B, 512, H/32, W/32]
        embedding = self.embedding_head(conv_features) # -> [B, embedding_dim]
        return conv_features, embedding

    def get_trainable_parameters(self):
        """ Helper to get parameters for differential learning rates """
        backbone_params = []
        head_params = []
        processed_params = set() # Keep track of params already assigned

        # Identify all parameters within the embedding_head explicitly
        head_param_ids = set(id(p) for p in self.embedding_head.parameters())

        # Iterate through all named parameters of the encoder
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue # Skip frozen parameters

            param_id = id(param)
            if param_id in processed_params: # Should not happen with named_parameters, but safeguard
                continue

            # Check if the parameter belongs to the embedding head
            if param_id in head_param_ids:
                # print(f"DEBUG: Assigning '{name}' to HEAD parameters.")
                head_params.append(param)
            else:
                # Assume it belongs to the backbone if not in the head and trainable
                # print(f"DEBUG: Assigning '{name}' to BACKBONE parameters.")
                backbone_params.append(param)
            processed_params.add(param_id)

        print(f"Differential LR setup: {len(backbone_params)} trainable backbone params, {len(head_params)} trainable head params.")
        if not head_params and any(p.requires_grad for p in self.embedding_head.parameters()):
             print("Warning: No parameters identified for the embedding head, but head seems trainable. Check logic.")
        if not backbone_params and any(p.requires_grad for p in self.feature_extractor.parameters()):
             print("Warning: No parameters identified for the backbone, but backbone seems trainable. Check logic.")

        return backbone_params, head_params


class ExplainablePrototypicalNet(nn.Module):
    """
    Main model using the ResNet encoder.
    Implements Level 1 Classification and Level 2 Patch Explanation.
    """
    def __init__(self, encoder):
        super().__init__()
        if not isinstance(encoder, ResNetEncoderWithDropout):
             raise TypeError("Encoder must be an instance of ResNetEncoderWithDropout")
        self.encoder = encoder

    def _calculate_prototypes(self, support_embeddings, support_labels, n_way):
        """Calculates class prototypes from support embeddings."""
        prototypes = torch.zeros(n_way, support_embeddings.size(1), device=support_embeddings.device)
        for c in range(n_way):
            # Select embeddings for the current class
            class_embeddings = support_embeddings[support_labels == c]
            if class_embeddings.size(0) > 0:
                # Compute mean embedding for the class
                prototypes[c] = class_embeddings.mean(dim=0)
            else:
                # Handle case where a class might have 0 support samples (shouldn't happen with good sampling)
                print(f"Warning: No support examples found for class {c} during prototype calculation.")
                # Prototype remains zero, distance will be large.
        return prototypes

    def _compute_distances(self, query_embeddings, prototypes):
        """Computes squared Euclidean distances between query embeddings and prototypes."""
        # query_embeddings: [N_query, EmbDim]
        # prototypes: [N_way, EmbDim]
        # Result: [N_query, N_way]
        return torch.cdist(query_embeddings, prototypes)**2

    def level1_classify(self, support_embeddings, support_labels, query_embeddings, n_way):
        """Performs classification based on distances to prototypes."""
        prototypes = self._calculate_prototypes(support_embeddings, support_labels, n_way)
        # distances shape: [n_query, n_way]
        distances = self._compute_distances(query_embeddings, prototypes)
        # Convert distances to logits (closer = higher logit)
        logits = -distances
        return logits

    def level2_explain_with_attention(self, query_conv_features, support_conv_features_list, predicted_class_support_indices):
        """
        Generates patch-level explanation using cross-attention.

        Args:
            query_conv_features (Tensor): Feature map for the single query image [1, D, H, W].
            support_conv_features_list (Tensor): Feature maps for ALL support images [N_support, D, H, W].
            predicted_class_support_indices (list): Indices (within the 0..N_support-1 range)
                                                    of support images belonging to the predicted class.

        Returns:
            tuple: (best_query_patch_idx_flat, best_support_patch_idx_in_image_flat, best_support_global_idx)
                   Returns (-1, -1, -1) if explanation cannot be generated.
        """
        if query_conv_features.size(0) != 1:
            raise ValueError("Level 2 explanation expects batch size 1 for query_conv_features.")
        if not predicted_class_support_indices:
            # print("Debug: No support indices provided for the predicted class.")
            return -1, -1, -1

        # Filter support features for the predicted class using the provided global indices
        support_features_pred_class = support_conv_features_list[predicted_class_support_indices]
        # Shape: [K_pred, D, H, W], K_pred = #support for predicted class

        if support_features_pred_class.numel() == 0:
            # print(f"Debug: Empty support features tensor for explanation after filtering.")
            return -1, -1, -1

        B, D, H, W = query_conv_features.shape # B is 1
        K = support_features_pred_class.size(0) # Number of support samples for this class
        N_PATCHES = H * W

        if N_PATCHES <= 0:
            print(f"Warning: Feature map has zero patches (H={H}, W={W}). Cannot calculate attention.")
            return -1, -1, -1
        if D <= 0:
             print(f"Warning: Feature map has zero dimension (D={D}). Cannot calculate attention.")
             return -1, -1, -1

        # Reshape features into patch lists: [NumPatches, Dim]
        # Query: [HW, D]
        query_patches = query_conv_features.permute(0, 2, 3, 1).reshape(N_PATCHES, D)
        # Support: [K*HW, D]
        support_patches = support_features_pred_class.permute(0, 2, 3, 1).reshape(K * N_PATCHES, D)

        try:
            # Normalize patches (optional, but can help stability)
            query_patches_norm = F.normalize(query_patches, p=2, dim=1)
            support_patches_norm = F.normalize(support_patches, p=2, dim=1)

            # Clean potential NaNs/Infs introduced by normalization or earlier steps
            query_patches_clean = torch.nan_to_num(query_patches_norm, nan=0.0)
            support_patches_clean = torch.nan_to_num(support_patches_norm, nan=0.0)

            # --- Cross Attention: Query attends to Support Patches ---
            # Calculate scaled dot-product similarity (cosine similarity on normalized features)
            # attention_scores shape: [HW, K*HW]
            scale_factor = 1.0 # No scaling needed for cosine similarity, or use math.sqrt(D) if unnormalized
            attention_scores = torch.matmul(query_patches_clean, support_patches_clean.t()) / scale_factor

            # Check for issues *after* matmul
            if torch.isnan(attention_scores).any() or torch.isinf(attention_scores).any():
                print("Warning: NaN/Inf detected in attention scores. Clamping before softmax.")
                attention_scores = torch.nan_to_num(attention_scores, nan=-1e9) # Use large negative for softmax

            # Softmax over support patches for each query patch
            attention_weights = F.softmax(attention_scores, dim=1) # Shape: [HW, K*HW]

            # --- Find the best matching patches ---
            # Find the maximum attention weight received by each query patch (from any support patch)
            max_attn_per_query, _ = torch.max(attention_weights, dim=1) # Shape: [HW]

            # Find the query patch index that has the overall highest attention score peak
            best_query_patch_idx_flat = torch.argmax(max_attn_per_query).item()

            # For that specific best query patch, find which support patch gave it the highest attention
            best_support_patch_idx_flat_overall = torch.argmax(attention_weights[best_query_patch_idx_flat]).item()

            # --- Map indices back ---
            # Map the flat support patch index back to:
            # 1. Which support image (within the predicted class subset) it belongs to
            best_support_image_local_idx = best_support_patch_idx_flat_overall // N_PATCHES
            # 2. The flat index of the patch within that support image's feature map
            best_support_patch_idx_in_image_flat = best_support_patch_idx_flat_overall % N_PATCHES

            # Get the *global* index of the best matching support image (from the original full support set)
            # using the local index relative to the predicted_class_support_indices list
            best_support_global_idx = predicted_class_support_indices[best_support_image_local_idx]

            return best_query_patch_idx_flat, best_support_patch_idx_in_image_flat, best_support_global_idx

        except RuntimeError as e:
            print(f"Runtime Error during Attention calculation: {e}")
            # Could be OOM, size mismatch, etc.
            return -1, -1, -1
        except ValueError as e:
             print(f"Value Error during Attention calculation (e.g., empty tensor): {e}")
             return -1, -1, -1


    def forward(self, support_images, support_labels, query_images, n_way):
        """
        Processes a batch of support and query images for an episode.

        Returns:
            logits (Tensor): Classification logits for query images [N_query, N_way].
            query_conv_features (Tensor): Feature maps for query images [N_query, D, H, W].
            support_conv_features (Tensor): Feature maps for support images [N_support, D, H, W].
        """
        n_support = support_images.size(0)
        n_query = query_images.size(0)

        # Concatenate support and query images to process them together through the encoder
        all_images = torch.cat([support_images, query_images], dim=0)

        # Get convolutional features and final embeddings from the encoder
        try:
            all_conv_features, all_embeddings = self.encoder(all_images)
        except Exception as e:
            print(f"Error during encoder forward pass: {e}. Returning dummy outputs.")
             # Create dummy outputs with expected shapes but potentially wrong device/dtype if error is severe
            dummy_logits = torch.zeros(n_query, n_way, device=support_images.device)
            feat_h, feat_w = 1, 1 # Placeholder spatial dims, adjust if possible
            emb_dim = self.encoder.embedding_layer.out_features
            feat_dim = self.encoder.embedding_layer.in_features
            dummy_q_conv = torch.zeros(n_query, feat_dim, feat_h, feat_w, device=support_images.device)
            dummy_s_conv = torch.zeros(n_support, feat_dim, feat_h, feat_w, device=support_images.device)
            return dummy_logits, dummy_q_conv, dummy_s_conv

        # Split the results back into support and query sets
        support_embeddings = all_embeddings[:n_support]
        query_embeddings = all_embeddings[n_support:]
        support_conv_features = all_conv_features[:n_support]
        query_conv_features = all_conv_features[n_support:]

        # Level 1 classification: Calculate logits based on prototype distances
        logits = self.level1_classify(support_embeddings, support_labels, query_embeddings, n_way)

        # Return logits and feature maps (needed for explanation)
        return logits, query_conv_features, support_conv_features