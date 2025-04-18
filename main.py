# main.py
import torch
import os
import numpy as np
import random

# Import modules
import config
import data_utils
import sampler
import model
import training
import visualization

def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    set_seed() # Set seed for reproducibility

    print(f"Using device: {config.DEVICE}")
    print(f"Data directory: {config.DATA_DIR}")
    if not os.path.exists(config.DATA_DIR):
         print("\n--- FATAL ERROR: Data directory not found. Please check config.py ---")
         exit()

    print("\n1. Preparing CUB Data Splits...")
    try:
        meta_train_data, meta_test_data, adjusted_n_way = data_utils.prepare_cub_data_splits(
            data_dir=config.DATA_DIR,
            images_dir=config.IMAGES_DIR,
            image_size=config.IMAGE_SIZE,
            n_way=config.N_WAY, # Pass initial N_WAY
            k_shot=config.K_SHOT,
            n_query=config.N_QUERY,
            n_meta_train_ratio=config.N_META_TRAIN_RATIO
        )
        # Update N_WAY in config if it was adjusted (optional, as samplers use the adjusted value)
        if adjusted_n_way != config.N_WAY:
             print(f"Note: N_WAY has been adjusted to {adjusted_n_way} based on available data.")
             # config.N_WAY = adjusted_n_way # Update global config if needed elsewhere

    except FileNotFoundError as e:
        print(f"\n--- ERROR: Metadata file not found. {e} ---")
        print("--- Please ensure CUB dataset files are present in the DATA_DIR. ---")
        exit()
    except ValueError as e:
         print(f"\n--- ERROR during data preparation: {e} ---")
         exit()
    except Exception as e:
         print(f"\n--- An unexpected error occurred during data preparation: {e} ---")
         exit()


    if meta_train_data and meta_test_data:
        print("\n2. Creating Episode Samplers...")
        try:
            # Use the potentially adjusted n_way
            train_sampler = sampler.EpisodeSampler(meta_train_data, adjusted_n_way, config.K_SHOT, config.N_QUERY)
            test_sampler = sampler.EpisodeSampler(meta_test_data, adjusted_n_way, config.K_SHOT, config.N_QUERY)

            # Sample and print shapes to verify
            s_img, s_lbl, q_img, q_lbl = train_sampler.sample()
            print("Sampled Train Episode shapes:")
            print(f"  Support Images: {s_img.shape}, Labels: {s_lbl.shape}")
            print(f"  Query Images:   {q_img.shape}, Labels: {q_lbl.shape}")
        except ValueError as e:
             print(f"\n--- ERROR creating samplers: {e} ---")
             exit()
        except Exception as e:
            print(f"\n--- An unexpected error occurred creating samplers: {e} ---")
            exit()

        print("\n3. Initializing Model (ResNet18 Encoder + ProtoNet)...")
        try:
            encoder = model.ResNetEncoderWithDropout(
                embedding_dim=config.EMBEDDING_DIM,
                pretrained=config.PRETRAINED,
                freeze_until=config.FREEZE_UNTIL_LAYER,
                dropout_rate=config.DROPOUT_RATE
            ).to(config.DEVICE)

            # Pass the initialized encoder to the main model
            protonet_model = model.ExplainablePrototypicalNet(encoder).to(config.DEVICE)

            # Print parameter counts
            trainable_params = sum(p.numel() for p in protonet_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in protonet_model.parameters())
            print(f"Model Parameters: Total={total_params:,}, Trainable={trainable_params:,}")

        except Exception as e:
             print(f"\n--- ERROR initializing model: {e} ---")
             exit()

        print("\n4. Starting Meta-Training...")
        try:
            trained_model = training.main_training_loop(
                model=protonet_model,
                train_sampler=train_sampler,
                test_sampler=test_sampler,
                n_train_episodes=config.N_TRAIN_EPISODES,
                n_test_episodes=config.N_TEST_EPISODES,
                lr_backbone=config.LR_BACKBONE,
                lr_head=config.LR_HEAD,
                wd=config.WEIGHT_DECAY,
                label_smoothing=config.LABEL_SMOOTHING,
                grad_clip_norm=config.GRADIENT_CLIP_NORM,
                test_eval_interval=config.TEST_EVAL_INTERVAL,
                log_interval=config.LOG_INTERVAL
            )
        except Exception as e:
             print(f"\n--- An error occurred during the training loop: {e} ---")
             # Optionally load the last saved best model if training failed
             trained_model = protonet_model # Keep the model instance
             save_path = "best_cub_explainable_protonet.pth"
             if os.path.exists(save_path):
                 print(f"Attempting to load best saved model from {save_path} for visualization.")
                 try:
                      trained_model.load_state_dict(torch.load(save_path, map_location=config.DEVICE))
                 except Exception as load_e:
                      print(f"Failed to load saved model: {load_e}")
             else:
                 print("No saved model found to load after training error.")
             # Decide whether to proceed to visualization or exit
             # exit()

        print("\n5. Visualizing Explanation on Test Episodes...")
        try:
            visualization.visualize_explanation(
                model=trained_model,
                test_sampler=test_sampler,
                num_explanations=config.NUM_VISUALIZATIONS
            )
        except Exception as e:
             print(f"\n--- An error occurred during visualization: {e} ---")

        print("\n--- Run Finished ---")

    else:
        print("\n--- ERROR: Failed to prepare data. Check paths and data integrity. Exiting. ---")