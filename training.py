# training.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm # Use standard tqdm if not in notebook

import config # Import DEVICE and other training params if needed

# Use CrossEntropyLoss with Label Smoothing
class LabelSmoothingLoss(nn.Module):
    """NLL loss with label smoothing."""
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # Create target distribution
            true_dist = torch.zeros_like(pred)
            # Fill with smoothed value
            true_dist.fill_(self.smoothing / (self.cls - 1))
            # Scatter the confidence value to the correct class index
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # Calculate KL divergence between predicted log-probabilities and smoothed target distribution
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

def setup_optimizer(model, lr_backbone, lr_head, wd):
     """ Creates optimizer with differential learning rates for encoder parts """
     # Ensure the model has the expected 'encoder' attribute
     if not hasattr(model, 'encoder') or not hasattr(model.encoder, 'get_trainable_parameters'):
         raise AttributeError("Model does not have the expected 'encoder' structure with 'get_trainable_parameters' method.")

     backbone_params, head_params = model.encoder.get_trainable_parameters()

     param_groups = []
     if backbone_params:
         param_groups.append({'params': backbone_params, 'lr': lr_backbone, 'weight_decay': wd})
     if head_params:
         param_groups.append({'params': head_params, 'lr': lr_head, 'weight_decay': wd})

     # Add other model parameters if they exist outside the encoder and require grad
     # Example: If PrototypicalNet had its own trainable parameters
     other_params = [p for n, p in model.named_parameters() if not n.startswith('encoder.') and p.requires_grad]
     if other_params:
          print(f"Adding {len(other_params)} other trainable parameters to optimizer group (LR={lr_head}).")
          # Decide LR and WD for these; using head LR and main WD here
          param_groups.append({'params': other_params, 'lr': lr_head, 'weight_decay': wd})

     if not param_groups:
          raise ValueError("No trainable parameters found for the optimizer.")

     # Using AdamW which handles weight decay correctly
     optimizer = optim.AdamW(param_groups)
     # Manually setting LR and WD per group for AdamW
     for group in optimizer.param_groups:
         group['lr'] = group.get('lr', lr_head) # Default LR if not specified
         group['weight_decay'] = group.get('weight_decay', wd) # Default WD if not specified

     print("Optimizer Parameter Groups:")
     for i, group in enumerate(optimizer.param_groups):
         print(f"  Group {i}: {len(group['params'])} params, LR={group['lr']}, WD={group['weight_decay']}")

     return optimizer


def train_step(model, optimizer, loss_fn, support_images, support_labels, query_images, query_labels, n_way, gradient_clip_norm):
    """Performs a single training step for one episode."""
    model.train()
    optimizer.zero_grad()

    # Move data to the configured device
    support_images = support_images.to(config.DEVICE)
    support_labels = support_labels.to(config.DEVICE)
    query_images = query_images.to(config.DEVICE)
    query_labels = query_labels.to(config.DEVICE)

    # Forward pass
    # Model returns logits, query features, support features
    logits, _, _ = model(support_images, support_labels, query_images, n_way)

    # Ensure logits are valid before loss calculation
    if torch.isnan(logits).any() or torch.isinf(logits).any():
         print("Warning: NaN/Inf logits detected BEFORE loss calculation. Skipping batch.")
         # Return NaN loss and 0 accuracy to signal the issue
         return float('nan'), 0.0

    # Calculate loss
    loss = loss_fn(logits, query_labels)

    # Check for NaN loss AFTER calculation (can happen with unstable training)
    if torch.isnan(loss) or torch.isinf(loss):
        print("Warning: NaN/Inf loss calculated. Skipping backward pass and optimizer step for this batch.")
        return float('nan'), 0.0 # Return NaN loss and 0 acc

    # Backward pass
    loss.backward()

    # Gradient Clipping (applied to all parameters regardless of group)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)

    # Optimizer step
    optimizer.step()

    # Calculate accuracy (on the query set for this episode)
    with torch.no_grad():
        predictions = torch.argmax(logits, dim=1)
        accuracy = torch.mean((predictions == query_labels).float())

    return loss.item(), accuracy.item()


def evaluate_step(model, support_images, support_labels, query_images, query_labels, n_way):
    """Evaluates the model on a single episode."""
    model.eval() # Set model to evaluation mode

    # Move data to the configured device
    support_images = support_images.to(config.DEVICE)
    support_labels = support_labels.to(config.DEVICE)
    query_images = query_images.to(config.DEVICE)
    query_labels = query_labels.to(config.DEVICE)

    with torch.no_grad(): # Ensure no gradients are computed
        logits, _, _ = model(support_images, support_labels, query_images, n_way)

        # Check for invalid logits during evaluation
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("Warning: NaN/Inf detected in logits during evaluation. Returning 0 accuracy for this episode.")
            return 0.0 # Return 0 accuracy for this problematic episode

        predictions = torch.argmax(logits, dim=1)
        accuracy = torch.mean((predictions == query_labels).float())

    return accuracy.item()

def evaluate_on_test_set(model, test_sampler, n_episodes):
    """Evaluates the model performance on the meta-test set over multiple episodes."""
    model.eval() # Ensure evaluation mode
    all_accuracies = []
    print(f"--- Evaluating on {n_episodes} Meta-Test episodes ---")
    pbar_eval = tqdm(range(n_episodes), desc="Meta-Testing", leave=False)
    for i in pbar_eval:
        try:
             # Sample an episode from the test sampler
             support_images, support_labels, query_images, query_labels = test_sampler.sample()
        except ValueError as e:
            print(f"Skipping test episode {i+1} due to sampler error: {e}")
            continue # Skip if sampling fails (e.g., not enough classes)

        # Perform evaluation on the sampled episode
        acc = evaluate_step(model, support_images, support_labels, query_images, query_labels, test_sampler.n_way)

        # Store accuracy if it's a valid number
        if not math.isnan(acc):
            all_accuracies.append(acc)
            avg_acc = np.mean(all_accuracies) if all_accuracies else 0.0
            pbar_eval.set_description(f"Meta-Testing (Episode {i+1}/{n_episodes}, Avg Acc: {avg_acc:.4f})")
        else:
             print(f"Warning: NaN accuracy returned for test episode {i+1}. Skipping.")

    # Calculate and return the mean accuracy over all evaluated episodes
    if not all_accuracies:
        print("Warning: No valid accuracies recorded during meta-test evaluation.")
        return 0.0
    final_avg_acc = np.mean(all_accuracies)
    print(f"--- Meta-Test Evaluation Complete ---")
    print(f"Average Accuracy over {len(all_accuracies)} valid episodes: {final_avg_acc:.4f}")
    return final_avg_acc

# --- Plotting Helper (kept within training as it uses training loop data) ---
def plot_metrics(train_losses, train_accuracies, test_accuracies, test_eval_interval, n_train_episodes):
     """Plots the training loss and accuracy curves."""
     plt.figure(figsize=(12, 5))

     # Smooth curves for better visualization (simple moving average)
     def smooth_curve(points, factor=0.9):
         smoothed_points = []
         if not points: return [] # Handle empty list
         for point in points:
             if smoothed_points:
                 previous = smoothed_points[-1]
                 smoothed_points.append(previous * factor + point * (1 - factor))
             else:
                 smoothed_points.append(point)
         return smoothed_points

     # Plot Training Loss
     plt.subplot(1, 2, 1)
     if train_losses:
         plt.plot(smooth_curve(train_losses), label='Smoothed Training Loss')
         plt.title("Training Loss")
         plt.xlabel("Episode")
         plt.ylabel("Loss")
         plt.ylim(bottom=max(0, np.min(train_losses)-0.1) if train_losses else 0) # Adjust ylim dynamically
         plt.grid(True, linestyle='--', alpha=0.6)
         plt.legend()
     else:
         plt.text(0.5, 0.5, "No valid loss data recorded", ha='center', va='center')
         plt.title("Training Loss")
         plt.xlabel("Episode")
         plt.ylabel("Loss")

     # Plot Accuracies
     plt.subplot(1, 2, 2)
     if train_accuracies:
         plt.plot(smooth_curve(train_accuracies), label="Smoothed Train Acc", alpha=0.8)
     if test_accuracies:
         # Calculate the episode numbers where test evaluation was performed
         eval_points = np.arange(test_eval_interval, n_train_episodes + 1, test_eval_interval)
         # Ensure we don't plot more points than we have evaluations
         eval_points = eval_points[:len(test_accuracies)]
         plt.plot(eval_points - 1, test_accuracies, label="Test Acc", marker='o', linestyle='--', markersize=6)

     plt.title("Accuracy")
     plt.xlabel("Episode")
     plt.ylabel("Accuracy")
     plt.ylim(0, 1.05) # Slightly above 1.0 for visibility
     plt.grid(True, linestyle='--', alpha=0.6)
     plt.legend()

     plt.tight_layout()
     plt.show()


def main_training_loop(model, train_sampler, test_sampler, n_train_episodes, n_test_episodes,
                      lr_backbone, lr_head, wd, label_smoothing, grad_clip_norm,
                      test_eval_interval, log_interval):
    """The main meta-training loop."""
    # Setup optimizer with differential LR and weight decay
    optimizer = setup_optimizer(model, lr_backbone, lr_head, wd)

    # Setup loss function with label smoothing
    loss_fn = LabelSmoothingLoss(classes=train_sampler.n_way, smoothing=label_smoothing).to(config.DEVICE)

    # Lists to store metrics
    train_losses, train_accuracies, test_accuracies = [], [], []
    best_test_acc = 0.0
    best_model_state = None # Store the best model state dict

    print("\n--- Starting Meta-Training ---")
    print(f"Device: {config.DEVICE}")
    print(f"Config: N-Way={train_sampler.n_way}, K-Shot={train_sampler.k_shot}, Q={train_sampler.n_query}")
    print(f"LRs: Backbone={lr_backbone}, Head={lr_head}, WD={wd}, LS={label_smoothing}")
    print(f"Episodes: Train={n_train_episodes}, Test={n_test_episodes}, GradClip={grad_clip_norm}")
    print(f"Evaluation Interval: {test_eval_interval}, Log Interval: {log_interval}")

    pbar_train = tqdm(range(n_train_episodes), desc="Meta-Training")
    for episode_idx in pbar_train:
        try:
            # Sample a training episode
            support_images, support_labels, query_images, query_labels = train_sampler.sample()
        except ValueError as e:
             print(f"\nError sampling train episode {episode_idx+1}: {e}. Skipping.")
             continue # Skip this episode if sampling fails

        # Perform a training step
        loss, acc = train_step(model, optimizer, loss_fn, support_images, support_labels,
                              query_images, query_labels, train_sampler.n_way, grad_clip_norm)

        # Log metrics if the step was successful (no NaN loss)
        if not math.isnan(loss):
            train_losses.append(loss)
            train_accuracies.append(acc)

            # Log average metrics periodically
            if (episode_idx + 1) % log_interval == 0 and train_losses:
                # Calculate averages over the last 'log_interval' episodes
                avg_loss = np.mean(train_losses[-log_interval:])
                avg_acc = np.mean(train_accuracies[-log_interval:])
                pbar_train.set_description(f"Ep {episode_idx+1}/{n_train_episodes} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | Best Test: {best_test_acc:.4f}")

            # Evaluate on the meta-test set periodically
            if (episode_idx + 1) % test_eval_interval == 0:
                 # Use fewer episodes for intermediate evaluations to save time
                 num_eval_eps = max(100, n_test_episodes // 5) # Evaluate on at least 100 or 1/5th
                 current_test_acc = evaluate_on_test_set(model, test_sampler, num_eval_eps)
                 test_accuracies.append(current_test_acc)

                 if current_test_acc > best_test_acc:
                     best_test_acc = current_test_acc
                     print(f"\n*** New best test accuracy: {best_test_acc:.4f} at episode {episode_idx+1}. Saving model state... ***")
                     # Save the model's state dictionary (in memory for now)
                     try:
                         best_model_state = model.state_dict()
                         # Optionally save to disk immediately
                         save_path = "best_cub_explainable_protonet.pth"
                         torch.save(best_model_state, save_path)
                         print(f"Model state saved to {save_path}")
                     except Exception as e:
                          print(f"Error saving model state: {e}")

                 # Restore train mode after evaluation
                 model.train()
                 # Update progress bar description again after evaluation completes
                 pbar_train.set_description(f"Ep {episode_idx+1}/{n_train_episodes} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | Best Test: {best_test_acc:.4f}")


        else:
             # Log if a step was skipped due to NaN
             if (episode_idx + 1) % log_interval == 0:
                  pbar_train.set_description(f"Ep {episode_idx+1}/{n_train_episodes} | Loss: NaN | Acc: --- | Best Test: {best_test_acc:.4f}")


    print("\n--- Finished Meta-Training ---")
    print(f"Best Meta-Test Accuracy recorded during training: {best_test_acc:.4f}")

    # Final evaluation using the full number of test episodes
    print("\n--- Final Evaluation on Meta-Test set ---")
    # Load the best model state found during training if available
    if best_model_state:
        print("Loading best model state for final evaluation...")
        try:
            model.load_state_dict(best_model_state)
        except Exception as e:
             print(f"Error loading best model state: {e}. Evaluating with the final model state.")

    final_test_accuracy = evaluate_on_test_set(model, test_sampler, n_test_episodes)
    print(f"Final Meta-Test Accuracy ({n_test_episodes} episodes): {final_test_accuracy:.4f}")

    # Plot the training and evaluation metrics
    plot_metrics(train_losses, train_accuracies, test_accuracies, test_eval_interval, n_train_episodes)

    return model # Return the model (potentially the one with the best state loaded)