import cv2

import numpy as np
import os
import random
import time
import torch
import torch.optim as optim
import torch.nn as nn

from dqn_slice_based.dqn import PerPixelCNNWithHistory, ReplayBuffer
from dqn_slice_based.env import PathReconstructionEnv

from matplotlib import pyplot as plt
from tqdm import tqdm

USE_ARTIFACTS = False

def obs_to_tensor(obs, device):
    """Convert observation dict to tensors on device."""
    slice_pixels = torch.FloatTensor(obs["slice_pixels"]).unsqueeze(0).to(device)  # (1, H, W)
    prev_preds = torch.FloatTensor(obs["prev_preds"]).unsqueeze(0).to(device)  # (1, history_len, H, W)
    return slice_pixels, prev_preds

def batch_obs_to_tensor(obs_list, device):
    """Convert list of observation dicts to batched tensors."""
    slice_pixels = torch.stack([torch.FloatTensor(o["slice_pixels"]) for o in obs_list]).to(device)  # (B, H, W)
    prev_preds = torch.stack([torch.FloatTensor(o["prev_preds"]) for o in obs_list]).to(device)  # (B, history_len, H, W)
    return slice_pixels, prev_preds

def epsilon_greedy_action(q_values, epsilon):
    """Epsilon-greedy action selection.
    
    Args:
        q_values: (H, W, 2) tensor of Q-values
        epsilon: exploration probability
    
    Returns:
        action: (H, W) tensor of binary actions
    """
    if np.random.rand() < epsilon:
        H, W = q_values.shape[0], q_values.shape[1]
        return torch.randint(0, 2, (H, W), device=q_values.device)
    else:
        return q_values.argmax(dim=-1)

def compute_metrics(pred, mask):
    """Compute segmentation metrics: IoU, F1, accuracy, coverage."""
    pred_binary = (pred > 0).astype(np.float32)
    mask_binary = (mask > 0).astype(np.float32)
    
    intersection = np.logical_and(pred_binary, mask_binary).sum()
    union = np.logical_or(pred_binary, mask_binary).sum()
    
    # IoU
    iou = intersection / (union + 1e-8)
    
    # F1
    tp = intersection
    fp = (pred_binary * (1 - mask_binary)).sum()
    fn = ((1 - pred_binary) * mask_binary).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # Pixel accuracy
    correct = (pred_binary == mask_binary).sum()
    total = pred_binary.size
    accuracy = correct / total
    
    # Coverage (|pred ∧ path| / |path|)
    coverage = intersection / (mask_binary.sum() + 1e-8)
    
    return {
        "iou": iou,
        "f1": f1,
        "accuracy": accuracy,
        "coverage": coverage,
        "precision": precision,
        "recall": recall
    }

def validate(policy_net, val_pairs, device, continuity_coef=0.1, continuity_decay_factor=0.7):
    """Run validation and return average metrics, loss, and reward."""
    policy_net.eval()
    all_metrics = []
    val_rewards = []
    
    with torch.no_grad():
        for volume, mask in val_pairs:
            # Reconstruct and compute metrics
            pred = reconstruct_volume(policy_net, 
                                     volume, 
                                     mask,
                                     continuity_coef=continuity_coef,
                                     continuity_decay_factor=continuity_decay_factor,
                                     device=device)
            metrics = compute_metrics(pred, mask)
            all_metrics.append(metrics)
            
            # Compute validation reward by running through environment
            env = PathReconstructionEnv(volume=volume,
                                        mask=mask,
                                        continuity_coef=continuity_coef,
                                        continuity_decay_factor=continuity_decay_factor,
                                        history_len=5)
            obs, _ = env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                slice_pixels, prev_preds = obs_to_tensor(obs, device)
                q = policy_net(slice_pixels, prev_preds)
                a = q.argmax(dim=-1).cpu().numpy()[0]  # Remove batch dimension
                next_obs, reward, terminated, truncated, info = env.step(a.flatten())
                episode_reward += reward
                obs = next_obs
                done = terminated or truncated
            
            val_rewards.append(episode_reward)
    
    policy_net.train()
    
    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    avg_metrics["reward"] = np.mean(val_rewards)
    
    return avg_metrics

def train_dqn_on_volumes(
    volume_mask_pairs,      # list of (volume, mask) tuples
    val_pairs=None,        # validation pairs
    num_epochs=20,
    buffer_capacity=10000,
    batch_size=64,
    gamma=0.95,
    lr=1e-3,
    target_update_every=500,
    start_epsilon=1.0,
    end_epsilon=0.01,
    epsilon_decay_epochs=15,
    continuity_coef=0.1,
    continuity_decay_factor=0.7,
    seed=42,
    device=None,
):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Infer volume shape
    sample_volume, _ = volume_mask_pairs[0]
    D, H, W = sample_volume.shape

    # Networks
    policy_net = PerPixelCNNWithHistory(
        input_channels=1,
        history_len=5,
        height=H,
        width=W
    ).to(device)
    
    target_net = PerPixelCNNWithHistory(
        input_channels=1,
        history_len=5,
        height=H,
        width=W
    ).to(device)
    
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    replay = ReplayBuffer(buffer_capacity)

    epsilon = start_epsilon
    # Exponential decay per epoch: epsilon = end + (start - end) * exp(-epoch / decay_epochs)
    epsilon_decay_rate = -np.log(end_epsilon / start_epsilon) / epsilon_decay_epochs

    global_step = 0
    losses = []
    val_losses = []
    train_returns = []
    base_returns = []
    continuity_returns = []
    val_returns = []
    epsilons = []
    
    # Metrics tracking
    train_metrics_history = {
        "iou": [], "f1": [], "accuracy": [], "coverage": [], "precision": [], "recall": []
    }
    val_metrics_history = {
        "iou": [], "f1": [], "accuracy": [], "coverage": [], "precision": [], "recall": []
    }

    for epoch in range(num_epochs):
        # Update epsilon at the start of each epoch
        epsilon = end_epsilon + (start_epsilon - end_epsilon) * np.exp(-epsilon_decay_rate * epoch)
        epsilon = max(end_epsilon, epsilon)
        
        random.shuffle(volume_mask_pairs)
        epoch_return = 0.0
        t0 = time.time()

        epoch_loss = 0
        c = 0
        epoch_train_metrics = []

        for volume, mask in tqdm(volume_mask_pairs, desc=f"Epoch {epoch+1}/{num_epochs}"):
            env = PathReconstructionEnv(volume=volume, 
                                        mask=mask, 
                                        continuity_coef=continuity_coef, 
                                        continuity_decay_factor=continuity_decay_factor,
                                        history_len=5)
            obs, _ = env.reset()

            done = False
            vol_return = 0.0
            base_return = 0.0
            continuity_return = 0.0

            while not done:
                slice_pixels, prev_preds = obs_to_tensor(obs, device)
                with torch.no_grad():
                    q = policy_net(slice_pixels, prev_preds)  # (1, H, W, 2)
                    q = q.squeeze(0)  # (H, W, 2)
                a = epsilon_greedy_action(q, epsilon)  # (H, W)

                # Keep action on GPU, only convert to numpy when needed for env
                next_obs, reward, terminated, truncated, info = env.step(a.cpu().numpy().flatten())
                pixel_rewards = info["pixel_rewards"]
                base_reward = info["base_rewards"].sum()     
                continuity_reward = info["continuity_rewards"].sum()
                done = terminated or truncated
                base_return += base_reward
                continuity_return += continuity_reward
                vol_return += reward

                replay.push(
                    obs,
                    a.cpu().numpy(),
                    pixel_rewards.astype(np.float32),
                    next_obs,
                    done
                )

                obs = next_obs
                global_step += 1

                # Optimize - batched processing
                if len(replay) >= batch_size:
                    batch = replay.sample(batch_size)

                    # Batch conversion - single GPU transfer
                    slice_pixels_batch, prev_preds_batch = batch_obs_to_tensor([t.obs for t in batch], device)
                    act_batch = torch.tensor(np.stack([t.action for t in batch]), dtype=torch.int64, device=device)
                    rew_batch = torch.tensor(np.stack([t.pixel_rewards for t in batch]), dtype=torch.float32, device=device)
                    next_slice_pixels_batch, next_prev_preds_batch = batch_obs_to_tensor([t.next_obs for t in batch], device)
                    done_batch = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device)

                    # Batched forward passes
                    q_s = policy_net(slice_pixels_batch, prev_preds_batch)  # (B, H, W, 2)
                    q_s_a = q_s.gather(dim=-1, index=act_batch.unsqueeze(-1)).squeeze(-1)  # (B, H, W)

                    with torch.no_grad():
                        q_next = target_net(next_slice_pixels_batch, next_prev_preds_batch)  # (B, H, W, 2)
                        q_next_max = q_next.max(dim=-1).values  # (B, H, W)

                    not_done = (1.0 - done_batch).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
                    target = rew_batch + gamma * not_done * q_next_max  # (B, H, W)

                    loss = torch.nn.functional.mse_loss(q_s_a, target)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=5.0)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    c += 1

                if global_step % target_update_every == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            base_returns.append(base_return)
            continuity_returns.append(continuity_return)
            # print(continuity_returns)
            train_returns.append(vol_return)
            epsilons.append(epsilon)
            
            # Compute metrics for this training volume
            pred_train = reconstruct_volume(policy_net, 
                                           volume, 
                                           mask,
                                           device=device,
                                           continuity_coef=continuity_coef,
                                           continuity_decay_factor=continuity_decay_factor)
                                           
            train_metrics = compute_metrics(pred_train, mask)
            epoch_train_metrics.append(train_metrics)
            
        # Average training metrics for epoch
        avg_train_metrics = {}
        for key in epoch_train_metrics[0].keys():
            val = np.mean([m[key] for m in epoch_train_metrics])
            avg_train_metrics[key] = val
            if key in train_metrics_history:
                train_metrics_history[key].append(val)
        
        # Validation
        if val_pairs:
            avg_val_metrics = validate(
                policy_net, val_pairs, device,
                continuity_coef=continuity_coef,
                continuity_decay_factor=continuity_decay_factor
            )
            
            # Compute validation loss
            policy_net.eval()
            val_epoch_loss = 0
            val_c = 0
            
            with torch.no_grad():
                for volume, mask in val_pairs:
                    env = PathReconstructionEnv(volume=volume,
                                                mask=mask,
                                                continuity_coef=continuity_coef,
                                                continuity_decay_factor=continuity_decay_factor,
                                                history_len=5)
                    obs, _ = env.reset()
                    done = False
                    
                    while not done:
                        slice_pixels, prev_preds = obs_to_tensor(obs, device)
                        q = policy_net(slice_pixels, prev_preds)
                        a = q.argmax(dim=-1)
                        
                        next_obs, reward, terminated, truncated, info = env.step(a.cpu().numpy()[0].flatten())
                        pixel_rewards = info["pixel_rewards"]
                        done = terminated or truncated
                        
                        # Compute validation loss
                        next_slice_pixels, next_prev_preds = obs_to_tensor(next_obs, device)
                        rew_tensor = torch.tensor(pixel_rewards, dtype=torch.float32, device=device).unsqueeze(0)
                        done_tensor = torch.tensor([done], dtype=torch.float32, device=device)
                        
                        q_s_a = q.gather(dim=-1, index=a.unsqueeze(-1)).squeeze(-1)
                        q_next = target_net(next_slice_pixels, next_prev_preds)
                        q_next_max = q_next.max(dim=-1).values
                        
                        not_done = (1.0 - done_tensor).unsqueeze(-1).unsqueeze(-1)
                        target = rew_tensor + gamma * not_done * q_next_max
                        
                        val_loss = torch.nn.functional.mse_loss(q_s_a, target)
                        val_epoch_loss += val_loss.item()
                        val_c += 1
                        
                        obs = next_obs
            
            policy_net.train()
            
            # Store validation metrics
            for key in avg_val_metrics.keys():
                if key in val_metrics_history:
                    val_metrics_history[key].append(avg_val_metrics[key])
            
            val_losses.append(val_epoch_loss / val_c if val_c > 0 else 0)
            val_returns.append(avg_val_metrics["reward"])
        
        losses.append(epoch_loss / c if c > 0 else 0)
        dt = time.time() - t0  
        torch.save(target_net.state_dict(), 'dqn_slice_based/models/model.pth')
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs} | Time: {dt:.1f}s")
        print(f"  Train - Loss: {losses[-1]:.4f} | Avg Return: {np.mean(train_returns[-len(volume_mask_pairs):]):.2f} | ε: {epsilon:.3f}")
        print(f"          IoU: {avg_train_metrics['iou']:.3f} | F1: {avg_train_metrics['f1']:.3f} | Acc: {avg_train_metrics['accuracy']:.3f} | Cov: {avg_train_metrics['coverage']:.3f}")
        if val_pairs:
            print(f"  Val   - Loss: {val_losses[-1]:.4f} | Avg Return: {val_returns[-1]:.2f}")
            print(f"          IoU: {avg_val_metrics['iou']:.3f} | F1: {avg_val_metrics['f1']:.3f} | Acc: {avg_val_metrics['accuracy']:.3f} | Cov: {avg_val_metrics['coverage']:.3f}")

    return {
        "policy_net": policy_net,
        "target_net": target_net,
        "returns": train_returns,
        "base_returns": base_returns,
        "continuity_returns": continuity_returns,
        "val_returns": val_returns,
        "epsilons": epsilons,
        "losses": losses,
        "val_losses": val_losses,
        "train_metrics": train_metrics_history,
        "val_metrics": val_metrics_history,
    }

def reconstruct_volume(policy_net, 
                      volume, 
                      mask, 
                      continuity_coef=0.2, 
                      continuity_decay_factor=0.7,
                      device = None):
    env = PathReconstructionEnv(volume, 
                                mask, 
                                continuity_coef=continuity_coef, 
                                continuity_decay_factor=continuity_decay_factor,
                                history_len=5, 
                                start_from_bottom=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    obs, _ = env.reset()

    pred = np.zeros(volume.shape, dtype=np.uint8)

    done = False
    policy_net.eval()
    with torch.no_grad():
        while not done:
            slice_pixels, prev_preds = obs_to_tensor(obs, device)
            q = policy_net(slice_pixels, prev_preds)  # (1, H, W, 2)
            a = q.argmax(dim=-1).cpu().numpy()[0]  # Remove batch dim -> (H, W)
            next_obs, reward, terminated, truncated, info = env.step(a.flatten())
            # Map current slice index to volume coordinate
            slice_idx = info.get("slice_index", None)
            if slice_idx is not None:
                pred[slice_idx, :, :] = a.astype(np.uint8)
            obs = next_obs
            done = terminated or truncated
    policy_net.train()
    return pred

def visualize_result(vol, mask, pred, save_dir: str = None, slice_idx: int = 32) -> None:
    """Visualize a single slice from the volume."""
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    axs[0].imshow(vol[slice_idx], cmap="gray")
    axs[0].set_title(f"Original Volume (Slice {slice_idx})")
    axs[0].axis("off")

    axs[1].imshow(mask[slice_idx], cmap="gray")
    axs[1].set_title("Mask")
    axs[1].axis("off")

    axs[2].imshow(pred[slice_idx], cmap="gray")
    axs[2].set_title("Reconstruction")
    axs[2].axis("off")

    if save_dir:
        plt.savefig(os.path.join(save_dir, f"reconstruction_slice_{slice_idx}.png"))
    plt.show()


if __name__ == "__main__":
    if USE_ARTIFACTS: 
        intermediate_dir = "with_artifacts"
    else:
        intermediate_dir = "data/ct_like/3d_new"  # Updated directory name
        
    train_data_dir = os.path.join(intermediate_dir, "train")
    val_data_dir = os.path.join(intermediate_dir, "val")
    
    # Load training data - assuming .npy files contain 3D volumes
    train_vols = []
    train_masks = []
    for file in sorted(os.listdir(train_data_dir)):
        if file.endswith(".npy"):
            data = np.load(os.path.join(train_data_dir, file))
            if "mask" in file:
                train_masks.append(data)
            else:
                train_vols.append(data)
    
    # Load validation data
    val_vols = []
    val_masks = []
    for file in sorted(os.listdir(val_data_dir)):
        if file.endswith(".npy"):
            data = np.load(os.path.join(val_data_dir, file))
            if "mask" in file:
                val_masks.append(data)
            else:
                val_vols.append(data)
    
    results = train_dqn_on_volumes(
        list(zip(train_vols, train_masks)),
        val_pairs=list(zip(val_vols, val_masks)),
        num_epochs=5,
        continuity_coef=0.1,
        continuity_decay_factor=0.5,
        seed=123,
        start_epsilon=1.0,
        end_epsilon=0.01,
        epsilon_decay_epochs=15,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Test on first validation volume
    vol_test = val_vols[0]
    mask_test = val_masks[0]

    pred = reconstruct_volume(results["policy_net"], vol_test, mask_test)

    print(np.unique(pred))
    visualize_result(vol_test, mask_test, pred, save_dir="dqn_slice_based", slice_idx=32)

    # Plot training curves
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Returns (Train vs Val)
    axes[0, 0].plot(results["returns"], alpha=0.3, color='blue', label='Train (per volume)')
    # Moving average for train returns
    window = len(train_vols)
    train_returns_ma = [np.mean(results["returns"][max(0, i-window+1):i+1]) 
                        for i in range(len(results["returns"]))]
    axes[0, 0].plot(train_returns_ma, color='blue', linewidth=2, label='Train (epoch avg)')
    # if results["val_returns"]:
    #     axes[0, 0].plot(np.arange(len(results["val_returns"])) * len(train_vols), 
    #                     results["val_returns"], color='red', marker='o', linewidth=2, label='Val')
    axes[0, 0].set_title("Episode Returns")
    axes[0, 0].set_ylabel("Return")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Base reward
    axes[0, 1].plot(np.convolve(results["base_returns"], 
                                 np.ones(len(train_vols))/len(train_vols), 
                                 mode='valid'), 
                    label="Base Reward", color='green')
    axes[0, 1].set_title("Base Reward (Moving Average)")
    axes[0, 1].set_ylabel("Base Reward")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Continuity reward
    axes[0, 2].plot(np.convolve(results["continuity_returns"], 
                                 np.ones(len(train_vols))/len(train_vols),
                                 mode='valid'), 
                    label="Continuity Reward", color='blue')
    axes[0, 2].set_title("Continuity Reward (Moving Average)")
    axes[0, 2].set_ylabel("Continuity Reward")
    axes[0, 2].set_xlabel("Episode")
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Epsilon
    axes[1, 0].plot(results["epsilons"], color='orange', linestyle='dashed')
    axes[1, 0].set_title("Exploration (Epsilon)")
    axes[1, 0].set_ylabel("ε")
    axes[1, 0].grid(True)
    
    # Loss (Train vs Val)
    axes[1, 1].plot(results["losses"], color='red', marker='o', label='Train')
    if results["val_losses"]:
        axes[1, 1].plot(results["val_losses"], color='darkred', marker='s', label='Val')
    axes[1, 1].set_title("Training Loss")
    axes[1, 1].set_ylabel("MSE Loss")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # IoU
    axes[1, 2].plot(results["train_metrics"]["iou"], label="Train", marker='o')
    if results["val_metrics"]["iou"]:
        axes[1, 2].plot(results["val_metrics"]["iou"], label="Val", marker='s')
    axes[1, 2].set_title("IoU")
    axes[1, 2].set_ylabel("IoU")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    # # F1 Score
    # axes[1, 1].plot(results["train_metrics"]["f1"], label="Train", marker='o')
    # if results["val_metrics"]["f1"]:
    #     axes[1, 1].plot(results["val_metrics"]["f1"], label="Val", marker='s')
    # axes[1, 1].set_title("F1 Score")
    # axes[1, 1].set_ylabel("F1")
    # axes[1, 1].set_xlabel("Epoch")
    # axes[1, 1].legend()
    # axes[1, 1].grid(True)
    
    # # Accuracy
    # axes[1, 2].plot(results["train_metrics"]["accuracy"], label="Train", marker='o')
    # if results["val_metrics"]["accuracy"]:
    #     axes[1, 2].plot(results["val_metrics"]["accuracy"], label="Val", marker='s')
    # axes[1, 2].set_title("Pixel Accuracy")
    # axes[1, 2].set_ylabel("Accuracy")
    # axes[1, 2].set_xlabel("Epoch")
    # axes[1, 2].legend()
    # axes[1, 2].grid(True)
    
    # # Coverage
    # axes[2, 0].plot(results["train_metrics"]["coverage"], label="Train", marker='o')
    # if results["val_metrics"]["coverage"]:
    #     axes[2, 0].plot(results["val_metrics"]["coverage"], label="Val", marker='s')
    # axes[2, 0].set_title("Coverage")
    # axes[2, 0].set_ylabel("Coverage")
    # axes[2, 0].set_xlabel("Epoch")
    # axes[2, 0].legend()
    # axes[2, 0].grid(True)

    plt.tight_layout()
    plt.savefig(f"{os.getcwd()}/dqn_slice_based/results.png", dpi=300)
    plt.show()