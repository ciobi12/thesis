"""
DQN Slice-Based Training for CT-Like Dataset (Full 128x128x128 Volumes)

Training on complete 128x128x128 volumes without subvolume splitting.

Key differences from main.py:
1. No subvolume grid positions - each sample is a complete volume
2. No full volume reconstruction - metrics computed per-volume
3. Simplified data loading for volume/mask pairs
4. Uses full 128x128x128 volumes instead of 64x64x64 subvolumes
"""

import argparse
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm

from env import PathReconstructionEnv
from dqn import PerPixelCNNWithHistory

# ============================================================================
# Configuration
# ============================================================================

# Data Augmentation Settings
FLIP_PROB = 0.5
ROTATE_PROB = 0.5
INTENSITY_PROB = 0.3
NOISE_PROB = 0.2

VOLUME_SHAPE = (128, 128, 128)  # Full volume shape

# ============================================================================
# Data Augmentation
# ============================================================================

def augment_volume_and_mask(vol: np.ndarray, mask: np.ndarray,
                            augment_prob: float = 0.0) -> tuple:
    """
    Apply random augmentations to volume and mask.
    All geometric transforms are applied identically to both.
    """
    if random.random() > augment_prob:
        return vol, mask

    vol = vol.copy()
    mask = mask.copy()

    # 1. Random flipping along each axis
    if random.random() < FLIP_PROB:
        axis = random.choice([0, 1, 2])
        vol = np.flip(vol, axis=axis).copy()
        mask = np.flip(mask, axis=axis).copy()

    # 2. Random 90-degree rotations in H-W plane
    if random.random() < ROTATE_PROB:
        k = random.choice([1, 2, 3])
        vol = np.rot90(vol, k=k, axes=(1, 2)).copy()
        mask = np.rot90(mask, k=k, axes=(1, 2)).copy()

    # 3. Random intensity scaling (only for volume)
    if random.random() < INTENSITY_PROB:
        scale = random.uniform(0.8, 1.2)
        shift = random.uniform(-0.1, 0.1)
        vol = np.clip(vol * scale + shift, 0, 1)

    # 4. Random Gaussian noise (only for volume)
    if random.random() < NOISE_PROB:
        noise_std = random.uniform(0.01, 0.05)
        noise = np.random.normal(0, noise_std, vol.shape).astype(vol.dtype)
        vol = np.clip(vol + noise, 0, 1)

    return vol, mask


# ============================================================================
# Data Loading
# ============================================================================

def load_volume_pairs(data_dir: str, min_fg_ratio: float = 0.0):
    """
    Load volume/mask pairs from directory (supports both full volumes and subvolumes).

    Expects directory structure (subvolumes):
        data_dir/
            vol_000/
                volumes/
                    subvol_d00_h00_w00.npy
                    ...
                masks/
                    subvol_d00_h00_w00.npy
                    ...
            vol_001/
                ...

    OR (full volumes):
        data_dir/
            volumes/
                root_volume_XXX.npy
            masks/
                root_mask_XXX.npy

    Args:
        data_dir: Path to directory containing either vol_XXX/ subdirectories or volumes/masks/ subdirectories
        min_fg_ratio: Minimum foreground ratio to include volume

    Returns:
        volumes: List of volume arrays
        masks: List of mask arrays
    """
    volumes = []
    masks = []

    # Check if this is a subvolume directory structure (has vol_XXX subdirectories)
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("vol_")]

    if subdirs:
        # Load subvolumes from vol_XXX subdirectories
        print(f"Found {len(subdirs)} volume directories with subvolumes")

        n_skipped_no_mask = 0
        n_skipped_empty = 0

        for vol_dir in sorted(subdirs):
            vol_dir_path = os.path.join(data_dir, vol_dir)
            volumes_dir = os.path.join(vol_dir_path, "volumes")
            masks_dir = os.path.join(vol_dir_path, "masks")

            if not os.path.exists(volumes_dir) or not os.path.exists(masks_dir):
                print(f"Skipping {vol_dir}: missing volumes/ or masks/ subdirectory")
                continue

            # Load all subvolumes from this directory
            vol_files = sorted([f for f in os.listdir(volumes_dir) if f.endswith(".npy")])

            for vol_file in vol_files:
                vol_path = os.path.join(volumes_dir, vol_file)
                mask_path = os.path.join(masks_dir, vol_file)  # Same filename for mask

                if not os.path.exists(mask_path):
                    n_skipped_no_mask += 1
                    continue

                # Load mask first to check foreground ratio
                mask = np.load(mask_path)
                fg_ratio = mask.mean()

                if fg_ratio < min_fg_ratio:
                    n_skipped_empty += 1
                    continue

                vol = np.load(vol_path)

                # Normalize volume to [0, 1] if needed
                if vol.max() > 1:
                    vol = vol.astype(np.float32) / 255.0

                volumes.append(vol)
                masks.append(mask)

                print(f"Loaded: {vol_dir}/{vol_file} (fg={fg_ratio*100:.4f}%)")

        print(f"\nSummary: Loaded {len(volumes)} subvolumes")
        if n_skipped_no_mask > 0:
            print(f"  Skipped {n_skipped_no_mask} (no matching mask)")
        if n_skipped_empty > 0:
            print(f"  Skipped {n_skipped_empty} (fg_ratio < {min_fg_ratio*100:.2f}%)")

    else:
        # Load full volumes from volumes/ and masks/ subdirectories (original behavior)
        volumes_dir = os.path.join(data_dir, "volumes")
        masks_dir = os.path.join(data_dir, "masks")

        if not os.path.exists(volumes_dir) or not os.path.exists(masks_dir):
            print(f"Error: Expected volumes/ and masks/ subdirectories in {data_dir}")
            return volumes, masks

        vol_files = sorted([f for f in os.listdir(volumes_dir) if f.endswith(".npy")])

        n_skipped_no_mask = 0
        n_skipped_empty = 0

        for vol_file in vol_files:
            vol_path = os.path.join(volumes_dir, vol_file)
            # Map volume filename to mask filename
            mask_file = vol_file.replace("volume", "mask")
            mask_path = os.path.join(masks_dir, mask_file)

            if not os.path.exists(mask_path):
                n_skipped_no_mask += 1
                continue

            # Load mask first to check foreground ratio
            mask = np.load(mask_path)
            fg_ratio = mask.mean()

            if fg_ratio < min_fg_ratio:
                n_skipped_empty += 1
                continue

            vol = np.load(vol_path)

            # Normalize volume to [0, 1] if needed
            if vol.max() > 1:
                vol = vol.astype(np.float32) / 255.0

            volumes.append(vol)
            masks.append(mask)

            print(f"Loaded: {vol_file} (fg={fg_ratio*100:.4f}%)")

        print(f"\nSummary: Loaded {len(volumes)} volumes")
        if n_skipped_no_mask > 0:
            print(f"  Skipped {n_skipped_no_mask} (no matching mask)")
        if n_skipped_empty > 0:
            print(f"  Skipped {n_skipped_empty} (fg_ratio < {min_fg_ratio*100:.2f}%)")

    return volumes, masks


def compute_metrics(pred: np.ndarray, mask: np.ndarray) -> dict:
    """Compute comprehensive metrics for a volume."""
    pred_flat = pred.flatten().astype(bool)
    mask_flat = mask.flatten().astype(bool)

    tp = np.logical_and(pred_flat, mask_flat).sum()
    fp = np.logical_and(pred_flat, ~mask_flat).sum()
    fn = np.logical_and(~pred_flat, mask_flat).sum()
    tn = np.logical_and(~pred_flat, ~mask_flat).sum()

    eps = 1e-8

    dice = 2 * tp / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn)
    }


# ============================================================================
# Training Utilities
# ============================================================================

def obs_to_tensor(obs, device):
    """Convert observation dict to tensors for network input."""
    slice_pixels = torch.from_numpy(obs["slice_pixels"]).float().unsqueeze(0).to(device)
    prev_slices = torch.from_numpy(obs["prev_slices"]).float().unsqueeze(0).to(device)
    future_slices = torch.from_numpy(obs["future_slices"]).float().unsqueeze(0).to(device)
    return slice_pixels, prev_slices, future_slices


def epsilon_greedy_action(q_values, epsilon, n_pixels):
    """Select actions using epsilon-greedy policy."""
    if random.random() < epsilon:
        return np.random.randint(0, 2, size=n_pixels)
    else:
        return q_values.argmax(dim=-1).cpu().numpy()[0].flatten()


def evaluate_model(policy_net, volumes, masks, device, history_len=5, future_len=5,
                   continuity_coef=0.2, continuity_decay_factor=0.7,
                   gradient_coef=0.1, dice_coef=1.0):
    """
    Evaluate model on a set of volumes.

    Returns:
        avg_metrics: Dictionary of average metrics across all volumes
        all_preds: List of predictions for each volume
    """
    policy_net.eval()
    all_metrics = []
    all_preds = []
    H, W = volumes[0].shape[1], volumes[0].shape[2]

    with torch.no_grad():
        for vol, mask in zip(volumes, masks):
            env = PathReconstructionEnv(
                vol, mask,
                continuity_coef=continuity_coef,
                continuity_decay_factor=continuity_decay_factor,
                gradient_coef=gradient_coef,
                dice_coef=dice_coef,
                history_len=history_len,
                future_len=future_len,
            )

            obs, _ = env.reset()
            done = False
            pred = np.zeros(vol.shape, dtype=np.uint8)

            while not done:
                slice_pixels, prev_slices, future_slices = obs_to_tensor(obs, device)
                q = policy_net(slice_pixels, prev_slices, future_slices)
                a = q.argmax(dim=-1).cpu().numpy()[0]

                next_obs, _, terminated, truncated, info = env.step(a.flatten())

                slice_idx = info.get("slice_index", None)
                if slice_idx is not None:
                    pred[slice_idx, :, :] = a.astype(np.uint8)

                obs = next_obs
                done = terminated or truncated

            metrics = compute_metrics(pred, mask)
            all_metrics.append(metrics)
            all_preds.append(pred)

    policy_net.train()

    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        if key in ["tp", "fp", "fn"]:
            avg_metrics[key] = sum(m[key] for m in all_metrics)
        else:
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    return avg_metrics, all_preds


# ============================================================================
# Main Training Loop
# ============================================================================

def train_dqn(
    train_volumes: list,
    train_masks: list,
    val_volumes: list = None,
    val_masks: list = None,
    # Hyperparameters
    aug_prob: float = 0.0,
    num_epochs: int = 100,
    eval_interval: int = 5,
    lr: float = 1e-3,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay_steps: int = 50000,
    replay_buffer_size: int = 50000,
    batch_size: int = 64,
    target_update_freq: int = 500,
    continuity_coef: float = 0.2,
    continuity_decay_factor: float = 0.7,
    gradient_coef: float = 0.1,
    dice_coef: float = 1.0,
    manhattan_coef: float = 0.0,
    future_len: int = 5,
    device: str = None,
    save_dir: str = "models",
    early_stopping_patience: int = 100,
):
    """
    Train DQN on full volumes with periodic evaluation.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Get dimensions from first volume
    D, H, W = train_volumes[0].shape
    history_len = 5

    # Use smaller network for small inputs
    small_input = (H <= 64 and W <= 64)
    print(f"Input size: {H}x{W} -> Using {'small' if small_input else 'standard'} network architecture")

    # Initialize networks
    policy_net = PerPixelCNNWithHistory(
        input_channels=1,
        history_len=history_len,
        height=H,
        width=W,
        future_len=future_len,
        small_input=small_input
    ).to(device)

    target_net = PerPixelCNNWithHistory(
        input_channels=1,
        history_len=history_len,
        height=H,
        width=W,
        future_len=future_len,
        small_input=small_input
    ).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    n_params = sum(p.numel() for p in policy_net.parameters() if p.requires_grad)
    print(f"Network parameters: {n_params:,}")

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = deque(maxlen=replay_buffer_size)

    # Training tracking
    epsilon = epsilon_start
    global_step = 0

    losses = []
    train_returns = []
    base_returns = []
    continuity_returns = []
    gradient_returns = []
    manhattan_returns = []
    dice_returns = []
    epsilons = []

    # Metrics history
    train_metrics_history = []
    val_metrics_history = []
    val_losses = []

    best_val_dice = 0.0
    best_val_pred = None
    best_epoch = 0

    # Early stopping
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    early_stopped = False

    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("Starting DQN training on CT-Like dataset")
    print(f"  - Training volumes: {len(train_volumes)}")
    print(f"  - Validation volumes: {len(val_volumes) if val_volumes else 0}")
    print(f"  - Volume shape: {train_volumes[0].shape}")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Eval every: {eval_interval} epochs")
    print(f"  - Continuity coef: {continuity_coef} (decay: {continuity_decay_factor})")
    print(f"  - Gradient coef: {gradient_coef}")
    print(f"  - Manhattan coef: {manhattan_coef}")
    print(f"  - DICE coefficient: {dice_coef}")
    print(f"  - Data augmentation prob: {aug_prob}")
    print(f"  - Early stopping patience: {early_stopping_patience} epochs")
    print(f"{'='*60}\n")

    for epoch in range(num_epochs):
        t0 = time.time()
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_returns = []
        epoch_base_returns = []
        epoch_continuity_returns = []
        epoch_gradient_returns = []
        epoch_manhattan_returns = []
        epoch_dice_returns = []

        # Shuffle training data
        indices = list(range(len(train_volumes)))
        random.shuffle(indices)

        # Train on all volumes
        for idx in tqdm(indices, desc=f"Epoch {epoch+1}/{num_epochs}"):
            vol = train_volumes[idx]
            mask = train_masks[idx]

            # Apply data augmentation
            vol_aug, mask_aug = augment_volume_and_mask(vol, mask, aug_prob)

            env = PathReconstructionEnv(
                vol_aug, mask_aug,
                continuity_coef=continuity_coef,
                continuity_decay_factor=continuity_decay_factor,
                gradient_coef=gradient_coef,
                dice_coef=dice_coef,
                manhattan_coef=manhattan_coef,
                history_len=history_len,
                future_len=future_len,
            )

            obs, _ = env.reset()
            done = False
            episode_reward = 0.0
            episode_base_reward = 0.0
            episode_continuity_reward = 0.0
            episode_gradient_reward = 0.0
            episode_manhattan_reward = 0.0
            episode_dice_reward = 0.0
            n_pixels = H * W

            while not done:
                slice_pixels, prev_slices, future_slices = obs_to_tensor(obs, device)

                with torch.no_grad():
                    q = policy_net(slice_pixels, prev_slices, future_slices)

                action = epsilon_greedy_action(q, epsilon, n_pixels)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                pixel_rewards = info["pixel_rewards"]
                episode_reward += reward
                episode_base_reward += info["base_rewards"].sum()
                episode_continuity_reward += info["continuity_rewards"].sum()
                episode_gradient_reward += info["gradient_rewards"].sum()
                episode_manhattan_reward += info.get("manhattan_reward", 0.0)

                # Store transition
                replay_buffer.append((
                    obs["slice_pixels"].copy(),
                    obs["prev_slices"].copy(),
                    obs["future_slices"].copy(),
                    action.copy(),
                    pixel_rewards.copy(),
                    next_obs["slice_pixels"].copy(),
                    next_obs["prev_slices"].copy(),
                    next_obs["future_slices"].copy(),
                    done
                ))

                obs = next_obs
                global_step += 1

                # Update epsilon (exponential decay)
                decay_rate = 1.0 / epsilon_decay_steps
                epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-decay_rate * global_step)
                epsilon = max(epsilon, epsilon_end)

                # Training step
                if len(replay_buffer) >= batch_size:
                    batch = random.sample(replay_buffer, batch_size)

                    sp_batch = torch.tensor(np.array([t[0] for t in batch]), dtype=torch.float32, device=device)
                    ps_batch = torch.tensor(np.array([t[1] for t in batch]), dtype=torch.float32, device=device)
                    fs_batch = torch.tensor(np.array([t[2] for t in batch]), dtype=torch.float32, device=device)
                    a_batch = torch.tensor(np.array([t[3] for t in batch]), dtype=torch.long, device=device)
                    r_batch = torch.tensor(np.array([t[4] for t in batch]), dtype=torch.float32, device=device)
                    next_sp_batch = torch.tensor(np.array([t[5] for t in batch]), dtype=torch.float32, device=device)
                    next_ps_batch = torch.tensor(np.array([t[6] for t in batch]), dtype=torch.float32, device=device)
                    next_fs_batch = torch.tensor(np.array([t[7] for t in batch]), dtype=torch.float32, device=device)
                    done_batch = torch.tensor(np.array([t[8] for t in batch]), dtype=torch.float32, device=device)

                    # Current Q values
                    q_values = policy_net(sp_batch, ps_batch, fs_batch)

                    a_batch_2d = a_batch.view(batch_size, H, W)
                    q_sa = q_values.gather(dim=-1, index=a_batch_2d.unsqueeze(-1)).squeeze(-1)

                    # Target Q values
                    with torch.no_grad():
                        q_next = target_net(next_sp_batch, next_ps_batch, next_fs_batch)
                        q_next_max = q_next.max(dim=-1).values
                        not_done = (1.0 - done_batch).unsqueeze(-1).unsqueeze(-1)
                        r_batch_2d = r_batch.view(batch_size, H, W)
                        target = r_batch_2d + gamma * not_done * q_next_max

                    loss = nn.functional.mse_loss(q_sa, target)

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=5.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    epoch_steps += 1

                # Update target network
                if global_step % target_update_freq == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            epoch_returns.append(episode_reward)
            epoch_base_returns.append(episode_base_reward)
            epoch_continuity_returns.append(episode_continuity_reward)
            epoch_gradient_returns.append(episode_gradient_reward)
            epoch_manhattan_returns.append(episode_manhattan_reward)

            if info.get("episode_dice") is not None:
                episode_dice_reward = info.get("dice_reward", 0.0)
                epoch_dice_returns.append(episode_dice_reward)

        # Store epoch results
        losses.append(epoch_loss / max(epoch_steps, 1))
        train_returns.extend(epoch_returns)
        base_returns.extend(epoch_base_returns)
        continuity_returns.extend(epoch_continuity_returns)
        gradient_returns.extend(epoch_gradient_returns)
        manhattan_returns.extend(epoch_manhattan_returns)
        dice_returns.extend(epoch_dice_returns)
        epsilons.append(epsilon)

        # ====================================================================
        # EVALUATION every N epochs
        # ====================================================================
        if (epoch + 1) % eval_interval == 0:
            print(f"\n[Epoch {epoch+1}] Evaluating on train and validation sets...")

            # Train evaluation
            train_metrics, _ = evaluate_model(
                policy_net, train_volumes, train_masks, device,
                history_len=history_len, future_len=future_len,
                continuity_coef=continuity_coef,
                continuity_decay_factor=continuity_decay_factor,
                gradient_coef=gradient_coef, dice_coef=dice_coef
            )
            train_metrics_history.append(train_metrics)

            print(f"  [TRAIN] DICE: {train_metrics['dice']:.4f} | IoU: {train_metrics['iou']:.4f}")
            print(f"  [TRAIN] Precision: {train_metrics['precision']:.4f} | Recall: {train_metrics['recall']:.4f}")

            # Validation evaluation
            if val_volumes:
                val_metrics, val_preds = evaluate_model(
                    policy_net, val_volumes, val_masks, device,
                    history_len=history_len, future_len=future_len,
                    continuity_coef=continuity_coef,
                    continuity_decay_factor=continuity_decay_factor,
                    gradient_coef=gradient_coef, dice_coef=dice_coef
                )
                val_metrics_history.append(val_metrics)

                print(f"  [VAL] DICE: {val_metrics['dice']:.4f} | IoU: {val_metrics['iou']:.4f}")
                print(f"  [VAL] Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f}")

                # Compute validation loss
                val_loss = 0.0
                val_loss_steps = 0
                policy_net.eval()

                with torch.no_grad():
                    for val_vol, val_mask in zip(val_volumes, val_masks):
                        val_env = PathReconstructionEnv(
                            val_vol, val_mask,
                            continuity_coef=continuity_coef,
                            continuity_decay_factor=continuity_decay_factor,
                            gradient_coef=gradient_coef,
                            dice_coef=dice_coef,
                            history_len=history_len,
                            future_len=future_len,
                        )
                        val_obs, _ = val_env.reset()
                        val_done = False

                        while not val_done:
                            val_sp, val_ps, val_fs = obs_to_tensor(val_obs, device)
                            val_q = policy_net(val_sp, val_ps, val_fs)
                            val_action = val_q.argmax(dim=-1).cpu().numpy()[0].flatten()

                            val_next_obs, _, val_term, val_trunc, val_info = val_env.step(val_action)
                            val_done = val_term or val_trunc

                            val_pixel_rewards = val_info["pixel_rewards"]
                            val_r = torch.tensor(val_pixel_rewards, dtype=torch.float32, device=device).view(1, H, W)
                            val_a = torch.tensor(val_action, dtype=torch.long, device=device).view(1, H, W)

                            val_q_sa = val_q.gather(dim=-1, index=val_a.unsqueeze(-1)).squeeze(-1)

                            if not val_done:
                                val_next_sp, val_next_ps, val_next_fs = obs_to_tensor(val_next_obs, device)
                                val_q_next = policy_net(val_next_sp, val_next_ps, val_next_fs)
                                val_q_next_max = val_q_next.max(dim=-1).values
                                val_target = val_r + gamma * val_q_next_max
                            else:
                                val_target = val_r

                            val_step_loss = nn.functional.mse_loss(val_q_sa, val_target)
                            val_loss += val_step_loss.item()
                            val_loss_steps += 1

                            val_obs = val_next_obs

                policy_net.train()
                avg_val_loss = val_loss / max(val_loss_steps, 1)
                val_losses.append(avg_val_loss)
                print(f"  [VAL] Loss: {avg_val_loss:.6f}")

                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    print(f"  [EARLY STOPPING] No improvement for {epochs_without_improvement}/{early_stopping_patience} epochs")

                if epochs_without_improvement >= early_stopping_patience:
                    print(f"\n{'='*60}")
                    print(f"EARLY STOPPING: Validation loss hasn't improved for {early_stopping_patience} epochs")
                    print(f"{'='*60}")
                    early_stopped = True

                # Track best model
                if val_metrics["dice"] > best_val_dice:
                    best_val_dice = val_metrics["dice"]
                    best_val_pred = val_preds
                    best_epoch = epoch + 1
                    best_model_path = os.path.join(save_dir, "best_model.pth")
                    torch.save(policy_net.state_dict(), best_model_path)
                    print(f"  [NEW BEST] Saved best model (val DICE={best_val_dice:.4f})")

        # Print epoch summary
        dt = time.time() - t0
        avg_return = np.mean(epoch_returns)
        avg_base = np.mean(epoch_base_returns)
        avg_cont = np.mean(epoch_continuity_returns)
        avg_grad = np.mean(epoch_gradient_returns)
        avg_dice_rew = np.mean(epoch_dice_returns) if epoch_dice_returns else 0.0
        print(f"Epoch {epoch+1}/{num_epochs} | Time: {dt:.1f}s | Loss: {losses[-1]:.4f} | ε: {epsilon:.3f}")
        print(f"  Returns - Total: {avg_return:.2f} | Base: {avg_base:.2f} | Cont: {avg_cont:.2f} | Grad: {avg_grad:.2f} | DICE: {avg_dice_rew:.2f} | Manhattan: {np.mean(epoch_manhattan_returns):.2f}")

        # Save latest model
        torch.save(policy_net.state_dict(), os.path.join(save_dir, "model_latest.pth"))

        if early_stopped:
            print(f"\nTraining stopped early at epoch {epoch + 1}")
            break

    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)

    final_train_metrics, _ = evaluate_model(
        policy_net, train_volumes, train_masks, device,
        history_len=history_len, future_len=future_len,
        continuity_coef=continuity_coef, continuity_decay_factor=continuity_decay_factor,
        gradient_coef=gradient_coef, dice_coef=dice_coef
    )

    print(f"Final Train DICE: {final_train_metrics['dice']:.4f}")
    print(f"Final Train IoU:  {final_train_metrics['iou']:.4f}")

    if val_volumes:
        final_val_metrics, _ = evaluate_model(
            policy_net, val_volumes, val_masks, device,
            history_len=history_len, future_len=future_len,
            continuity_coef=continuity_coef, continuity_decay_factor=continuity_decay_factor,
            gradient_coef=gradient_coef, dice_coef=dice_coef
        )
        print(f"Final Val DICE: {final_val_metrics['dice']:.4f}")
        print(f"Final Val IoU:  {final_val_metrics['iou']:.4f}")

    return {
        "policy_net": policy_net,
        "target_net": target_net,
        "train_metrics_history": train_metrics_history,
        "val_metrics_history": val_metrics_history,
        "val_losses": val_losses,
        "best_val_dice": best_val_dice,
        "best_val_pred": best_val_pred,
        "best_epoch": best_epoch,
        "losses": losses,
        "train_returns": train_returns,
        "base_returns": base_returns,
        "continuity_returns": continuity_returns,
        "gradient_returns": gradient_returns,
        "manhattan_returns": manhattan_returns,
        "dice_returns": dice_returns,
        "epsilons": epsilons,
        "n_train_vols": len(train_volumes),
        "eval_interval": eval_interval,
    }


# ============================================================================
# Visualization
# ============================================================================

def plot_training_results(results: dict, save_dir: str = "results"):
    """Plot comprehensive training results."""
    os.makedirs(save_dir, exist_ok=True)

    n_train_vols = results.get("n_train_vols", 1)
    eval_interval = results.get("eval_interval", 5)
    window = n_train_vols

    # ========================================================================
    # FIGURE 1: General Training Results (2x3 layout)
    # ========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    # Episode Returns
    axes[0, 0].plot(results["train_returns"], alpha=0.3, color='blue', label='Train (per volume)')
    if len(results["train_returns"]) >= window:
        train_returns_ma = [np.mean(results["train_returns"][max(0, i-window+1):i+1])
                           for i in range(len(results["train_returns"]))]
        axes[0, 0].plot(train_returns_ma, color='blue', linewidth=2, label='Train (moving average)')
    axes[0, 0].set_title("Episode Returns")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Epsilon
    axes[0, 1].plot(results["epsilons"], color='orange', linestyle='dashed')
    axes[0, 1].set_title("Exploration (Epsilon)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].grid(True)

    # Loss
    axes[0, 2].plot(results["losses"], color='red', marker='o', markersize=4, label='Train')
    if results.get("val_losses") and len(results["val_losses"]) > 0:
        val_epochs = list(range(eval_interval, len(results["val_losses"]) * eval_interval + 1, eval_interval))
        axes[0, 2].plot(val_epochs, results["val_losses"], color='darkred', marker='s', markersize=2, label='Val')
    axes[0, 2].set_title("MSE Loss per Epoch")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # IoU
    if results["train_metrics_history"]:
        epochs = list(range(eval_interval, len(results["train_metrics_history"]) * eval_interval + 1, eval_interval))
        train_iou = [m["iou"] for m in results["train_metrics_history"]]
        axes[1, 0].plot(epochs, train_iou, label="Train", marker='o', markersize=2)

        if results.get("val_metrics_history"):
            val_iou = [m["iou"] for m in results["val_metrics_history"]]
            val_epochs = epochs[:len(val_iou)]
            axes[1, 0].plot(val_epochs, val_iou, label="Val", marker='s', markersize=2)
    axes[1, 0].set_title("IoU")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # DICE
    if results["train_metrics_history"]:
        epochs = list(range(eval_interval, len(results["train_metrics_history"]) * eval_interval + 1, eval_interval))
        train_dice = [m["dice"] for m in results["train_metrics_history"]]
        axes[1, 1].plot(epochs, train_dice, label="Train", marker='o', markersize=2)

        if results.get("val_metrics_history"):
            val_dice = [m["dice"] for m in results["val_metrics_history"]]
            val_epochs = epochs[:len(val_dice)]
            axes[1, 1].plot(val_epochs, val_dice, label="Val", marker='s', markersize=2)
    axes[1, 1].set_title("DICE Score")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Recall
    if results["train_metrics_history"]:
        epochs = list(range(eval_interval, len(results["train_metrics_history"]) * eval_interval + 1, eval_interval))
        train_recall = [m["recall"] for m in results["train_metrics_history"]]
        axes[1, 2].plot(epochs, train_recall, label="Train", marker='o', markersize=2)

        if results.get("val_metrics_history"):
            val_recall = [m["recall"] for m in results["val_metrics_history"]]
            val_epochs = epochs[:len(val_recall)]
            axes[1, 2].plot(val_epochs, val_recall, label="Val", marker='s', markersize=2)
    axes[1, 2].set_title("Coverage (Recall)")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_results.png"), dpi=300)
    plt.show()

    # ========================================================================
    # FIGURE 2: Reward Components (2x2 layout)
    # ========================================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

    # Base Reward
    if results.get("base_returns") and len(results["base_returns"]) >= window:
        base_ma = np.convolve(results["base_returns"], np.ones(window)/window, mode='valid')
        axes2[0, 0].plot(base_ma, label="Base Reward", color='green')
    else:
        axes2[0, 0].plot(results.get("base_returns", []), label="Base Reward", color='green', alpha=0.5)
    axes2[0, 0].set_title("Base Reward (Moving Average)")
    axes2[0, 0].set_ylabel("Reward")
    axes2[0, 0].set_xlabel("Episode")
    axes2[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes2[0, 0].grid(True)

    # Continuity Reward
    if results.get("continuity_returns") and len(results["continuity_returns"]) >= window:
        cont_ma = np.convolve(results["continuity_returns"], np.ones(window)/window, mode='valid')
        axes2[0, 1].plot(cont_ma, label="Continuity Reward", color='blue')
    else:
        axes2[0, 1].plot(results.get("continuity_returns", []), label="Continuity Reward", color='blue', alpha=0.5)
    axes2[0, 1].set_title("Continuity Reward (Moving Average)")
    axes2[0, 1].set_ylabel("Reward")
    axes2[0, 1].set_xlabel("Episode")
    axes2[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes2[0, 1].grid(True)

    # Gradient Reward
    if results.get("gradient_returns") and len(results["gradient_returns"]) >= window:
        gradient_returns = np.array(results.get("gradient_returns", []))
        if gradient_returns.ndim > 1:
            gradient_returns = gradient_returns.flatten()
        if len(gradient_returns) >= window:
            grad_ma = np.convolve(gradient_returns, np.ones(window)/window, mode='valid')
            axes2[1, 0].plot(grad_ma, label="Gradient Reward", color='orange')
        else:
            axes2[1, 0].plot(gradient_returns, label="Gradient Reward", color='orange', alpha=0.5)
    axes2[1, 0].set_title("Gradient Reward (Moving Average)")
    axes2[1, 0].set_ylabel("Reward")
    axes2[1, 0].set_xlabel("Episode")
    axes2[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes2[1, 0].grid(True)

    # Manhattan Reward
    if results.get("manhattan_returns") and len(results["manhattan_returns"]) >= window:
        manh_ma = np.convolve(results["manhattan_returns"], np.ones(window)/window, mode='valid')
        axes2[1, 1].plot(manh_ma, label="Manhattan Reward", color='red')
    else:
        axes2[1, 1].plot(results.get("manhattan_returns", []), label="Manhattan Reward", color='red', alpha=0.5)
    axes2[1, 1].set_title("Manhattan Reward (Moving Average)")
    axes2[1, 1].set_ylabel("Reward")
    axes2[1, 1].set_xlabel("Episode")
    axes2[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes2[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "reward_components_analysis.png"), dpi=300)
    plt.show()

    # ========================================================================
    # FIGURE 3: Train vs Val Metrics
    # ========================================================================
    if results["train_metrics_history"] and results.get("val_metrics_history"):
        fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))

        epochs = list(range(eval_interval, len(results["train_metrics_history"]) * eval_interval + 1, eval_interval))
        val_epochs = epochs[:len(results["val_metrics_history"])]

        # DICE
        train_dice = [m["dice"] for m in results["train_metrics_history"]]
        val_dice = [m["dice"] for m in results["val_metrics_history"]]
        axes3[0, 0].plot(epochs, train_dice, 'b-o', label='Train', linewidth=2)
        axes3[0, 0].plot(val_epochs, val_dice, 'r-s', label='Val', linewidth=2)
        axes3[0, 0].set_title("DICE Score")
        axes3[0, 0].set_ylabel("DICE")
        axes3[0, 0].set_xlabel("Epoch")
        axes3[0, 0].legend()
        axes3[0, 0].grid(True)

        # IoU
        train_iou = [m["iou"] for m in results["train_metrics_history"]]
        val_iou = [m["iou"] for m in results["val_metrics_history"]]
        axes3[0, 1].plot(epochs, train_iou, 'b-o', label='Train', linewidth=2)
        axes3[0, 1].plot(val_epochs, val_iou, 'r-s', label='Val', linewidth=2)
        axes3[0, 1].set_title("IoU Score")
        axes3[0, 1].set_ylabel("IoU")
        axes3[0, 1].set_xlabel("Epoch")
        axes3[0, 1].legend()
        axes3[0, 1].grid(True)

        # Precision
        train_precision = [m["precision"] for m in results["train_metrics_history"]]
        val_precision = [m["precision"] for m in results["val_metrics_history"]]
        axes3[1, 0].plot(epochs, train_precision, 'b-o', label='Train', linewidth=2)
        axes3[1, 0].plot(val_epochs, val_precision, 'r-s', label='Val', linewidth=2)
        axes3[1, 0].set_title("Precision")
        axes3[1, 0].set_ylabel("Precision")
        axes3[1, 0].set_xlabel("Epoch")
        axes3[1, 0].legend()
        axes3[1, 0].grid(True)

        # Recall
        train_recall = [m["recall"] for m in results["train_metrics_history"]]
        val_recall = [m["recall"] for m in results["val_metrics_history"]]
        axes3[1, 1].plot(epochs, train_recall, 'b-o', label='Train', linewidth=2)
        axes3[1, 1].plot(val_epochs, val_recall, 'r-s', label='Val', linewidth=2)
        axes3[1, 1].set_title("Recall")
        axes3[1, 1].set_ylabel("Recall")
        axes3[1, 1].set_xlabel("Epoch")
        axes3[1, 1].legend()
        axes3[1, 1].grid(True)

        plt.suptitle("Train vs Validation Metrics", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "global_metrics.png"), dpi=300)
        plt.show()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN training for CT-Like dataset (full 128x128x128 volumes)")
    parser.add_argument("--data_dir", type=str, default="../../data/ct_like/3d",
                        help="Directory containing train/val subdirectories")
    parser.add_argument("--min_fg_ratio", type=float, default=0.0001,
                        help="Minimum foreground ratio to include volume")
    parser.add_argument("--aug_prob", type=float, default=0.0,
                        help="Probability of applying data augmentation")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--cont", type=float, default=0.1,
                        help="Continuity reward coefficient")
    parser.add_argument("--grad", type=float, default=0.1,
                        help="Gradient reward coefficient")
    parser.add_argument("--manhattan", type=float, default=0.1,
                        help="Manhattan distance reward coefficient")
    parser.add_argument("--dice", type=float, default=1.0,
                        help="DICE reward coefficient")
    parser.add_argument("--eval_interval", type=int, default=1,
                        help="Evaluate every N epochs")
    args = parser.parse_args()

    data_dir = args.data_dir

    # Load training volumes
    print("Loading training volumes...")
    train_vols, train_masks = load_volume_pairs(
        os.path.join(data_dir, "train"),
        min_fg_ratio=args.min_fg_ratio
    )
    print(f"Loaded {len(train_vols)} training volumes")

    # Load validation volumes
    print("\nLoading validation volumes...")
    val_vols, val_masks = load_volume_pairs(
        os.path.join(data_dir, "val"),
        min_fg_ratio=args.min_fg_ratio
    )
    print(f"Loaded {len(val_vols)} validation volumes")

    # Create save directory
    save_dir = os.path.join(f"results/ct_like_full",
                            f"cont_{args.cont}_grad_{args.grad}_manh_{args.manhattan}")
    os.makedirs(save_dir, exist_ok=True)

    # Train
    results = train_dqn(
        train_volumes=train_vols,
        train_masks=train_masks,
        val_volumes=val_vols,
        val_masks=val_masks,
        aug_prob=args.aug_prob,
        num_epochs=args.epochs,
        eval_interval=args.eval_interval,
        lr=1e-3,
        gamma=0.99,
        batch_size=64,
        continuity_coef=args.cont,
        continuity_decay_factor=0.5,
        dice_coef=args.dice,
        gradient_coef=args.grad,
        manhattan_coef=args.manhattan,
        future_len=5,
        save_dir=save_dir
    )

    # Plot results
    plot_training_results(results, save_dir=save_dir)

    print(f"\nTraining complete. Results saved to: {save_dir}")
    print(f"Best validation DICE: {results['best_val_dice']:.4f} (epoch {results['best_epoch']})")
