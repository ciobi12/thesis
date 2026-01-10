"""
DQN Slice-Based Training with Global DICE Evaluation

Key improvements over main.py:
1. Tracks subvolume grid positions for reconstruction
2. Every 5 epochs: reconstructs FULL volume from subvolume predictions
3. Computes global DICE score against full ground truth mask
4. Logs global metrics separately from local per-subvolume metrics
5. Uses DICE-based reward in environment for better class imbalance handling
"""

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

from env_dice import PathReconstructionEnv
from dqn import PerPixelCNNWithHistory

# ============================================================================
# Configuration
# ============================================================================

# Data Augmentation Settings
AUGMENT_PROB = 0.1  # Probability of applying any augmentation (LOW FOR NOW)
FLIP_PROB = 0.5     # Probability of each flip axis
ROTATE_PROB = 0.5   # Probability of rotation
INTENSITY_PROB = 0.3  # Probability of intensity augmentation
NOISE_PROB = 0.2    # Probability of adding noise

SUBVOL_SHAPE = (32, 32, 32)  # Must match preprocessing
# Full volume: (800, 466, 471) -> Grid: 8x7x7 subvolumes
FULL_VOLUME_SHAPE = (800, 448, 448)

# ============================================================================
# Data Augmentation
# ============================================================================

def augment_volume_and_mask(vol: np.ndarray, mask: np.ndarray, 
                            augment_prob: float = AUGMENT_PROB) -> tuple:
    """
    Apply random augmentations to volume and mask.
    All geometric transforms are applied identically to both.
    
    Args:
        vol: Input volume (D, H, W)
        mask: Binary mask (D, H, W)
        augment_prob: Probability of applying augmentation pipeline
        
    Returns:
        Augmented (vol, mask) tuple
    """
    if random.random() > augment_prob:
        return vol, mask
    
    vol = vol.copy()
    mask = mask.copy()
    
    # 1. Random flipping along each axis
    if random.random() < FLIP_PROB:
        axis = random.choice([0, 1, 2])  # D, H, or W
        vol = np.flip(vol, axis=axis).copy()
        mask = np.flip(mask, axis=axis).copy()
    
    # 2. Random 90-degree rotations in H-W plane (preserves slice structure)
    if random.random() < ROTATE_PROB:
        k = random.choice([1, 2, 3])  # 90, 180, or 270 degrees
        vol = np.rot90(vol, k=k, axes=(1, 2)).copy()
        mask = np.rot90(mask, k=k, axes=(1, 2)).copy()
    
    # 3. Random intensity scaling (only for volume, not mask)
    if random.random() < INTENSITY_PROB:
        # Scale intensity by 0.8-1.2x and shift by -0.1 to 0.1
        scale = random.uniform(0.8, 1.2)
        shift = random.uniform(-0.1, 0.1)
        vol = np.clip(vol * scale + shift, 0, 1)
    
    # 4. Random Gaussian noise (only for volume)
    if random.random() < NOISE_PROB:
        noise_std = random.uniform(0.01, 0.05)
        noise = np.random.normal(0, noise_std, vol.shape).astype(vol.dtype)
        vol = np.clip(vol + noise, 0, 1)
    
    return vol, mask


def augment_slice_online(slice_pixels: np.ndarray, slice_mask: np.ndarray) -> tuple:
    """
    Lighter per-slice augmentation for online use during episode.
    Only applies 2D transforms to current slice.
    
    Args:
        slice_pixels: Current slice (H, W)
        slice_mask: Current mask slice (H, W)
        
    Returns:
        Augmented (slice_pixels, slice_mask) tuple
    """
    if random.random() > 0.3:  # 30% chance of per-slice augmentation
        return slice_pixels, slice_mask
    
    # Random 2D flip
    if random.random() < 0.5:
        axis = random.choice([0, 1])
        slice_pixels = np.flip(slice_pixels, axis=axis).copy()
        slice_mask = np.flip(slice_mask, axis=axis).copy()
    
    # Random 90-degree rotation
    if random.random() < 0.5:
        k = random.choice([1, 2, 3])
        slice_pixels = np.rot90(slice_pixels, k=k).copy()
        slice_mask = np.rot90(slice_mask, k=k).copy()
    
    return slice_pixels, slice_mask

# ============================================================================
# Subvolume Loading with Grid Positions
# ============================================================================

def parse_grid_index(filename: str):
    """
    Extract grid indices (d, h, w) from filename.
    Expected format: {prefix}_d{DD}_h{HH}_w{WW}.npy
    """
    import re
    pattern = r'd(\d+)_h(\d+)_w(\d+)'
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None


def load_subvolume_pairs_with_positions(data_dir: str, min_fg_ratio: float = 0.0):
    """
    Load subvolumes along with their grid positions for reconstruction.
    
    Expects directory structure:
        data_dir/
            volumes/
                {prefix}_d{DD}_h{HH}_w{WW}.npy
            masks/
                {prefix}_d{DD}_h{HH}_w{WW}.npy
    
    Args:
        data_dir: Path to directory containing volumes/ and masks/ subdirectories
        min_fg_ratio: Minimum foreground ratio to include subvolume (0.0 = include all)
    
    Returns:
        volumes: List of volume arrays
        masks: List of mask arrays
        positions: List of (d_idx, h_idx, w_idx) tuples
    """
    volumes = []
    masks = []
    positions = []
    
    volumes_dir = os.path.join(data_dir, "volumes")
    masks_dir = os.path.join(data_dir, "masks")
    
    if not os.path.exists(volumes_dir) or not os.path.exists(masks_dir):
        print(f"Error: Expected volumes/ and masks/ subdirectories in {data_dir}")
        return volumes, masks, positions
    
    vol_files = sorted([f for f in os.listdir(volumes_dir) if f.endswith(".npy")])
    
    n_skipped_no_mask = 0
    n_skipped_empty = 0
    
    for vol_file in vol_files:
        vol_path = os.path.join(volumes_dir, vol_file)
        mask_path = os.path.join(masks_dir, vol_file)  # Same filename in masks/
        
        if not os.path.exists(mask_path):
            n_skipped_no_mask += 1
            continue
        
        # Parse grid position from filename
        pos = parse_grid_index(vol_file)
        if pos is None:
            print(f"Warning: Could not parse grid position from {vol_file}")
            continue
        
        # Load mask first to check foreground ratio
        mask = np.load(mask_path)
        fg_ratio = mask.mean()
        
        # Skip subvolumes with no/little foreground
        if fg_ratio < min_fg_ratio:
            n_skipped_empty += 1
            continue
        
        vol = np.load(vol_path)
        
        volumes.append(vol)
        masks.append(mask)
        positions.append(pos)
        
        print(f"Loaded: {vol_file} at position {pos} (fg={fg_ratio*100:.4f}%)")
    
    print(f"\nSummary: Loaded {len(volumes)} subvolumes")
    if n_skipped_no_mask > 0:
        print(f"  Skipped {n_skipped_no_mask} (no matching mask)")
    if n_skipped_empty > 0:
        print(f"  Skipped {n_skipped_empty} (fg_ratio < {min_fg_ratio*100:.2f}%)")
    
    return volumes, masks, positions


def reconstruct_full_volume_from_subvolumes(
    policy_net: nn.Module,
    volumes: list,
    positions: list,
    subvol_shape: tuple = SUBVOL_SHAPE,
    full_shape: tuple = FULL_VOLUME_SHAPE,
    device: str = None,
    continuity_coef: float = 0.2,
    continuity_decay_factor: float = 0.7,
    dice_coef: float = 1.0,
    future_len: int = 3
) -> np.ndarray:
    """
    Reconstruct full volume prediction by running policy on each subvolume
    and placing predictions at their original grid positions.
    
    Args:
        policy_net: Trained policy network
        volumes: List of subvolume arrays
        positions: List of (d_idx, h_idx, w_idx) grid positions
        subvol_shape: Shape of each subvolume
        full_shape: Shape of full reconstructed volume
        device: Torch device
        
    Returns:
        full_pred: Full volume prediction (D, H, W) binary mask
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sub_d, sub_h, sub_w = subvol_shape
    
    # Initialize full prediction volume
    full_pred = np.zeros(full_shape, dtype=np.uint8)
    
    policy_net.eval()
    with torch.no_grad():
        for vol, (d_idx, h_idx, w_idx) in zip(volumes, positions):
            # Create dummy mask (not used for prediction, only for env initialization)
            dummy_mask = np.zeros_like(vol)
            
            # Reconstruct this subvolume
            env = PathReconstructionEnv(
                vol, dummy_mask,
                continuity_coef=continuity_coef,
                continuity_decay_factor=continuity_decay_factor,
                dice_coef=dice_coef,
                history_len=5,
                future_len=future_len,
            )
            
            obs, _ = env.reset()
            done = False
            subvol_pred = np.zeros(vol.shape, dtype=np.uint8)
            
            while not done:
                slice_pixels, prev_preds, future_slices = obs_to_tensor(obs, device)
                q = policy_net(slice_pixels, prev_preds, future_slices)
                a = q.argmax(dim=-1).cpu().numpy()[0]
                
                next_obs, _, terminated, truncated, info = env.step(a.flatten())
                
                slice_idx = info.get("slice_index", None)
                if slice_idx is not None:
                    subvol_pred[slice_idx, :, :] = a.astype(np.uint8)
                
                obs = next_obs
                done = terminated or truncated
            
            # Place subvolume prediction at its grid position
            d_start = d_idx * sub_d
            h_start = h_idx * sub_h
            w_start = w_idx * sub_w
            
            full_pred[
                d_start:d_start + sub_d,
                h_start:h_start + sub_h,
                w_start:w_start + sub_w
            ] = subvol_pred
    
    policy_net.train()
    return full_pred


def compute_global_dice(pred: np.ndarray, mask: np.ndarray) -> float:
    """Compute DICE score between full predicted and ground truth masks."""
    pred_flat = pred.flatten().astype(bool)
    mask_flat = mask.flatten().astype(bool)
    
    intersection = np.logical_and(pred_flat, mask_flat).sum()
    total = pred_flat.sum() + mask_flat.sum()
    
    if total == 0:
        return 1.0  # Both empty = perfect match
    
    return 2 * intersection / total


def compute_global_metrics(pred: np.ndarray, mask: np.ndarray) -> dict:
    """Compute comprehensive metrics for full volume."""
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
# Training Utilities (from main.py)
# ============================================================================

def obs_to_tensor(obs, device):
    """Convert observation dict to tensors for network input."""
    slice_pixels = torch.from_numpy(obs["slice_pixels"]).float().unsqueeze(0).to(device)
    prev_preds = torch.from_numpy(obs["prev_preds"]).float().unsqueeze(0).to(device)
    future_slices = torch.from_numpy(obs["future_slices"]).float().unsqueeze(0).to(device)
    return slice_pixels, prev_preds, future_slices


def epsilon_greedy_action(q_values, epsilon, n_pixels):
    """Select actions using epsilon-greedy policy."""
    if random.random() < epsilon:
        return np.random.randint(0, 2, size=n_pixels)
    else:
        return q_values.argmax(dim=-1).cpu().numpy()[0].flatten()


def compute_metrics(pred, mask):
    """Compute metrics for a single volume."""
    pred_flat = pred.flatten()
    mask_flat = mask.flatten()
    
    tp = ((pred_flat == 1) & (mask_flat == 1)).sum()
    fp = ((pred_flat == 1) & (mask_flat == 0)).sum()
    fn = ((pred_flat == 0) & (mask_flat == 1)).sum()
    tn = ((pred_flat == 0) & (mask_flat == 0)).sum()
    
    eps = 1e-8
    iou = tp / (tp + fp + fn + eps)
    f1 = 2 * tp / (2 * tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    coverage = tp / (tp + fn + eps)  # recall
    
    return {"iou": iou, "f1": f1, "accuracy": accuracy, "coverage": coverage}


# ============================================================================
# Main Training Loop with Global DICE
# ============================================================================

def train_with_global_dice(
    train_volumes: list,
    train_masks: list,
    train_positions: list,
    full_train_mask: np.ndarray,
    val_volumes: list = None,
    val_masks: list = None,
    val_positions: list = None,
    full_val_mask: np.ndarray = None,
    # Hyperparameters
    num_epochs: int = 100,
    global_eval_interval: int = 5,
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
    dice_coef: float = 1.0,
    manhattan_coef: float = 0.0,
    future_len: int = 3,
    device: str = None,
    save_dir: str = "models",
):
    """
    Train DQN with global DICE evaluation every N epochs.
    
    Key addition: Every `global_eval_interval` epochs, reconstruct the FULL volume
    from all subvolume predictions and compute global DICE against full mask.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Get dimensions from first subvolume
    H, W = train_volumes[0].shape[1], train_volumes[0].shape[2]
    history_len = 5
    
    # Use smaller network for small inputs (32x32, 64x64)
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
    
    # Print network parameter count
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
    global_dice_history = []
    global_metrics_history = []
    
    # Validation tracking
    val_global_dice_history = []
    val_global_metrics_history = []
    val_losses = []  # Validation loss computed at each global_eval_interval
    best_val_dice = 0.0
    best_val_pred = None
    best_epoch = 0
    
    # Per-epoch metrics
    train_metrics_history = {
        "iou": [], "f1": [], "accuracy": [], "coverage": [], "dice": []
    }
    
    print(f"\n{'='*60}")
    print("Starting training with global DICE evaluation")
    print(f"  - Training subvolumes: {len(train_volumes)}")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Global eval every: {global_eval_interval} epochs")
    print(f"  - Full volume shape: {FULL_VOLUME_SHAPE}")
    print(f"  - DICE coefficient: {dice_coef}")
    print(f"  - Data augmentation: {'ENABLED' if AUGMENT_PROB > 0 else 'DISABLED'} (prob={AUGMENT_PROB})")
    print("  - Epsilon decay: exponential (smoother)")
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
        epoch_metrics = []
        
        # Shuffle training data
        indices = list(range(len(train_volumes)))
        random.shuffle(indices)
        
        # Train on all subvolumes
        for idx in tqdm(indices, desc=f"Epoch {epoch+1}/{num_epochs}"):
            vol = train_volumes[idx]
            mask = train_masks[idx]
            
            # Apply data augmentation (training only)
            vol_aug, mask_aug = augment_volume_and_mask(vol, mask)
            
            env = PathReconstructionEnv(
                vol_aug, mask_aug,
                continuity_coef=continuity_coef,
                continuity_decay_factor=continuity_decay_factor,
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
                slice_pixels, prev_preds, future_slices = obs_to_tensor(obs, device)
                
                with torch.no_grad():
                    q = policy_net(slice_pixels, prev_preds, future_slices)
                
                action = epsilon_greedy_action(q, epsilon, n_pixels)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                pixel_rewards = info["pixel_rewards"]
                episode_reward += reward
                episode_base_reward += info["base_rewards"].sum()
                episode_continuity_reward += info["continuity_rewards"].sum()
                episode_gradient_reward += info["gradient_rewards"]
                episode_manhattan_reward += info.get("manhattan_reward", 0.0)  # Per-slice Manhattan
                
                # Store transition (now includes future_slices)
                replay_buffer.append((
                    obs["slice_pixels"].copy(),
                    obs["prev_preds"].copy(),
                    obs["future_slices"].copy(),
                    action.copy(),
                    pixel_rewards.copy(),
                    next_obs["slice_pixels"].copy(),
                    next_obs["prev_preds"].copy(),
                    next_obs["future_slices"].copy(),
                    done
                ))
                
                obs = next_obs
                global_step += 1
                
                # Update epsilon (exponential decay for smoother exploration)
                # epsilon = epsilon_end + (epsilon_start - epsilon_end) * exp(-global_step / epsilon_decay_steps)
                decay_rate = 1.0 / epsilon_decay_steps  # reaches ~5% of initial after epsilon_decay_steps
                epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-decay_rate * global_step)
                epsilon = max(epsilon, epsilon_end)  # ensure it doesn't go below epsilon_end
                
                # Training step
                if len(replay_buffer) >= batch_size:
                    batch = random.sample(replay_buffer, batch_size)
                    
                    sp_batch = torch.tensor(np.array([t[0] for t in batch]), dtype=torch.float32, device=device)
                    pp_batch = torch.tensor(np.array([t[1] for t in batch]), dtype=torch.float32, device=device)
                    fs_batch = torch.tensor(np.array([t[2] for t in batch]), dtype=torch.float32, device=device)
                    a_batch = torch.tensor(np.array([t[3] for t in batch]), dtype=torch.long, device=device)
                    r_batch = torch.tensor(np.array([t[4] for t in batch]), dtype=torch.float32, device=device)
                    next_sp_batch = torch.tensor(np.array([t[5] for t in batch]), dtype=torch.float32, device=device)
                    next_pp_batch = torch.tensor(np.array([t[6] for t in batch]), dtype=torch.float32, device=device)
                    next_fs_batch = torch.tensor(np.array([t[7] for t in batch]), dtype=torch.float32, device=device)
                    done_batch = torch.tensor(np.array([t[8] for t in batch]), dtype=torch.float32, device=device)
                    
                    # Current Q values: (B, H, W, 2)
                    q_values = policy_net(sp_batch, pp_batch, fs_batch)
                    
                    # Reshape action batch from (B, H*W) to (B, H, W) for gather
                    a_batch_2d = a_batch.view(batch_size, H, W)
                    q_sa = q_values.gather(dim=-1, index=a_batch_2d.unsqueeze(-1)).squeeze(-1)  # (B, H, W)
                    
                    # Target Q values
                    with torch.no_grad():
                        q_next = target_net(next_sp_batch, next_pp_batch, next_fs_batch)  # (B, H, W, 2)
                        q_next_max = q_next.max(dim=-1).values  # (B, H, W)
                        not_done = (1.0 - done_batch).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
                        # r_batch is (B, H*W), reshape to (B, H, W)
                        r_batch_2d = r_batch.view(batch_size, H, W)
                        target = r_batch_2d + gamma * not_done * q_next_max  # (B, H, W)
                    
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
            
            # Get episode DICE from info (computed at episode end)
            if info.get("episode_dice") is not None:
                epoch_metrics.append({"dice": info["episode_dice"]})
                # Capture DICE reward at episode end
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
        
        # Average metrics
        if epoch_metrics:
            avg_dice = np.mean([m["dice"] for m in epoch_metrics])
            train_metrics_history["dice"].append(avg_dice)
        
        # ====================================================================
        # GLOBAL DICE EVALUATION every N epochs
        # ====================================================================
        if (epoch + 1) % global_eval_interval == 0:
            print(f"\n[Epoch {epoch+1}] Computing global DICE on full volume...")
            
            # Reconstruct full volume from all training subvolumes
            full_pred = reconstruct_full_volume_from_subvolumes(
                policy_net,
                train_volumes,
                train_positions,
                device=device,
                continuity_coef=continuity_coef,
                continuity_decay_factor=continuity_decay_factor,
                dice_coef=dice_coef,
                future_len=future_len
            )
            
            # Compute global metrics
            global_metrics = compute_global_metrics(full_pred, full_train_mask)
            global_dice_history.append(global_metrics["dice"])
            global_metrics_history.append(global_metrics)
            
            print(f"  GLOBAL DICE: {global_metrics['dice']:.4f}")
            print(f"  GLOBAL IoU:  {global_metrics['iou']:.4f}")
            print(f"  Precision:   {global_metrics['precision']:.4f}")
            print(f"  Recall:      {global_metrics['recall']:.4f}")
            print(f"  TP: {global_metrics['tp']:,} | FP: {global_metrics['fp']:,} | FN: {global_metrics['fn']:,}")
            
            # Save model checkpoint
            checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch+1}_dice_{global_metrics['dice']:.4f}.pth")
            torch.save(policy_net.state_dict(), checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
            
            # Validation on full val volume
            if val_volumes and full_val_mask is not None:
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
                            dice_coef=dice_coef,
                            history_len=history_len,
                            future_len=future_len,
                        )
                        val_obs, _ = val_env.reset()
                        val_done = False
                        
                        while not val_done:
                            val_sp, val_pp, val_fs = obs_to_tensor(val_obs, device)
                            val_q = policy_net(val_sp, val_pp, val_fs)  # (1, H, W, 2)
                            val_action = val_q.argmax(dim=-1).cpu().numpy()[0].flatten()
                            
                            val_next_obs, _, val_term, val_trunc, val_info = val_env.step(val_action)
                            val_done = val_term or val_trunc
                            
                            # Compute TD loss for this step (no gradient)
                            val_pixel_rewards = val_info["pixel_rewards"]
                            val_r = torch.tensor(val_pixel_rewards, dtype=torch.float32, device=device).view(1, H, W)
                            val_a = torch.tensor(val_action, dtype=torch.long, device=device).view(1, H, W)
                            
                            val_q_sa = val_q.gather(dim=-1, index=val_a.unsqueeze(-1)).squeeze(-1)  # (1, H, W)
                            
                            if not val_done:
                                val_next_sp, val_next_pp, val_next_fs = obs_to_tensor(val_next_obs, device)
                                val_q_next = policy_net(val_next_sp, val_next_pp, val_next_fs)
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
                
                val_pred = reconstruct_full_volume_from_subvolumes(
                    policy_net,
                    val_volumes,
                    val_positions,
                    device=device,
                    continuity_coef=continuity_coef,
                    continuity_decay_factor=continuity_decay_factor,
                    dice_coef=dice_coef,
                    future_len=future_len
                )
                val_metrics = compute_global_metrics(val_pred, full_val_mask)
                val_global_dice_history.append(val_metrics["dice"])
                val_global_metrics_history.append(val_metrics)
                print(f"  [VAL] DICE: {val_metrics['dice']:.4f} | IoU: {val_metrics['iou']:.4f}")
                print(f"  [VAL] Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f}")
                
                # Track best validation model
                if val_metrics["dice"] > best_val_dice:
                    best_val_dice = val_metrics["dice"]
                    best_val_pred = val_pred.copy()
                    best_epoch = epoch + 1
                    # Save best model
                    best_model_path = os.path.join(save_dir, "best_model.pth")
                    torch.save(policy_net.state_dict(), best_model_path)
                    print(f"  [NEW BEST] Saved best model (val DICE={best_val_dice:.4f})")
        
        # Print epoch summary
        dt = time.time() - t0
        avg_return = np.mean(epoch_returns)
        avg_base = np.mean(epoch_base_returns)
        avg_cont = np.mean(epoch_continuity_returns)
        avg_grad = np.mean(epoch_gradient_returns)
        avg_dice_rew = np.mean(epoch_dice_returns)
        print(f"Epoch {epoch+1}/{num_epochs} | Time: {dt:.1f}s | Loss: {losses[-1]:.4f} | ε: {epsilon:.3f}")
        print(f"  Returns - Total: {avg_return:.2f} | Base: {avg_base:.2f} | Cont: {avg_cont:.2f} | Grad: {avg_grad:.2f}| DICE: {avg_dice_rew:.2f} | Manhattan: {np.mean(epoch_manhattan_returns):.2f}")
        
        # Save latest model
        torch.save(policy_net.state_dict(), os.path.join(save_dir, "model_latest.pth"))
    
    # Final global evaluation
    print("\n" + "="*60)
    print("FINAL GLOBAL EVALUATION")
    print("="*60)
    
    final_pred = reconstruct_full_volume_from_subvolumes(
        policy_net, train_volumes, train_positions, device=device, dice_coef=dice_coef, future_len=future_len
    )
    final_metrics = compute_global_metrics(final_pred, full_train_mask)
    
    print(f"Final DICE: {final_metrics['dice']:.4f}")
    print(f"Final IoU:  {final_metrics['iou']:.4f}")
    
    return {
        "policy_net": policy_net,
        "target_net": target_net,
        "global_dice_history": global_dice_history,
        "global_metrics_history": global_metrics_history,
        "val_global_dice_history": val_global_dice_history,
        "val_global_metrics_history": val_global_metrics_history,
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
        "train_metrics": train_metrics_history,
        "n_train_vols": len(train_volumes),
        "global_eval_interval": global_eval_interval,  # Pass this for correct plotting
    }


# ============================================================================
# Visualization
# ============================================================================

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
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"reconstruction_slice_{slice_idx}.png"))
    plt.show()


def plot_training_results(results: dict, save_dir: str = "results"):
    """Plot comprehensive training results with train/val comparison."""
    os.makedirs(save_dir, exist_ok=True)
    
    n_train_vols = results.get("n_train_vols", 1)
    # Get actual global_eval_interval from results (default to 5 for backward compatibility)
    global_eval_interval = results.get("global_eval_interval", 5)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # ========================================
    # Plot 1: Episode Returns
    # ========================================
    axes[0, 0].plot(results["train_returns"], alpha=0.3, color='blue', label='Train (per volume)')
    # Moving average
    window = n_train_vols
    if len(results["train_returns"]) >= window:
        train_returns_ma = [np.mean(results["train_returns"][max(0, i-window+1):i+1]) 
                           for i in range(len(results["train_returns"]))]
        axes[0, 0].plot(train_returns_ma, color='blue', linewidth=2, label='Train (epoch avg)')
    axes[0, 0].set_title("Episode Returns")
    axes[0, 0].set_ylabel("Return")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # ========================================
    # Plot 2: Reward Components (Moving Average)
    # ========================================
    if results["base_returns"] and len(results["base_returns"]) >= window:
        base_ma = np.convolve(results["base_returns"], 
                              np.ones(window)/window, 
                              mode='valid')
        axes[0, 1].plot(base_ma, label="Base Reward", color='green')
    else:
        axes[0, 1].plot(results["base_returns"], label="Base Reward", color='green', alpha=0.5)

    if results["continuity_returns"] and len(results["continuity_returns"]) >= window:
        cont_ma = np.convolve(results["continuity_returns"], 
                              np.ones(window)/window, 
                              mode='valid')
        axes[0, 1].plot(cont_ma, label="Continuity Reward", color='orange')
    else:
        axes[0, 1].plot(results["continuity_returns"], label="Continuity Reward", color='orange', alpha=0.5)    

    # Gradient reward plotting
    if results.get("gradient_returns"):
        if len(results["gradient_returns"]) >= window:
            grad_ma = np.convolve(results["gradient_returns"],
                                  np.ones(window)/window,
                                  mode='valid')
            axes[0, 1].plot(grad_ma, label="Gradient Reward", color='purple')
        else:
            axes[0, 1].plot(results["gradient_returns"], label="Gradient Reward", color='purple', alpha=0.5)

    # Manhattan distance reward plotting
    if results.get("manhattan_returns"):
        if len(results["manhattan_returns"]) >= window:
            manh_ma = np.convolve(results["manhattan_returns"],
                                  np.ones(window)/window,
                                  mode='valid')
            axes[0, 1].plot(manh_ma, label="Manhattan Reward", color='red')
        else:
            axes[0, 1].plot(results["manhattan_returns"], label="Manhattan Reward", color='red', alpha=0.5)

    axes[0, 1].set_title("Reward Components (Moving Average)")
    axes[0, 1].set_ylabel("Reward")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # ========================================
    # Plot 3: DICE Score (Train vs Val)
    # ========================================
    if results["global_dice_history"]:
        epochs = list(range(global_eval_interval, 
                           len(results["global_dice_history"]) * global_eval_interval + 1, 
                           global_eval_interval))
        axes[0, 2].plot(epochs, results["global_dice_history"], 'b-o', linewidth=2, 
                        markersize=6, label='Train DICE')
        
        # Plot validation DICE if available
        if results.get("val_global_dice_history"):
            val_epochs = epochs[:len(results["val_global_dice_history"])]
            axes[0, 2].plot(val_epochs, results["val_global_dice_history"], 'r-s', 
                           linewidth=2, markersize=2, label='Val DICE')
            # # Mark best validation
            # if results.get("best_epoch"):
            #     best_idx = results["val_global_dice_history"].index(results["best_val_dice"]) if results["best_val_dice"] in results["val_global_dice_history"] else -1
            #     if best_idx >= 0:
            #         axes[0, 2].scatter([val_epochs[best_idx]], [results["best_val_dice"]], 
            #                           color='red', s=150, zorder=5, marker='*')
            #         axes[0, 2].annotate(f'Best: {results["best_val_dice"]:.4f}', 
            #                            xy=(val_epochs[best_idx], results["best_val_dice"]),
            #                            xytext=(val_epochs[best_idx] + 3, results["best_val_dice"] - 0.05),
            #                            fontsize=10, color='red')
    axes[0, 2].set_title("DICE Score (Train vs Val)")
    axes[0, 2].set_ylabel("DICE")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # ========================================
    # Plot 4: Epsilon (Exponential Decay)
    # ========================================
    axes[1, 0].plot(results["epsilons"], color='orange', linewidth=2)
    axes[1, 0].set_title("Exploration (Epsilon - Exponential Decay)")
    axes[1, 0].set_ylabel("ε")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].grid(True)
    
    # ========================================
    # Plot 5: Training & Validation Loss
    # ========================================
    axes[1, 1].plot(results["losses"], color='blue', marker='o', markersize=4, label='Train Loss')
    
    # Plot validation loss if available (computed at global_eval_interval epochs)
    if results.get("val_losses") and len(results["val_losses"]) > 0:
        val_loss_epochs = list(range(global_eval_interval, 
                                     len(results["val_losses"]) * global_eval_interval + 1, 
                                     global_eval_interval))
        # Adjust epochs to be 0-indexed for plotting alignment
        val_loss_epochs_adj = [e - 1 for e in val_loss_epochs]  # Convert to 0-indexed
        axes[1, 1].plot(val_loss_epochs_adj, results["val_losses"], color='red', 
                       marker='s', markersize=2, linewidth=2, label='Val Loss')
    
    axes[1, 1].set_title("Training & Validation Loss")
    axes[1, 1].set_ylabel("MSE Loss")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # ========================================
    # Plot 6: IoU Score (Train vs Val)
    # ========================================
    if results["global_metrics_history"]:
        epochs = list(range(global_eval_interval, 
                           len(results["global_metrics_history"]) * global_eval_interval + 1, 
                           global_eval_interval))
        train_iou = [m["iou"] for m in results["global_metrics_history"]]
        axes[1, 2].plot(epochs, train_iou, 'b-o', linewidth=2, markersize=6, label='Train IoU')
        
        # Plot validation IoU if available
        if results.get("val_global_metrics_history"):
            val_iou = [m["iou"] for m in results["val_global_metrics_history"]]
            val_epochs = epochs[:len(val_iou)]
            axes[1, 2].plot(val_epochs, val_iou, 'r-s', linewidth=2, markersize=2, label='Val IoU')
    axes[1, 2].set_title("IoU Score (Train vs Val)")
    axes[1, 2].set_ylabel("IoU")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    # plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_results.png"), dpi=300)
    plt.show()
    
    # Additional plot: Detailed Train vs Val metrics comparison
    if results["global_metrics_history"] and results.get("val_global_metrics_history"):
        fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = list(range(global_eval_interval, 
                           len(results["global_metrics_history"]) * global_eval_interval + 1, 
                           global_eval_interval))
        val_epochs = epochs[:len(results["val_global_metrics_history"])]
        
        # DICE
        axes2[0, 0].plot(epochs, results["global_dice_history"], 'b-o', label='Train', linewidth=2)
        axes2[0, 0].plot(val_epochs, results["val_global_dice_history"], 'r-s', label='Val', linewidth=2)
        axes2[0, 0].set_title("DICE Score")
        axes2[0, 0].set_ylabel("DICE")
        axes2[0, 0].set_xlabel("Epoch")
        axes2[0, 0].legend()
        axes2[0, 0].grid(True)
        
        # IoU
        train_iou = [m["iou"] for m in results["global_metrics_history"]]
        val_iou = [m["iou"] for m in results["val_global_metrics_history"]]
        axes2[0, 1].plot(epochs, train_iou, 'b-o', label='Train', linewidth=2)
        axes2[0, 1].plot(val_epochs, val_iou, 'r-s', label='Val', linewidth=2)
        axes2[0, 1].set_title("IoU Score")
        axes2[0, 1].set_ylabel("IoU")
        axes2[0, 1].set_xlabel("Epoch")
        axes2[0, 1].legend()
        axes2[0, 1].grid(True)
        
        # Precision
        train_precision = [m["precision"] for m in results["global_metrics_history"]]
        val_precision = [m["precision"] for m in results["val_global_metrics_history"]]
        axes2[1, 0].plot(epochs, train_precision, 'b-o', label='Train', linewidth=2)
        axes2[1, 0].plot(val_epochs, val_precision, 'r-s', label='Val', linewidth=2)
        axes2[1, 0].set_title("Precision")
        axes2[1, 0].set_ylabel("Precision")
        axes2[1, 0].set_xlabel("Epoch")
        axes2[1, 0].legend()
        axes2[1, 0].grid(True)
        
        # Recall
        train_recall = [m["recall"] for m in results["global_metrics_history"]]
        val_recall = [m["recall"] for m in results["val_global_metrics_history"]]
        axes2[1, 1].plot(epochs, train_recall, 'b-o', label='Train', linewidth=2)
        axes2[1, 1].plot(val_epochs, val_recall, 'r-s', label='Val', linewidth=2)
        axes2[1, 1].set_title("Recall")
        axes2[1, 1].set_ylabel("Recall")
        axes2[1, 1].set_xlabel("Epoch")
        axes2[1, 1].legend()
        axes2[1, 1].grid(True)
        
        plt.suptitle("Train vs Validation Metrics", fontsize=14, fontweight='bold')
        plt.tight_layout()
        # plt.savefig(os.path.join(save_dir, "train_val_metrics.png"), dpi=300)
        # plt.show()
        plt.savefig(os.path.join(save_dir, "global_metrics.png"), dpi=300)
        plt.show()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Paths - new directory structure with volumes/ and masks/ subdirectories
    # Note: Paths are relative to dqn_slice_based/ directory
    data_dir = "../../data/rapids-p/subvolumes_new"
    
    # Minimum foreground ratio to include subvolume (filter out empty subvolumes)
    MIN_FG_RATIO = 0.0001  # 0.01% minimum foreground
    
    # Load subvolumes WITH positions
    print("Loading training subvolumes with grid positions...")
    train_vols, train_masks, train_positions = load_subvolume_pairs_with_positions(
        os.path.join(data_dir, "train"),
        min_fg_ratio=MIN_FG_RATIO
    )
    print(f"Loaded {len(train_vols)} training subvolumes")
    
    print("\nLoading validation subvolumes...")
    val_vols, val_masks, val_positions = load_subvolume_pairs_with_positions(
        os.path.join(data_dir, "val"),
        min_fg_ratio=MIN_FG_RATIO
    )
    print(f"Loaded {len(val_vols)} validation subvolumes")
    
    # Load full ground truth mask (needed for global DICE)
    # The mask is in NIfTI format
    import nibabel as nib
    
    full_mask_path = "../../data/rapids-p/week2-joint-root-class.nii.gz"
    print(f"\nLoading full ground truth mask from {full_mask_path}...")
    
    if os.path.exists(full_mask_path):
        mask_nii = nib.load(full_mask_path)
        full_train_mask = (mask_nii.get_fdata() > 0).astype(np.uint8)
        print(f"Full mask shape (original): {full_train_mask.shape}")
        
        # Crop to grid-aligned shape
        D, H, W = full_train_mask.shape
        crop_d = (D // SUBVOL_SHAPE[0]) * SUBVOL_SHAPE[0]
        crop_h = (H // SUBVOL_SHAPE[1]) * SUBVOL_SHAPE[1]
        crop_w = (W // SUBVOL_SHAPE[2]) * SUBVOL_SHAPE[2]
        full_train_mask = full_train_mask[:crop_d, :crop_h, :crop_w]
        
        print(f"Full mask shape (cropped): {full_train_mask.shape}")
        print(f"Full mask foreground: {full_train_mask.mean()*100:.4f}%")
    else:
        print(f"WARNING: Full mask not found at {full_mask_path}")
        print("Cannot compute global DICE without full ground truth mask!")
        exit(1)
    
    full_val_mask = nib.load("../../data/rapids-p/week3-joint-root-class.nii.gz")
    full_val_mask = (full_val_mask.get_fdata() > 0).astype(np.uint8)
    full_val_mask = full_val_mask[:crop_d, :crop_h, :crop_w]

    # Ensure output directory exists
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Train with global DICE evaluation
    results = train_with_global_dice(
        train_volumes=train_vols,
        train_masks=train_masks,
        train_positions=train_positions,
        full_train_mask=full_train_mask,
        val_volumes=val_vols,
        val_masks=val_masks,
        val_positions=val_positions,
        full_val_mask=full_val_mask,
        num_epochs=50,
        global_eval_interval=1,
        lr=1e-3,
        gamma=0.99,
        batch_size=64,
        continuity_coef=0.0,
        continuity_decay_factor=0.5,
        dice_coef=1.,  
        manhattan_coef = 0.1
    )
    
    # Plot comprehensive training results
    plot_training_results(results, save_dir="results/rapids-p/subvolumes/32x32x32")
    
    # Save best validation prediction as NIfTI file
    if results.get("best_val_pred") is not None:
        print(f"\nSaving best validation prediction (epoch {results['best_epoch']}, DICE={results['best_val_dice']:.4f})...")
        
        # Create NIfTI image from prediction
        # Use same affine as the original mask for proper spatial alignment
        val_mask_nii = nib.load("../../data/rapids-p/week3-joint-root-class.nii.gz")
        affine = val_mask_nii.affine
        header = val_mask_nii.header.copy()
        
        # Create full-size prediction array (padded to original size if needed)
        best_pred = results["best_val_pred"]
        original_shape = val_mask_nii.shape
        
        # Pad prediction back to original size if it was cropped
        if best_pred.shape != original_shape:
            full_pred_padded = np.zeros(original_shape, dtype=np.uint8)
            d, h, w = best_pred.shape
            full_pred_padded[:d, :h, :w] = best_pred
            best_pred = full_pred_padded
        
        pred_nii = nib.Nifti1Image(best_pred.astype(np.uint8), affine, header)
        pred_save_path = "results/best_val_prediction.nii.gz"
        nib.save(pred_nii, pred_save_path)
        print(f"Saved best validation prediction to: {pred_save_path}")
        print(f"  Shape: {best_pred.shape}")
        print(f"  Foreground voxels: {best_pred.sum():,} ({best_pred.mean()*100:.4f}%)")
    else:
        print("\nNo best validation prediction available to save.")
    
    # Visualize sample reconstruction
    if train_vols:
        vol_test = train_vols[0]
        mask_test = train_masks[0]
        
        # Reconstruct single subvolume for visualization
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        env = PathReconstructionEnv(
            vol_test, mask_test,
            continuity_coef=0.1,
            continuity_decay_factor=0.5,
            dice_coef=1.0,
            history_len=5
        )
        
        results["policy_net"].eval()
        obs, _ = env.reset()
        pred = np.zeros(vol_test.shape, dtype=np.uint8)
        done = False
        
        with torch.no_grad():
            while not done:
                slice_pixels, prev_preds = obs_to_tensor(obs, device)
                q = results["policy_net"](slice_pixels, prev_preds)
                a = q.argmax(dim=-1).cpu().numpy()[0]
                next_obs, _, terminated, truncated, info = env.step(a.flatten())
                slice_idx = info.get("slice_index", None)
                if slice_idx is not None:
                    pred[slice_idx] = a.astype(np.uint8)
                obs = next_obs
                done = terminated or truncated
        
        # visualize_result(vol_test, mask_test, pred, 
        #                 save_dir="results/rapids-p/subvolumes/32x32x32", 
        #                 slice_idx=vol_test.shape[0] // 2)
