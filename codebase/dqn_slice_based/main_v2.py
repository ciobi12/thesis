"""
Training script optimized for large real CT volumes with limited data.

Key improvements:
1. Data augmentation: random slice sampling, intensity augmentation, random crops
2. Memory efficiency: gradient accumulation, mixed precision training
3. Better exploration: cosine annealing epsilon, longer warm-up
4. Class-weighted rewards for imbalanced foreground/background
5. Multi-episode training per volume to increase diversity
"""
import cv2
import nibabel as nib
import numpy as np
import os
import random
import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.cuda.amp as amp
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

from dqn_slice_based.dqn_v2 import PerPixelDQNV2, UNetDQN, ReplayBuffer, PrioritizedReplayBuffer
from dqn_slice_based.env_v2 import PathReconstructionEnvV2, SliceSamplerEnv

from matplotlib import pyplot as plt
from tqdm import tqdm


def obs_to_tensor(obs, device):
    """Convert observation dict to tensors on device."""
    slice_pixels = torch.FloatTensor(obs["slice_pixels"]).unsqueeze(0).to(device)
    prev_preds = torch.FloatTensor(obs["prev_preds"]).unsqueeze(0).to(device)
    prev_slices = torch.FloatTensor(obs["prev_slices"]).unsqueeze(0).to(device)
    return slice_pixels, prev_preds, prev_slices


def batch_obs_to_tensor(obs_list, device):
    """Convert list of observation dicts to batched tensors."""
    slice_pixels = torch.stack([torch.FloatTensor(o["slice_pixels"]) for o in obs_list]).to(device)
    prev_preds = torch.stack([torch.FloatTensor(o["prev_preds"]) for o in obs_list]).to(device)
    prev_slices = torch.stack([torch.FloatTensor(o["prev_slices"]) for o in obs_list]).to(device)
    return slice_pixels, prev_preds, prev_slices


def epsilon_greedy_action(q_values, epsilon):
    """Epsilon-greedy action selection."""
    if np.random.rand() < epsilon:
        H, W = q_values.shape[0], q_values.shape[1]
        return torch.randint(0, 2, (H, W), device=q_values.device)
    else:
        return q_values.argmax(dim=-1)


def compute_metrics(pred, mask):
    """Compute segmentation metrics."""
    pred_binary = (pred > 0).astype(np.float32)
    mask_binary = (mask > 0).astype(np.float32)
    
    intersection = np.logical_and(pred_binary, mask_binary).sum()
    union = np.logical_or(pred_binary, mask_binary).sum()
    
    iou = intersection / (union + 1e-8)
    
    tp = intersection
    fp = (pred_binary * (1 - mask_binary)).sum()
    fn = ((1 - pred_binary) * mask_binary).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    correct = (pred_binary == mask_binary).sum()
    total = pred_binary.size
    accuracy = correct / total
    
    coverage = intersection / (mask_binary.sum() + 1e-8)
    
    return {
        "iou": iou,
        "f1": f1,
        "accuracy": accuracy,
        "coverage": coverage,
        "precision": precision,
        "recall": recall
    }


def dice_loss(pred, target):
    """Differentiable Dice loss."""
    smooth = 1e-6
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def validate(policy_net, val_volume, val_mask, device, config):
    """Run validation on the full volume."""
    policy_net.eval()
    
    env = PathReconstructionEnvV2(
        volume=val_volume,
        mask=val_mask,
        continuity_coef=config['continuity_coef'],
        continuity_decay_factor=config['continuity_decay_factor'],
        history_len=config['history_len'],
        use_class_weights=False,  # Don't weight during validation
        random_slice_order=False,
        start_from_bottom=True,
    )
    
    pred = np.zeros(val_volume.shape, dtype=np.uint8)
    obs, _ = env.reset()
    done = False
    episode_reward = 0.0
    
    with torch.no_grad():
        while not done:
            slice_pixels, prev_preds, prev_slices = obs_to_tensor(obs, device)
            q = policy_net(slice_pixels, prev_preds, prev_slices)
            a = q.argmax(dim=-1).cpu().numpy()[0]
            
            next_obs, reward, terminated, truncated, info = env.step(a.flatten())
            episode_reward += reward
            
            slice_idx = info.get("slice_index", None)
            if slice_idx is not None:
                pred[slice_idx, :, :] = a.astype(np.uint8)
            
            obs = next_obs
            done = terminated or truncated
    
    policy_net.train()
    
    metrics = compute_metrics(pred, val_mask)
    metrics["reward"] = episode_reward
    
    return metrics, pred


def augment_volume(volume, mask):
    """
    Data augmentation for 3D volumes.
    Returns augmented copies to increase training diversity.
    """
    augmented = []
    
    # Original
    augmented.append((volume.copy(), mask.copy()))
    
    # Flip along different axes
    augmented.append((np.flip(volume, axis=1).copy(), np.flip(mask, axis=1).copy()))
    augmented.append((np.flip(volume, axis=2).copy(), np.flip(mask, axis=2).copy()))
    
    # Intensity variations
    for scale in [0.9, 1.1]:
        vol_scaled = np.clip(volume * scale, 0, 1 if volume.max() <= 1 else 255)
        augmented.append((vol_scaled.copy(), mask.copy()))
    
    return augmented


def train_on_real_volumes(
    train_volume,
    train_mask,
    val_volume,
    val_mask,
    # Training parameters
    num_epochs=50,
    episodes_per_epoch=10,  # Multiple episodes per epoch from same volume
    slices_per_episode=64,  # Sample subset of slices per episode
    buffer_capacity=50000,
    batch_size=16,
    gradient_accumulation_steps=4,  # Effective batch = batch_size * accumulation
    gamma=0.95,
    lr=3e-4,
    weight_decay=1e-5,
    target_update_every=200,
    # Exploration
    start_epsilon=1.0,
    end_epsilon=0.05,
    warmup_epochs=5,  # Epochs before epsilon starts decaying
    # Reward parameters
    continuity_coef=0.1,
    continuity_decay_factor=0.7,
    foreground_weight=2.0, 
    boundary_coef=0.1,
    # Model
    model_type='perpixel',  # 'perpixel' or 'unet'
    history_len=5,
    base_channels=32,
    # Misc
    use_mixed_precision=True,
    use_prioritized_replay=True,
    save_dir='dqn_slice_based/models',
    seed=42,
    device=None,
):
    """
    Train DQN on limited real CT data.
    
    Key strategies for limited data:
    1. Multiple episodes per volume with random slice sampling
    2. Data augmentation (intensity, flips)
    3. Longer exploration phase
    4. Class-weighted rewards
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Training on device: {device}")
    print(f"Train volume shape: {train_volume.shape}")
    print(f"Val volume shape: {val_volume.shape}")
    
    D, H, W = train_volume.shape
    
    # Normalize volumes
    if train_volume.max() > 1:
        train_volume = train_volume.astype(np.float32) / 255.0
    if val_volume.max() > 1:
        val_volume = val_volume.astype(np.float32) / 255.0
    
    train_mask = (train_mask > 0).astype(np.float32)
    val_mask = (val_mask > 0).astype(np.float32)
    
    # Create model
    if model_type == 'unet':
        policy_net = UNetDQN(history_len=history_len, base_channels=base_channels).to(device)
        target_net = UNetDQN(history_len=history_len, base_channels=base_channels).to(device)
    else:
        policy_net = PerPixelDQNV2(history_len=history_len, base_channels=base_channels).to(device)
        target_net = PerPixelDQNV2(history_len=history_len, base_channels=base_channels).to(device)
    
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    # Count parameters
    num_params = sum(p.numel() for p in policy_net.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(policy_net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)
    
    # Mixed precision
    scaler = amp.GradScaler() if use_mixed_precision and device == "cuda" else None
    
    # Replay buffer
    if use_prioritized_replay:
        replay = PrioritizedReplayBuffer(buffer_capacity)
    else:
        replay = ReplayBuffer(buffer_capacity)
    
    # Config for validation
    config = {
        'continuity_coef': continuity_coef,
        'continuity_decay_factor': continuity_decay_factor,
        'history_len': history_len,
    }
    
    # Training tracking
    global_step = 0
    losses = []
    train_returns = []
    val_returns = []
    train_metrics_history = {"iou": [], "f1": [], "accuracy": [], "coverage": []}
    val_metrics_history = {"iou": [], "f1": [], "accuracy": [], "coverage": []}
    best_val_iou = 0.0
    
    # Data augmentation
    augmented_data = augment_volume(train_volume, train_mask)
    print(f"Augmented training data: {len(augmented_data)} volume variants")
    
    for epoch in range(num_epochs):
        # Epsilon schedule with warmup
        if epoch < warmup_epochs:
            epsilon = start_epsilon
        else:
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            epsilon = end_epsilon + (start_epsilon - end_epsilon) * (1 - progress) ** 2
        epsilon = max(end_epsilon, epsilon)
        
        epoch_return = 0.0
        epoch_losses = []
        epoch_metrics = []
        t0 = time.time()
        
        # Multiple episodes per epoch for diversity
        for ep_idx in tqdm(range(episodes_per_epoch)):
            # Randomly select augmented version
            vol, mask = random.choice(augmented_data)
            
            # Use slice sampler for random slice sequences
            env = SliceSamplerEnv(
                volume=vol,
                mask=mask,
                slices_per_episode=slices_per_episode,
                continuity_coef=continuity_coef,
                continuity_decay_factor=continuity_decay_factor,
                history_len=history_len,
                use_class_weights=True,
                foreground_weight=foreground_weight,
                boundary_coef=boundary_coef,
                random_direction=True,
            )
            
            obs, _ = env.reset()
            done = False
            episode_return = 0.0
            
            optimizer.zero_grad()
            accumulated_loss = 0.0
            accumulation_count = 0
            
            while not done:
                # Get action
                slice_pixels, prev_preds, prev_slices = obs_to_tensor(obs, device)
                
                with torch.no_grad():
                    q = policy_net(slice_pixels, prev_preds, prev_slices)
                    q = q.squeeze(0)  # (H, W, 2)
                
                a = epsilon_greedy_action(q, epsilon)
                
                # Environment step
                next_obs, reward, terminated, truncated, info = env.step(a.cpu().numpy().flatten())
                pixel_rewards = info["pixel_rewards"]
                done = terminated or truncated
                episode_return += reward
                
                # Store transition
                replay.push(
                    obs,
                    a.cpu().numpy(),
                    pixel_rewards.astype(np.float32),
                    next_obs,
                    done
                )
                
                obs = next_obs
                global_step += 1
                
                # Training step
                if len(replay) >= batch_size:
                    if use_prioritized_replay:
                        batch, indices, weights = replay.sample(batch_size)
                        weights = weights.to(device)
                    else:
                        batch = replay.sample(batch_size)
                        indices = None
                        weights = None
                    
                    # Batch conversion
                    slice_pixels_batch, prev_preds_batch, prev_slices_batch = batch_obs_to_tensor(
                        [t.obs for t in batch], device
                    )
                    act_batch = torch.tensor(
                        np.stack([t.action for t in batch]), dtype=torch.int64, device=device
                    )
                    rew_batch = torch.tensor(
                        np.stack([t.pixel_rewards for t in batch]), dtype=torch.float32, device=device
                    )
                    next_slice_batch, next_prev_preds_batch, next_prev_slices_batch = batch_obs_to_tensor(
                        [t.next_obs for t in batch], device
                    )
                    done_batch = torch.tensor(
                        [t.done for t in batch], dtype=torch.float32, device=device
                    )
                    
                    # Forward pass with mixed precision
                    if scaler is not None:
                        with amp.autocast():
                            q_s = policy_net(slice_pixels_batch, prev_preds_batch, prev_slices_batch)
                            q_s_a = q_s.gather(dim=-1, index=act_batch.unsqueeze(-1)).squeeze(-1)
                            
                            with torch.no_grad():
                                q_next = target_net(next_slice_batch, next_prev_preds_batch, next_prev_slices_batch)
                                q_next_max = q_next.max(dim=-1).values
                            
                            not_done = (1.0 - done_batch).unsqueeze(-1).unsqueeze(-1)
                            target = rew_batch + gamma * not_done * q_next_max
                            
                            # TD error for prioritized replay
                            td_error = (q_s_a - target).abs().mean(dim=(1, 2))
                            
                            if weights is not None:
                                loss = (weights.unsqueeze(-1).unsqueeze(-1) * 
                                       (q_s_a - target) ** 2).mean()
                            else:
                                loss = nn.functional.mse_loss(q_s_a, target)
                            
                            loss = loss / gradient_accumulation_steps
                        
                        scaler.scale(loss).backward()
                        accumulated_loss += loss.item()
                        accumulation_count += 1
                        
                        if accumulation_count >= gradient_accumulation_steps:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=5.0)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            
                            epoch_losses.append(accumulated_loss * gradient_accumulation_steps)
                            accumulated_loss = 0.0
                            accumulation_count = 0
                    else:
                        q_s = policy_net(slice_pixels_batch, prev_preds_batch, prev_slices_batch)
                        q_s_a = q_s.gather(dim=-1, index=act_batch.unsqueeze(-1)).squeeze(-1)
                        
                        with torch.no_grad():
                            q_next = target_net(next_slice_batch, next_prev_preds_batch, next_prev_slices_batch)
                            q_next_max = q_next.max(dim=-1).values
                        
                        not_done = (1.0 - done_batch).unsqueeze(-1).unsqueeze(-1)
                        target = rew_batch + gamma * not_done * q_next_max
                        
                        td_error = (q_s_a - target).abs().mean(dim=(1, 2))
                        
                        if weights is not None:
                            loss = (weights.unsqueeze(-1).unsqueeze(-1) * 
                                   (q_s_a - target) ** 2).mean()
                        else:
                            loss = nn.functional.mse_loss(q_s_a, target)
                        
                        loss = loss / gradient_accumulation_steps
                        loss.backward()
                        accumulated_loss += loss.item()
                        accumulation_count += 1
                        
                        if accumulation_count >= gradient_accumulation_steps:
                            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=5.0)
                            optimizer.step()
                            optimizer.zero_grad()
                            
                            epoch_losses.append(accumulated_loss * gradient_accumulation_steps)
                            accumulated_loss = 0.0
                            accumulation_count = 0
                    
                    # Update priorities
                    if use_prioritized_replay and indices is not None:
                        replay.update_priorities(indices, td_error.detach().cpu().numpy())
                
                # Target network update
                if global_step % target_update_every == 0:
                    target_net.load_state_dict(policy_net.state_dict())
            
            epoch_return += episode_return
            train_returns.append(episode_return)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Epoch summary
        avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0
        losses.append(avg_epoch_loss)
        
        # Validation
        val_metrics, val_pred = validate(policy_net, val_volume, val_mask, device, config)
        
        for key in ["iou", "f1", "accuracy", "coverage"]:
            if key in val_metrics:
                val_metrics_history[key].append(val_metrics[key])
        val_returns.append(val_metrics["reward"])
        
        # Save best model
        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_metrics["iou"],
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"  [NEW BEST] Saved model with IoU: {val_metrics['iou']:.4f}")
        
        # Periodic save
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, f'model_epoch_{epoch+1}.pth'))
        
        dt = time.time() - t0
        print(f"\nEpoch {epoch+1}/{num_epochs} | Time: {dt:.1f}s | LR: {scheduler.get_last_lr()[0]:.2e}")
        print(f"  Train - Loss: {avg_epoch_loss:.4f} | Avg Return: {epoch_return/episodes_per_epoch:.2f} | ε: {epsilon:.3f}")
        print(f"  Val   - IoU: {val_metrics['iou']:.4f} | F1: {val_metrics['f1']:.4f} | "
              f"Acc: {val_metrics['accuracy']:.4f} | Cov: {val_metrics['coverage']:.4f}")
    
    return {
        "policy_net": policy_net,
        "target_net": target_net,
        "losses": losses,
        "train_returns": train_returns,
        "val_returns": val_returns,
        "train_metrics": train_metrics_history,
        "val_metrics": val_metrics_history,
        "best_val_iou": best_val_iou,
    }


def reconstruct_volume(policy_net, volume, mask, device, config):
    """Reconstruct full volume using trained model."""
    policy_net.eval()
    
    env = PathReconstructionEnvV2(
        volume=volume,
        mask=mask,
        continuity_coef=config.get('continuity_coef', 0.1),
        continuity_decay_factor=config.get('continuity_decay_factor', 0.7),
        history_len=config.get('history_len', 5),
        use_class_weights=False,
        random_slice_order=False,
    )
    
    pred = np.zeros(volume.shape, dtype=np.uint8)
    obs, _ = env.reset()
    done = False
    
    with torch.no_grad():
        while not done:
            slice_pixels, prev_preds, prev_slices = obs_to_tensor(obs, device)
            q = policy_net(slice_pixels, prev_preds, prev_slices)
            a = q.argmax(dim=-1).cpu().numpy()[0]
            
            next_obs, reward, terminated, truncated, info = env.step(a.flatten())
            
            slice_idx = info.get("slice_index", None)
            if slice_idx is not None:
                pred[slice_idx, :, :] = a.astype(np.uint8)
            
            obs = next_obs
            done = terminated or truncated
    
    policy_net.train()
    return pred


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='data/ct_like/rapids-p/train')
    parser.add_argument('--val_dir', type=str, default='data/ct_like/rapids-p/val')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--episodes_per_epoch', type=int, default=10)
    parser.add_argument('--slices_per_episode', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--model_type', type=str, default='perpixel', choices=['perpixel', 'unet'])
    parser.add_argument('--base_channels', type=int, default=32)
    parser.add_argument('--foreground_weight', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Load data
    train_files = sorted(os.listdir(args.train_dir))
    val_files = sorted(os.listdir(args.val_dir))
    
    # Load training volume and mask
    train_volume = None
    train_mask = None
    for f in train_files:
        path = os.path.join(args.train_dir, f)
        if f.endswith('.npy'):
            data = np.load(path)
            if 'mask' in f.lower():
                train_mask = data
            else:
                train_volume = data
        elif f.endswith('.nii') or f.endswith('.nii.gz'):
            data = nib.load(path).get_fdata()
            if 'mask' in f.lower():
                train_mask = data
            else:
                train_volume = data
    
    # Load validation volume and mask
    val_volume = None
    val_mask = None
    for f in val_files:
        path = os.path.join(args.val_dir, f)
        if f.endswith('.npy'):
            data = np.load(path)
            if 'mask' in f.lower():
                val_mask = data
            else:
                val_volume = data
        elif f.endswith('.nii') or f.endswith('.nii.gz'):
            data = nib.load(path).get_fdata()
            if 'mask' in f.lower():
                val_mask = data
            else:
                val_volume = data
    
    print(f"Loaded training volume: {train_volume.shape}, mask: {train_mask.shape}")
    print(f"Loaded validation volume: {val_volume.shape}, mask: {val_mask.shape}")
    
    # Train
    results = train_on_real_volumes(
        train_volume=train_volume,
        train_mask=train_mask,
        val_volume=val_volume,
        val_mask=val_mask,
        num_epochs=args.epochs,
        episodes_per_epoch=args.episodes_per_epoch,
        slices_per_episode=args.slices_per_episode,
        batch_size=args.batch_size,
        lr=args.lr,
        model_type=args.model_type,
        base_channels=args.base_channels,
        foreground_weight=args.foreground_weight,
        seed=args.seed,
    )
    
    print(f"\nTraining complete. Best validation IoU: {results['best_val_iou']:.4f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(results["losses"])
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(results["train_returns"], alpha=0.3, label="Episode")
    window = args.episodes_per_epoch
    ma = [np.mean(results["train_returns"][max(0, i-window+1):i+1]) 
          for i in range(len(results["train_returns"]))]
    axes[0, 1].plot(ma, label="Moving Avg")
    axes[0, 1].set_title("Training Returns")
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(results["val_metrics"]["iou"], marker='o', label="IoU")
    axes[1, 0].plot(results["val_metrics"]["f1"], marker='s', label="F1")
    axes[1, 0].set_title("Validation Metrics")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(results["val_returns"], marker='o')
    axes[1, 1].set_title("Validation Reward")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig("dqn_slice_based/results/training_results.png", dpi=300)
    plt.show()
