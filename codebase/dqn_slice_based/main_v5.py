"""
Training script v5: Shape-Constrained Root Segmentation

Key improvements:
1. Shape constraints - penalize wide/blobby predictions
2. Masked intensity - only use in GT region
3. Precision penalty - explicit FP penalty
4. Curriculum learning - start with accuracy, add continuity
5. ROI mask as input instead of intensity hint
6. Cache environment setup (don't recreate every episode)
"""

import os
import sys
import time
import random
import argparse
import logging
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.amp as amp
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import deque
from matplotlib import pyplot as plt
from scipy.ndimage import label as connected_components

from dqn_slice_based.env_v5 import RootShapeEnv

logging.basicConfig(
    filename='training_v5.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)


# ============================================================================
# Network: UNet with ROI mask input
# ============================================================================

class RootUNet(nn.Module):
    """
    U-Net that takes:
    - Current slice (1 channel)
    - Previous predictions (history_len channels)
    - ROI mask (1 channel) - where roots could be
    
    Total: 2 + history_len channels
    """
    def __init__(self, history_len: int = 3, base_channels: int = 32):
        super().__init__()
        self.history_len = history_len
        
        in_channels = 1 + history_len + 1  # slice + history + roi_mask
        
        # Encoder
        self.enc1 = self._block(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = self._block(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = self._block(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self._block(base_channels * 4, base_channels * 8)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = self._block(base_channels * 8, base_channels * 4)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = self._block(base_channels * 4, base_channels * 2)
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = self._block(base_channels * 2, base_channels)
        
        # Output: Q-values for 2 actions
        self.out = nn.Conv2d(base_channels, 2, 1)
    
    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, slice_pixels, prev_preds, roi_mask):
        # Handle single sample
        if slice_pixels.dim() == 2:
            slice_pixels = slice_pixels.unsqueeze(0)
            prev_preds = prev_preds.unsqueeze(0)
            roi_mask = roi_mask.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        # Stack inputs
        x = torch.cat([
            slice_pixels.unsqueeze(1),
            prev_preds,
            roi_mask.unsqueeze(1),
        ], dim=1)
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool3(e3))
        
        # Decoder
        d3 = self.up3(b)
        if d3.shape != e3.shape:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        if d2.shape != e2.shape:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        if d1.shape != e1.shape:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        q = self.out(d1)  # (B, 2, H, W)
        q = q.permute(0, 2, 3, 1)  # (B, H, W, 2)
        
        if squeeze:
            q = q.squeeze(0)
        
        return q


# ============================================================================
# Replay Buffer
# ============================================================================

from collections import namedtuple
Transition = namedtuple("Transition", ("obs", "action", "reward", "next_obs", "done"))


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        self.buffer.append(Transition(*args))
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices], indices
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# Data Loading
# ============================================================================

def load_nifti_or_npy(path):
    if path.endswith('.npy'):
        return np.load(path)
    else:
        return nib.load(path).get_fdata()


def load_subvolume_pairs(data_dir):
    """
    Load paired volume and mask subvolumes from a directory.
    Files are named: {prefix}_d{d}_h{h}_w{w}_vol.npy and {prefix}_d{d}_h{h}_w{w}_mask.npy
    
    Returns:
        List of (volume, mask) tuples
    """
    pairs = []
    
    # Find all volume files and match with their masks
    vol_files = sorted([f for f in os.listdir(data_dir) if f.endswith("_vol.npy")])
    
    for vol_file in vol_files:
        mask_file = vol_file.replace("_vol.npy", "_mask.npy")
        
        vol_path = os.path.join(data_dir, vol_file)
        mask_path = os.path.join(data_dir, mask_file)
        
        if os.path.exists(mask_path):
            vol = np.load(vol_path).astype(np.float32)
            mask = np.load(mask_path).astype(np.float32)
            
            # Normalize volume to 0-1
            if vol.max() > 1:
                vol = vol / 255.0
            
            pairs.append((vol, mask))
            logger.info(f"Loaded: {vol_file} ({vol.shape}, fg={mask.mean()*100:.4f}%)")
        else:
            logger.warning(f"No mask found for {vol_file}")
    
    return pairs


# ============================================================================
# Utilities
# ============================================================================

def obs_to_tensor(obs, device):
    s = torch.FloatTensor(obs["slice_pixels"]).to(device)
    p = torch.FloatTensor(obs["prev_preds"]).to(device)
    r = torch.FloatTensor(obs["roi_mask"]).to(device)
    return s, p, r


def batch_obs_to_tensor(obs_list, device):
    s = torch.stack([torch.FloatTensor(o["slice_pixels"]) for o in obs_list]).to(device)
    p = torch.stack([torch.FloatTensor(o["prev_preds"]) for o in obs_list]).to(device)
    r = torch.stack([torch.FloatTensor(o["roi_mask"]) for o in obs_list]).to(device)
    return s, p, r


def compute_metrics(pred, mask):
    pred_bin = (pred > 0.5).astype(np.float32)
    mask_bin = (mask > 0).astype(np.float32)
    
    intersection = (pred_bin * mask_bin).sum()
    union = pred_bin.sum() + mask_bin.sum()
    
    iou = intersection / (union - intersection + 1e-8)
    dice = (2 * intersection) / (union + 1e-8)
    
    tp = intersection
    fp = (pred_bin * (1 - mask_bin)).sum()
    fn = ((1 - pred_bin) * mask_bin).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    _, pred_cc = connected_components(pred_bin)
    _, gt_cc = connected_components(mask_bin)
    
    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "pred_cc": int(pred_cc),
        "gt_cc": int(gt_cc),
    }


def reconstruct_volume(net, volume, mask, roi_masks, device, history_len, patch_size=None):
    """Reconstruct predictions for full volume."""
    net.eval()
    D, H, W = volume.shape
    pred = np.zeros((D, H, W), dtype=np.float32)
    
    vol = volume.astype(np.float32)
    if vol.max() > 1:
        vol = vol / 255.0
    
    prev_preds = deque(
        [np.zeros((H, W), dtype=np.float32) for _ in range(history_len)],
        maxlen=history_len
    )
    
    with torch.no_grad():
        for i in range(D):
            slice_t = torch.FloatTensor(vol[i]).unsqueeze(0).to(device)
            hist_t = torch.FloatTensor(np.stack(list(prev_preds))).unsqueeze(0).to(device)
            
            # Get ROI mask and ensure correct shape
            roi = roi_masks[i] if i < len(roi_masks) else np.zeros((H, W), dtype=np.float32)
            if roi.shape != (H, W):
                # Transpose if dimensions are swapped
                if roi.shape == (W, H):
                    roi = roi.T
                else:
                    # Resize if completely different
                    from scipy.ndimage import zoom
                    roi = zoom(roi, (H / roi.shape[0], W / roi.shape[1]), order=0)
            roi_t = torch.FloatTensor(roi).unsqueeze(0).to(device)
            
            if patch_size and patch_size < min(H, W):
                # Tiled inference
                slice_pred = np.zeros((H, W), dtype=np.float32)
                counts = np.zeros((H, W), dtype=np.float32)
                stride = patch_size // 2
                
                for y in range(0, H - patch_size + 1, stride):
                    for x in range(0, W - patch_size + 1, stride):
                        s_p = slice_t[:, y:y+patch_size, x:x+patch_size]
                        h_p = hist_t[:, :, y:y+patch_size, x:x+patch_size]
                        r_p = roi_t[:, y:y+patch_size, x:x+patch_size]
                        
                        q = net(s_p, h_p, r_p)
                        p = q.argmax(dim=-1).cpu().numpy().squeeze()
                        
                        slice_pred[y:y+patch_size, x:x+patch_size] += p
                        counts[y:y+patch_size, x:x+patch_size] += 1
                
                counts = np.maximum(counts, 1)
                pred[i] = (slice_pred / counts > 0.5).astype(np.float32)
            else:
                q = net(slice_t, hist_t, roi_t)
                pred[i] = q.argmax(dim=-1).cpu().numpy().squeeze().astype(np.float32)
            
            prev_preds.append(pred[i].copy())
    
    net.train()
    return pred


def visualize(volume, mask, pred, roi_masks, slice_idx, save_path):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    vol = volume[slice_idx]
    if vol.max() > 1:
        vol = vol / 255.0
    
    axes[0].imshow(vol, cmap='gray')
    axes[0].set_title(f'CT Slice {slice_idx}')
    axes[0].axis('off')
    
    axes[1].imshow(roi_masks[slice_idx], cmap='hot')
    axes[1].set_title('ROI Mask')
    axes[1].axis('off')
    
    axes[2].imshow(mask[slice_idx], cmap='gray')
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')
    
    axes[3].imshow(pred[slice_idx], cmap='gray')
    axes[3].set_title('Prediction')
    axes[3].axis('off')
    
    overlay = np.zeros((*vol.shape, 3))
    overlay[:, :, 0] = vol
    overlay[:, :, 1] = pred[slice_idx] * 0.7 + vol * 0.3
    overlay[:, :, 2] = mask[slice_idx] * 0.7 + vol * 0.3
    axes[4].imshow(overlay)
    axes[4].set_title('Overlay (G=Pred, B=GT)')
    axes[4].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Training
# ============================================================================

def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Load subvolume pairs
    logger.info(f"Loading training subvolumes from {args.train_dir}")
    train_pairs = load_subvolume_pairs(args.train_dir)
    logger.info(f"Loaded {len(train_pairs)} training subvolumes")
    
    logger.info(f"Loading validation subvolumes from {args.val_dir}")
    val_pairs = load_subvolume_pairs(args.val_dir)
    logger.info(f"Loaded {len(val_pairs)} validation subvolumes")
    
    if len(train_pairs) == 0:
        raise ValueError(f"No training data found in {args.train_dir}")
    if len(val_pairs) == 0:
        raise ValueError(f"No validation data found in {args.val_dir}")
    
    # Get dimensions from first subvolume
    sample_vol, sample_mask = train_pairs[0]
    D, H, W = sample_vol.shape
    logger.info(f"Subvolume shape: {sample_vol.shape}")
    
    # Observation size is the slice size (64x64 for subvolumes)
    obs_size = H  # No patches needed, slices are already small
    policy_net = RootUNet(history_len=args.history_len, base_channels=args.base_channels).to(device)
    target_net = RootUNet(history_len=args.history_len, base_channels=args.base_channels).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    n_params = sum(p.numel() for p in policy_net.parameters() if p.requires_grad)
    logger.info(f"Model params: {n_params:,}")
    
    optimizer = optim.AdamW(policy_net.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    replay = ReplayBuffer(args.buffer_capacity)
    
    os.makedirs(args.save_dir, exist_ok=True)
    results_dir = os.path.join(args.save_dir, 'results_v5')
    os.makedirs(results_dir, exist_ok=True)
    
    global_step = 0
    best_iou = 0.0
    
    logger.info(f"Starting training: {args}")
    
    for epoch in range(args.epochs):
        t0 = time.time()
        
        # Epsilon schedule
        if epoch < args.warmup_epochs:
            epsilon = args.start_epsilon
        else:
            progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
            epsilon = args.end_epsilon + (args.start_epsilon - args.end_epsilon) * (1 - progress) ** 2
        epsilon = max(args.end_epsilon, epsilon)
        
        epoch_losses = []
        epoch_rewards = []
        reward_components = {
            "dice": [], "recall": [], "precision": [],
            "shape": [], "intensity": [], "continuity": [], "boundary": []
        }
        
        policy_net.train()
        
        # Shuffle training subvolumes each epoch
        random.shuffle(train_pairs)
        
        # Train on each subvolume
        for subvol_idx, (train_vol, train_mask) in enumerate(train_pairs):
            # Create environment for this subvolume
            env = RootShapeEnv(
                volume=train_vol,
                mask=train_mask,
                dice_coef=args.dice_coef,
                recall_coef=args.recall_coef,
                precision_coef=args.precision_coef,
                shape_coef=args.shape_coef,
                max_avg_width=args.max_avg_width,
                intensity_coef=args.intensity_coef,
                roi_dilation=args.roi_dilation,
                continuity_coef=args.continuity_coef,
                boundary_coef=args.boundary_coef,
                history_len=args.history_len,
                slices_per_episode=min(args.slices_per_episode, train_vol.shape[0]),
                patch_size=None,  # No patches for 64x64 slices
                patch_jitter=0,
            )
            
            obs, _ = env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                s, p, r = obs_to_tensor(obs, device)
                
                with torch.no_grad():
                    q = policy_net(s, p, r)
                    if q.dim() == 3:
                        pass
                    else:
                        q = q.squeeze(0)
                
                if random.random() < epsilon:
                    action = torch.randint(0, 2, (obs_size, obs_size), device=device)
                else:
                    action = q.argmax(dim=-1)
                
                next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy().flatten())
                done = terminated or truncated
                episode_reward += reward
                
                # Track components
                for key in reward_components:
                    if key in info:
                        reward_components[key].append(info[key])
                
                replay.push(obs, action.cpu().numpy(), reward, next_obs, done)
                obs = next_obs
                global_step += 1
                
                # Training
                if len(replay) >= args.batch_size:
                    batch, _ = replay.sample(args.batch_size)
                    
                    s_b, p_b, r_b = batch_obs_to_tensor([t.obs for t in batch], device)
                    a_b = torch.tensor(np.stack([t.action for t in batch]), dtype=torch.int64, device=device)
                    rew_b = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device)
                    ns_b, np_b, nr_b = batch_obs_to_tensor([t.next_obs for t in batch], device)
                    done_b = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device)
                    
                    if scaler:
                        with amp.autocast('cuda'):
                            q_s = policy_net(s_b, p_b, r_b)
                            q_s_a = q_s.gather(-1, a_b.unsqueeze(-1)).squeeze(-1).mean(dim=(1, 2))
                            
                            with torch.no_grad():
                                q_n = target_net(ns_b, np_b, nr_b)
                                q_n_max = q_n.max(dim=-1).values.mean(dim=(1, 2))
                            
                            target = rew_b + args.gamma * (1 - done_b) * q_n_max
                            loss = F.mse_loss(q_s_a, target)
                        
                        optimizer.zero_grad()
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 5.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        q_s = policy_net(s_b, p_b, r_b)
                        q_s_a = q_s.gather(-1, a_b.unsqueeze(-1)).squeeze(-1).mean(dim=(1, 2))
                        
                        with torch.no_grad():
                            q_n = target_net(ns_b, np_b, nr_b)
                            q_n_max = q_n.max(dim=-1).values.mean(dim=(1, 2))
                        
                        target = rew_b + args.gamma * (1 - done_b) * q_n_max
                        loss = F.mse_loss(q_s_a, target)
                        
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 5.0)
                        optimizer.step()
                    
                    epoch_losses.append(loss.item())
                
                if global_step % args.target_update_every == 0:
                    target_net.load_state_dict(policy_net.state_dict())
            
            epoch_rewards.append(episode_reward)
        
        scheduler.step()
        
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        
        # Validation - evaluate on all validation subvolumes
        all_val_metrics = []
        for val_vol, val_mask in val_pairs:
            val_env = RootShapeEnv(
                volume=val_vol,
                mask=val_mask,
                patch_size=None,
                history_len=args.history_len,
            )
            
            val_pred = reconstruct_volume(
                policy_net, val_vol, val_mask, val_env.roi_masks,
                device, args.history_len,
                patch_size=None
            )
            metrics = compute_metrics(val_pred, val_mask)
            all_val_metrics.append(metrics)
        
        # Average metrics across validation subvolumes
        val_metrics = {
            key: np.mean([m[key] for m in all_val_metrics])
            for key in all_val_metrics[0].keys()
        }
        
        elapsed = time.time() - t0
        
        # Log
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"IoU: {val_metrics['iou']:.4f} | "
            f"Dice: {val_metrics['dice']:.4f} | "
            f"Prec: {val_metrics['precision']:.4f} | "
            f"Rec: {val_metrics['recall']:.4f} | "
            f"CC: {val_metrics['pred_cc']} | "
            f"Eps: {epsilon:.3f} | "
            f"Time: {elapsed:.1f}s"
        )
        
        # Reward breakdown
        logger.info(
            f"  Rewards: dice={np.mean(reward_components['dice']):.3f}, "
            f"rec={np.mean(reward_components['recall']):.3f}, "
            f"prec={np.mean(reward_components['precision']):.3f}, "
            f"shape={np.mean(reward_components['shape']):.3f}, "
            f"int={np.mean(reward_components['intensity']):.3f}, "
            f"cont={np.mean(reward_components['continuity']):.3f}, "
            f"bound={np.mean(reward_components['boundary']):.3f}"
        )
        
        # Save best
        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            torch.save({
                'epoch': epoch,
                'model': policy_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iou': best_iou,
            }, os.path.join(args.save_dir, 'best_model_v5.pth'))
            logger.info(f"  New best IoU: {best_iou:.4f}")
        
        # Visualize periodically (use first validation subvolume)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            val_vol_viz, val_mask_viz = val_pairs[0]
            val_env_viz = RootShapeEnv(
                volume=val_vol_viz,
                mask=val_mask_viz,
                patch_size=None,
                history_len=args.history_len,
            )
            val_pred_viz = reconstruct_volume(
                policy_net, val_vol_viz, val_mask_viz, val_env_viz.roi_masks,
                device, args.history_len, patch_size=None
            )
            
            fg_slices = np.where(val_mask_viz.sum(axis=(1, 2)) > 0)[0]
            if len(fg_slices) > 0:
                mid = fg_slices[len(fg_slices) // 2]
                visualize(
                    val_vol_viz, val_mask_viz, val_pred_viz, val_env_viz.roi_masks,
                    mid, os.path.join(results_dir, f'slice_{mid}_epoch_{epoch+1}.png')
                )
    
    logger.info(f"Training complete. Best IoU: {best_iou:.4f}")


def main():
    parser = argparse.ArgumentParser()
    
    # Data - updated for subvolume dataset
    parser.add_argument('--train_dir', default='../data/rapids-p/subvolumes_100x64x64/train')
    parser.add_argument('--val_dir', default='../data/rapids-p/subvolumes_100x64x64/val')
    
    # Training - adjusted for subvolumes
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--slices_per_episode', type=int, default=100)  # Full subvolume depth
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--buffer_capacity', type=int, default=30000)  # Smaller buffer
    
    # Epsilon
    parser.add_argument('--start_epsilon', type=float, default=1.0)
    parser.add_argument('--end_epsilon', type=float, default=0.05)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    
    # Model
    parser.add_argument('--history_len', type=int, default=3)
    parser.add_argument('--base_channels', type=int, default=32)
    parser.add_argument('--target_update_every', type=int, default=200)
    
    # Reward coefficients - BALANCED for accuracy first
    parser.add_argument('--dice_coef', type=float, default=1.0)
    parser.add_argument('--recall_coef', type=float, default=0.8)
    parser.add_argument('--precision_coef', type=float, default=0.5)   # Penalize FP
    parser.add_argument('--shape_coef', type=float, default=0.3)       # Shape constraint
    parser.add_argument('--intensity_coef', type=float, default=0.2)   # Lower
    parser.add_argument('--continuity_coef', type=float, default=0.1)  # Start low
    parser.add_argument('--boundary_coef', type=float, default=0.2)
    parser.add_argument('--max_avg_width', type=float, default=6.0)    # Smaller for 64x64
    parser.add_argument('--roi_dilation', type=int, default=10)        # Smaller dilation
    
    # Output
    parser.add_argument('--save_dir', default='dqn_slice_based/models')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
