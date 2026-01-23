import argparse
import cv2
import numpy as np
import os
import random
import time
import torch
import torch.nn.functional as F

from dqn_patch_based.dqn import NeighborContextCNN, ReplayBuffer, obs_to_tensor, batch_obs_to_tensor, epsilon_greedy_action
from dqn_patch_based.env import PatchClassificationEnv

from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image


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


def update_dataset(data_dir, size=(256, 256)):
    for root, _, files in os.walk(data_dir):
        for file in sorted(files):
            file_path = os.path.join(root, file)
            
            # Skip the mask folder (used in other datasets)
            if os.sep + "mask" + os.sep in file_path:
                continue
            
            if "train" in root:
                if "segm" in root:  # Check if in ground_truth folder
                    train_masks_paths.append(file_path)
                    img = np.array(Image.open(file_path).convert("L").resize(size))
                    if img is not None:
                        train_masks.append(img)
                elif "images" in root:  # Check if in images folder
                    train_imgs_paths.append(file_path)
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, size)
                    if img is not None:
                        train_imgs.append(img)
                    
            elif "val" in root:
                if "segm" in root:  # Check if in ground_truth folder
                    val_masks_paths.append(file_path)
                    img = np.array(Image.open(file_path).convert("L").resize(size))
                    if img is not None:
                        val_masks.append(img)
                elif "images" in root:  # Check if in images folder
                    val_imgs_paths.append(file_path)
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, size)  
                    if img is not None:
                        val_imgs.append(img)


def validate(policy_net, 
             val_pairs, 
             device, 
             patch_size=16,
             base_coef=1.0,
             continuity_coef=0.1):
    """Run validation and return average metrics, loss, and reward."""
    policy_net.eval()
    all_metrics = []
    val_rewards = []
    
    with torch.no_grad():
        for image, mask in val_pairs:
            # Reconstruct and compute metrics
            pred = reconstruct_image(policy_net, 
                                     image, 
                                     mask,
                                     patch_size=patch_size,
                                     base_coef=base_coef,
                                     continuity_coef=continuity_coef,
                                     device=device)
            metrics = compute_metrics(pred, mask)
            all_metrics.append(metrics)
            
            # Compute validation reward by running through environment
            env = PatchClassificationEnv(image=image,
                                         mask=mask,
                                         patch_size=patch_size,
                                         base_coef=base_coef,
                                         continuity_coef=continuity_coef)
            obs, _ = env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                patch_pixels, below_patches, above_patches, left_patch, neighbor_masks = obs_to_tensor(obs, device)
                q = policy_net(patch_pixels, below_patches, above_patches, left_patch, neighbor_masks)
                a = q.argmax(dim=-1).cpu().numpy()  # (N, N)
                next_obs, reward, terminated, truncated, info = env.step(a)
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


def train_dqn_on_images(
    image_mask_pairs,    # list of (image, mask) tuples
    val_pairs=None,      
    num_epochs=20,
    patch_size=16,
    buffer_capacity=10000,
    batch_size=64,
    gamma=0.95,
    lr=1e-3,
    target_update_every=500,
    start_epsilon=1.0,
    end_epsilon=0.01,
    epsilon_decay_epochs=15,
    base_coef=1.0,
    continuity_coef=0.1,
    save_dir=None,
    seed=42,
    device=None,
):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Infer image shape
    sample_image, _ = image_mask_pairs[0]
    C = 1 if sample_image.ndim == 2 else sample_image.shape[2]

    # Networks - use new neighbor context architecture
    policy_net = NeighborContextCNN(input_channels=C, patch_size=patch_size).to(device)
    target_net = NeighborContextCNN(input_channels=C, patch_size=patch_size).to(device)
    
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    replay = ReplayBuffer(buffer_capacity)

    epsilon = start_epsilon
    # Exponential decay per epoch: epsilon = end + (start - end) * exp(-epoch / decay_epochs)
    epsilon_decay_rate = -np.log(end_epsilon / start_epsilon) / epsilon_decay_epochs

    global_step = 0
    train_losses = []
    val_losses = []

    train_returns = []
    val_returns = []
    base_returns = []
    continuity_returns = []
    
    epsilons = []
    
    # Metrics tracking
    train_metrics_history, val_metrics_history = [{"iou": [], 
                                                   "f1": [], 
                                                   "accuracy": [], 
                                                   "coverage": [], 
                                                   "precision": [], 
                                                   "recall": []} for x in range(2)]
    
    # Info tracking
    info_history = {
        "true_positives": [],
        "false_positives": [],
        "false_negatives": []
    }

    for epoch in range(num_epochs):
        # Update epsilon at the start of each epoch
        epsilon = end_epsilon + (start_epsilon - end_epsilon) * np.exp(-epsilon_decay_rate * epoch)
        epsilon = max(end_epsilon, epsilon)
        
        random.shuffle(image_mask_pairs)
        t0 = time.time()

        epoch_loss = 0
        c = 0
        epoch_train_metrics = []

        for image, mask in tqdm(image_mask_pairs, desc=f"Epoch {epoch+1}/{num_epochs}"):
            env = PatchClassificationEnv(image=image, 
                                         mask=mask, 
                                         patch_size=patch_size,
                                         base_coef=base_coef,
                                         continuity_coef=continuity_coef,
                                         start_from_bottom_left=True)
            obs, _ = env.reset()

            done = False
            img_return = 0.0
            base_return = 0.0
            continuity_return = 0.0
            
            # Track info for this episode
            episode_info = {
                "true_positives": [],
                "false_positives": [],
                "false_negatives": []
            }

            while not done:
                patch_pixels, below_patches, above_patches, left_patch, neighbor_masks = obs_to_tensor(obs, device)
                with torch.no_grad():
                    q = policy_net(patch_pixels, below_patches, above_patches, left_patch, neighbor_masks)  # (N, N, 2)
                a = epsilon_greedy_action(q, epsilon)  # (N, N)

                # Keep action on GPU, only convert to numpy when needed for env
                next_obs, reward, terminated, truncated, info = env.step(a.cpu().numpy())
                base_reward = info["base_reward"]
                continuity_reward = info["continuity_reward"]
                
                # Track info metrics
                for key in episode_info.keys():
                    if key in info:
                        episode_info[key].append(info[key])
                
                done = terminated or truncated
                base_return += base_reward
                continuity_return += continuity_reward
                img_return += reward

                # Store scalar reward (not pixel rewards)
                replay.push(
                    obs,
                    a.cpu().numpy(),
                    reward,  # scalar reward
                    next_obs,
                    done
                )

                obs = next_obs
                global_step += 1

                # Optimize - batched processing with scalar rewards
                if len(replay) >= batch_size:
                    batch = replay.sample(batch_size)

                    # Batch conversion - single GPU transfer
                    patch_pixels_batch, below_patches_batch, above_patches_batch, left_patch_batch, neighbor_masks_batch = batch_obs_to_tensor([t.obs for t in batch], device)
                    act_batch = torch.tensor(np.stack([t.action for t in batch]), dtype=torch.int64, device=device)  # (B, N, N)
                    rew_batch = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device)  # (B,) scalar rewards
                    next_patch_pixels_batch, next_below_patches_batch, next_above_patches_batch, next_left_patch_batch, next_neighbor_masks_batch = batch_obs_to_tensor([t.next_obs for t in batch], device)
                    done_batch = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device)

                    # Batched forward passes
                    q_s = policy_net(patch_pixels_batch, below_patches_batch, above_patches_batch, left_patch_batch, neighbor_masks_batch)  # (B, N, N, 2)
                    q_s_a = q_s.gather(dim=-1, index=act_batch.unsqueeze(-1)).squeeze(-1)  # (B, N, N)
                    
                    # Aggregate Q-values per patch (mean over pixels)
                    q_s_a_mean = q_s_a.mean(dim=(1, 2))  # (B,)

                    with torch.no_grad():
                        q_next = target_net(next_patch_pixels_batch, next_below_patches_batch, next_above_patches_batch, next_left_patch_batch, next_neighbor_masks_batch)  # (B, N, N, 2)
                        q_next_max = q_next.max(dim=-1).values  # (B, N, N)
                        q_next_max_mean = q_next_max.mean(dim=(1, 2))  # (B,)

                    not_done = (1.0 - done_batch)  # (B,)
                    target = rew_batch + gamma * not_done * q_next_max_mean  # (B,) scalar targets

                    loss = F.mse_loss(q_s_a_mean, target)
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
            train_returns.append(img_return)
            epsilons.append(epsilon)
            
            # Aggregate info for this episode (average across patches)
            for key in episode_info.keys():
                if len(episode_info[key]) > 0:
                    info_history[key].append(np.mean(episode_info[key]))
                else:
                    info_history[key].append(0.0)
            
            # Compute metrics for this training image
            pred_train = reconstruct_image(policy_net, 
                                           image, 
                                           mask,
                                           patch_size=patch_size,
                                           base_coef=base_coef,
                                           continuity_coef=continuity_coef,
                                           device=device)
                                           
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
        avg_val_metrics = {"f1": 0.0, "iou": 0.0, "accuracy": 0.0, "coverage": 0.0, "reward": 0.0}
        if val_pairs:
            avg_val_metrics = validate(
                policy_net, val_pairs, device,
                patch_size=patch_size,
                base_coef=base_coef,
                continuity_coef=continuity_coef,
            )
            
            # Compute validation loss
            policy_net.eval()
            val_epoch_loss = 0
            val_c = 0
            
            with torch.no_grad():
                for image, mask in val_pairs:
                    env = PatchClassificationEnv(image=image,
                                                 mask=mask,
                                                 patch_size=patch_size,
                                                 base_coef=base_coef,
                                                 continuity_coef=continuity_coef)
                    obs, _ = env.reset()
                    done = False
                    
                    while not done:
                        patch_pixels, below_patches, above_patches, left_patch, neighbor_masks = obs_to_tensor(obs, device)
                        q = policy_net(patch_pixels, below_patches, above_patches, left_patch, neighbor_masks)
                        a = q.argmax(dim=-1)
                        
                        next_obs, reward, terminated, truncated, info = env.step(a.cpu().numpy())
                        done = terminated or truncated
                        
                        # Compute validation loss with scalar rewards
                        next_patch_pixels, next_below_patches, next_above_patches, next_left_patch, next_neighbor_masks = obs_to_tensor(next_obs, device)
                        rew_tensor = torch.tensor([reward], dtype=torch.float32, device=device)
                        done_tensor = torch.tensor([done], dtype=torch.float32, device=device)
                        
                        q_s_a = q.unsqueeze(0).gather(dim=-1, index=a.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
                        q_s_a_mean = q_s_a.mean(dim=(1, 2))  # (1,)
                        
                        q_next = target_net(next_patch_pixels, next_below_patches, next_above_patches, next_left_patch, next_neighbor_masks)
                        q_next_max = q_next.unsqueeze(0).max(dim=-1).values
                        q_next_max_mean = q_next_max.mean(dim=(1, 2))  # (1,)
                        
                        not_done = (1.0 - done_tensor)
                        target = rew_tensor + gamma * not_done * q_next_max_mean
                        
                        val_loss = F.mse_loss(q_s_a_mean, target)
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

            pred = reconstruct_image(policy_net, 
                                     val_pairs[0][0], 
                                     val_pairs[0][1], 
                                     patch_size=patch_size,
                                     base_coef=base_coef,
                                     continuity_coef=continuity_coef,
                                     device=device)
            if epoch % 5 == 0:
                visualize_result(val_pairs[0][0], val_pairs[0][1], pred, f"dqn_patch_based/results/{save_dir}/reconstructions/reconstruction_1st_img_epoch_{epoch+1}.png")
        
        train_losses.append(epoch_loss / c if c > 0 else 0)
        dt = time.time() - t0
        torch.save(target_net.state_dict(), f"dqn_patch_based/models/{save_dir}/model_epoch_{epoch}_f1_{avg_val_metrics['f1']:.3f}.pth")
        
        # Print epoch summary
        avg_base = np.mean(base_returns[-len(image_mask_pairs):])
        avg_cont = np.mean(continuity_returns[-len(image_mask_pairs):])
        
        print(f"\nEpoch {epoch+1}/{num_epochs} | Time: {dt:.1f}s")
        print(f"  Train - Loss: {train_losses[-1]:.4f} | Avg Return: {np.mean(train_returns[-len(image_mask_pairs):]):.2f} | ε: {epsilon:.3f}")
        print(f"          IoU: {avg_train_metrics['iou']:.3f} | F1: {avg_train_metrics['f1']:.3f} | Acc: {avg_train_metrics['accuracy']:.3f} | Cov: {avg_train_metrics['coverage']:.3f}")
        print(f"          Rewards -> Base: {avg_base:.2f} | Cont: {avg_cont:.2f}")
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
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_metrics": train_metrics_history,
        "val_metrics": val_metrics_history,
        "info_history": info_history,
    }


def reconstruct_image(policy_net, 
                      image, 
                      mask, 
                      patch_size=16,
                      base_coef=1.0,
                      continuity_coef=0.1,
                      device=None):
    """Reconstruct image using trained policy network."""
    env = PatchClassificationEnv(image, 
                                 mask, 
                                 patch_size=patch_size,
                                 base_coef=base_coef,
                                 continuity_coef=continuity_coef,
                                 start_from_bottom_left=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    obs, _ = env.reset()

    canvas = env.reconstruct_blank()

    done = False
    policy_net.eval()
    with torch.no_grad():
        while not done:
            patch_pixels, below_patches, above_patches, left_patch, neighbor_masks = obs_to_tensor(obs, device)
            q = policy_net(patch_pixels, below_patches, above_patches, left_patch, neighbor_masks)  # (N, N, 2)
            a = q.argmax(dim=-1).cpu().numpy().astype(np.uint8)  # (N, N)
            
            # Write into canvas
            y0, x0 = env._order[env._idx]
            canvas[y0:y0+env.N, x0:x0+env.N] = a
            
            next_obs, reward, terminated, truncated, info = env.step(a)
            obs = next_obs
            done = terminated or truncated
    
    policy_net.train()
    return env.crop_to_original(canvas)


def visualize_result(img, mask, pred, save_path: str = None) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(9, 6))
    axs[0].imshow(img, cmap="gray")
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(mask, cmap="gray")
    axs[1].set_title("Mask")
    axs[1].axis("off")

    axs[2].imshow(pred, cmap="gray")
    axs[2].set_title("Patch-based Reconstruction")
    axs[2].axis("off")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--ct_like', action='store_true', help='Use CT-Like dataset')
    parser.add_argument('--drive', action='store_true', help='Use DRIVE dataset')
    parser.add_argument('--stare', action='store_true', help='Use STARE dataset')
    parser.add_argument('--image_size', type=int, nargs=2, default=[384, 384], help='Resize images to this size (H W)')

    # Model
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size (N x N)")

    # RL env
    parser.add_argument('--base_coef', type=float, default=1.0, help='Base reward coefficient')
    parser.add_argument("--cont_coef", type=float, default=0.1, help="Continuity reward coefficient")
    
    args = parser.parse_args()

    train_imgs = []
    train_masks = []
    
    val_imgs = []
    val_masks = []
    
    train_imgs_paths = []
    train_masks_paths = []
    
    val_imgs_paths = []
    val_masks_paths = []

    if args.ct_like:
        update_dataset("../data/ct_like/2d", size=args.image_size)
        save_dir = "ct_like"
    elif args.drive and not args.stare:
        update_dataset("../data/DRIVE", size=args.image_size)
        save_dir = "drive"
    elif args.stare and not args.drive:
        update_dataset("../data/STARE", size=args.image_size)    
        save_dir = "stare"
    elif args.drive and args.stare:
        update_dataset("../data/DRIVE", size=args.image_size)
        update_dataset("../data/STARE", size=args.image_size)    
        save_dir = "drive+stare"
    else:
        # Default fallback
        save_dir = "default"

    save_dir = os.path.join(save_dir, f"base_{args.base_coef}_cont_{args.cont_coef}")
    os.makedirs(f"dqn_patch_based/results/{save_dir}/reconstructions", exist_ok=True)
    os.makedirs(f"dqn_patch_based/models/{save_dir}", exist_ok=True)
                    
    results = train_dqn_on_images(
        list(zip(train_imgs, train_masks)),
        val_pairs=list(zip(val_imgs, val_masks)),
        num_epochs=args.epochs,
        patch_size=args.patch_size,
        start_epsilon=1.0,
        end_epsilon=0.01,
        epsilon_decay_epochs=15,
        base_coef=args.base_coef,
        continuity_coef=args.cont_coef,
        save_dir=save_dir,
        seed=123,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    for i, (img_test, mask_test) in enumerate(list(zip(val_imgs, val_masks))):
        pred = reconstruct_image(results["policy_net"], 
                                 img_test, 
                                 mask_test, 
                                 patch_size=args.patch_size,
                                 base_coef=args.base_coef,
                                 continuity_coef=args.cont_coef,
                                 neighbor_coef=args.neighbor_coef,
                                 history_len=args.history_len,
                                 future_len=args.future_len)
        visualize_result(img_test, mask_test, pred, save_path=f"dqn_patch_based/results/{save_dir}/reconstructions/final_image_{i+1}.png")

    # Plot training curves
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    
    # Returns (Train vs Val)
    axes[0, 0].plot(results["returns"], alpha=0.3, color='blue', label='Train (per image)')
    # Moving average for train returns
    window = len(train_imgs) if len(train_imgs) > 0 else 1
    train_returns_ma = [np.mean(results["returns"][max(0, i-window+1):i+1]) 
                        for i in range(len(results["returns"]))]
    axes[0, 0].plot(train_returns_ma, color='blue', linewidth=2, label='Train (moving average)')
    axes[0, 0].set_title("Episode Returns")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Epsilon
    axes[0, 1].plot(results["epsilons"], color='orange', linestyle='dashed')
    axes[0, 1].set_title("Exploration (Epsilon)")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].grid(True)
    
    # Loss (Train vs Val)
    axes[0, 2].plot(results["train_losses"], color='red', marker='o', label='Train')
    if results["val_losses"]:
        axes[0, 2].plot(results["val_losses"], color='darkred', marker='s', label='Val')
    axes[0, 2].set_title("MSE loss per Epoch")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # IoU
    axes[1, 0].plot(results["train_metrics"]["iou"], label="Train", marker='o', markersize=2)
    if results["val_metrics"]["iou"]:
        axes[1, 0].plot(results["val_metrics"]["iou"], label="Val", marker='s', markersize=2)
    axes[1, 0].set_title("IoU")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # F1 Score
    axes[1, 1].plot(results["train_metrics"]["f1"], label="Train", marker='o', markersize=2)
    if results["val_metrics"]["f1"]:
        axes[1, 1].plot(results["val_metrics"]["f1"], label="Val", marker='s', markersize=2)
    axes[1, 1].set_title("F1 Score")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Coverage
    axes[1, 2].plot(results["train_metrics"]["coverage"], label="Train", marker='o', markersize=2)
    if results["val_metrics"]["coverage"]:
        axes[1, 2].plot(results["val_metrics"]["coverage"], label="Val", marker='s', markersize=2)
    axes[1, 2].set_title("Coverage")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig(f"dqn_patch_based/results/{save_dir}/training_results.png", dpi=300)
    plt.close()
    
    # Plot reward components
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    
    # Moving average window
    window = len(train_imgs) if len(train_imgs) > 0 else 1

    # Base reward
    if len(results["base_returns"]) >= window:
        axes2[0].plot(np.convolve(results["base_returns"], 
                                  np.ones(window)/window, 
                                  mode='valid'), 
                      label="Base Reward", color='green')
    axes2[0].set_title("Base Reward (Moving Average)")
    axes2[0].set_ylabel("Reward")
    axes2[0].set_xlabel("Episode")
    axes2[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes2[0].grid(True)

    # Continuity reward
    if len(results["continuity_returns"]) >= window:
        axes2[1].plot(np.convolve(results["continuity_returns"], 
                                  np.ones(window)/window,
                                  mode='valid'), 
                      label="Continuity Reward", color='blue')
    axes2[1].set_title("Continuity Reward (Moving Average)")
    axes2[1].set_xlabel("Episode")
    axes2[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes2[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"dqn_patch_based/results/{save_dir}/reward_components_analysis.png", dpi=300)
    plt.close()
