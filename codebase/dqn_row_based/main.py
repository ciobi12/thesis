import cv2

import numpy as np
import os
import random
import time
import torch
import torch.optim as optim
import torch.nn as nn

from dqn_row_based.dqn import PerPixelCNN, ReplayBuffer, obs_to_tensor, batch_obs_to_tensor, epsilon_greedy_action
from dqn_row_based.env import PathReconstructionEnv

from matplotlib import pyplot as plt
from tqdm import tqdm

USE_ARTIFACTS = False

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
        for image, mask in val_pairs:
            # Reconstruct and compute metrics
            pred = reconstruct_image(policy_net, 
                                     image, 
                                     mask,
                                     continuity_coef=continuity_coef,
                                     continuity_decay_factor=continuity_decay_factor,
                                     device=device)
            metrics = compute_metrics(pred, mask)
            all_metrics.append(metrics)
            
            # Compute validation reward by running through environment
            env = PathReconstructionEnv(image=image,
                                        mask=mask,
                                        continuity_coef=continuity_coef,
                                        continuity_decay_factor=continuity_decay_factor,
                                        history_len=5)
            obs, _ = env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                obs_tensor = obs_to_tensor(obs, device)
                q = policy_net(obs_tensor)
                a = q.argmax(dim=-1).cpu().numpy()
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
    image_mask_pairs,      # list of (image, mask) tuples
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

    # Infer image shape
    sample_image, _ = image_mask_pairs[0]
    H, W = sample_image.shape[:2]
    C = 1 if sample_image.ndim == 2 else sample_image.shape[2]

    # Networks
    policy_net = PerPixelCNN(W=W, C=C, history_len=5).to(device)
    target_net = PerPixelCNN(W=W, C=C, history_len=5).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    replay = ReplayBuffer(buffer_capacity)

    epsilon = start_epsilon
    # Exponential decay per epoch: epsilon = end + (start - end) * exp(-epoch / decay_epochs)
    epsilon_decay_rate = -np.log(end_epsilon / start_epsilon) / epsilon_decay_epochs

    global_step = 0
    losses = []
    val_losses = []  # Track validation losses
    train_returns = []
    val_returns = []  # Track validation returns
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
        
        random.shuffle(image_mask_pairs)
        epoch_return = 0.0
        t0 = time.time()

        epoch_loss = 0
        c = 0
        epoch_train_metrics = []

        for image, mask in tqdm(image_mask_pairs, desc=f"Epoch {epoch+1}/{num_epochs}"):
            env = PathReconstructionEnv(image=image, 
                                        mask=mask, 
                                        continuity_coef=continuity_coef, 
                                        continuity_decay_factor=continuity_decay_factor,
                                        history_len=5)
            obs, _ = env.reset()

            done = False
            img_return = 0.0

            while not done:
                obs_tensor = obs_to_tensor(obs, device)
                with torch.no_grad():
                    q = policy_net(obs_tensor)  # (W, 2)
                a = epsilon_greedy_action(q, epsilon)  # (W,)

                # Keep action on GPU, only convert to numpy when needed for env
                next_obs, reward, terminated, truncated, info = env.step(a.cpu().numpy())
                pixel_rewards = info["pixel_rewards"]
                done = terminated or truncated
                img_return += reward

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
                    obs_batch = batch_obs_to_tensor([t.obs for t in batch], device)
                    act_batch = torch.tensor(np.stack([t.action for t in batch]), dtype=torch.int64, device=device)
                    rew_batch = torch.tensor(np.stack([t.pixel_rewards for t in batch]), dtype=torch.float32, device=device)
                    next_obs_batch = batch_obs_to_tensor([t.next_obs for t in batch], device)
                    done_batch = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device)

                    # Batched forward passes - much faster on GPU
                    q_s = policy_net(obs_batch)  # (B, W, 2)
                    q_s_a = q_s.gather(dim=-1, index=act_batch.unsqueeze(-1)).squeeze(-1)  # (B, W)

                    with torch.no_grad():
                        q_next = target_net(next_obs_batch)  # (B, W, 2)
                        q_next_max = q_next.max(dim=-1).values  # (B, W)

                    not_done = (1.0 - done_batch).unsqueeze(-1)  # (B, 1)
                    target = rew_batch + gamma * not_done * q_next_max  # (B, W)

                    loss = torch.nn.functional.mse_loss(q_s_a, target)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=5.0)
                    optimizer.step()
                    
                    # Single CPU transfer for logging
                    epoch_loss += loss.item()
                    c += 1

                if global_step % target_update_every == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            train_returns.append(img_return)
            epsilons.append(epsilon)
            
            # Compute metrics for this training image
            pred_train = reconstruct_image(policy_net, 
                                           image, 
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
        # print(train_metrics_history)
        # Validation
        if val_pairs:
            avg_val_metrics = validate(
                policy_net, val_pairs, device,
                continuity_coef=continuity_coef,
                continuity_decay_factor=continuity_decay_factor
            )
            
            # Compute validation loss using batched approach
            policy_net.eval()
            val_epoch_loss = 0
            val_c = 0
            
            with torch.no_grad():
                for image, mask in val_pairs:
                    env = PathReconstructionEnv(image=image,
                                                mask=mask,
                                                continuity_coef=continuity_coef,
                                                continuity_decay_factor=continuity_decay_factor,
                                                history_len=5)
                    obs, _ = env.reset()
                    done = False
                    
                    while not done:
                        obs_tensor = obs_to_tensor(obs, device)
                        q = policy_net(obs_tensor)
                        a = q.argmax(dim=-1)
                        
                        next_obs, reward, terminated, truncated, info = env.step(a.cpu().numpy())
                        pixel_rewards = info["pixel_rewards"]
                        done = terminated or truncated
                        
                        # Compute validation loss (TD error)
                        # Batch the single observation properly
                        obs_batch = batch_obs_to_tensor([obs], device)
                        next_obs_batch = batch_obs_to_tensor([next_obs], device)
                        a_batch = a.unsqueeze(0)  # (1, W)
                        rew_tensor = torch.tensor(pixel_rewards, dtype=torch.float32, device=device).unsqueeze(0)  # (1, W)
                        done_tensor = torch.tensor([done], dtype=torch.float32, device=device)  # (1,)
                        
                        q_s = policy_net(obs_batch)  # (1, W, 2)
                        q_s_a = q_s.gather(dim=-1, index=a_batch.unsqueeze(-1)).squeeze(-1)  # (1, W)
                        q_next = target_net(next_obs_batch)  # (1, W, 2)
                        q_next_max = q_next.max(dim=-1).values  # (1, W)
                        
                        not_done = (1.0 - done_tensor).unsqueeze(-1)  # (1, 1)
                        target = rew_tensor + gamma * not_done * q_next_max  # (1, W)
                        
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
        torch.save(target_net.state_dict(), 'dqn_row_based/models/model_cont.pth')
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs} | Time: {dt:.1f}s")
        print(f"  Train - Loss: {losses[-1]:.4f} | Avg Return: {np.mean(train_returns[-len(image_mask_pairs):]):.2f} | ε: {epsilon:.3f}")
        print(f"          IoU: {avg_train_metrics['iou']:.3f} | F1: {avg_train_metrics['f1']:.3f} | Acc: {avg_train_metrics['accuracy']:.3f} | Cov: {avg_train_metrics['coverage']:.3f} | Prec: {avg_train_metrics['precision']:.3f} | Rec: {avg_train_metrics['recall']:.3f}")
        if val_pairs:
            print(f"  Val   - Loss: {val_losses[-1]:.4f} | Avg Return: {val_returns[-1]:.2f}")
            print(f"          IoU: {avg_val_metrics['iou']:.3f} | F1: {avg_val_metrics['f1']:.3f} | Acc: {avg_val_metrics['accuracy']:.3f} | Cov: {avg_val_metrics['coverage']:.3f} | Prec: {avg_val_metrics['precision']:.3f} | Rec: {avg_val_metrics['recall']:.3f}")

    return {
        "policy_net": policy_net,
        "target_net": target_net,
        "returns": train_returns,
        "val_returns": val_returns,
        "epsilons": epsilons,
        "losses": losses,
        "val_losses": val_losses,
        "train_metrics": train_metrics_history,
        "val_metrics": val_metrics_history,
    }

def reconstruct_image(policy_net, 
                      image, 
                      mask, 
                      continuity_coef=0.2, 
                      continuity_decay_factor=0.7,
                      device = None):
    env = PathReconstructionEnv(image, 
                                mask, 
                                continuity_coef=continuity_coef, 
                                continuity_decay_factor=continuity_decay_factor,
                                history_len=5, 
                                start_from_bottom=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    obs, _ = env.reset()

    pred = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    done = False
    policy_net.eval()
    with torch.no_grad():
        while not done:
            x = obs_to_tensor(obs, device)
            q = policy_net(x)  # (W, 2)
            a = q.argmax(dim=-1).cpu().numpy()  # (W,)
            next_obs, reward, terminated, truncated, info = env.step(a)
            # Map current row index to image coordinate
            row = info.get("row_index", None)
            if row is not None:
                pred[row] = a.astype(np.uint8)
            obs = next_obs
            done = terminated or truncated
    policy_net.train()
    return pred

def visualize_result(img, mask, pred, save_dir: str = None) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    axs[0].imshow(img, cmap="gray")
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(mask, cmap="gray")
    axs[1].set_title("Mask")
    axs[1].axis("off")

    axs[2].imshow(pred, cmap="gray")
    axs[2].set_title("Reconstruction")
    axs[2].axis("off")

    if save_dir:
        plt.savefig(os.path.join(save_dir, "reconstruction.png"))
    plt.show()


if __name__ == "__main__":
    if USE_ARTIFACTS: 
        intermediate_dir = "with_artifacts"
    else:
        # intermediate_dir = "noise_only"
        intermediate_dir = "ct_like/2d"
        
    train_data_dir = os.path.join("data", intermediate_dir, "train")
    val_data_dir = os.path.join("data", intermediate_dir, "val")
    
    # Load training data
    train_imgs = []
    train_masks = []
    for file in sorted(os.listdir(train_data_dir)):
        if file.endswith(".npy"):
            continue
        obj = cv2.imread(os.path.join(train_data_dir, file), cv2.IMREAD_GRAYSCALE)
        if "mask" in file and file.endswith(".png"):
            train_masks.append(obj)
        else:
            train_imgs.append(obj)
    
    # Load validation data
    val_imgs = []
    val_masks = []
    for file in sorted(os.listdir(val_data_dir)):
        if file.endswith(".npy"):
            continue
        obj = cv2.imread(os.path.join(val_data_dir, file), cv2.IMREAD_GRAYSCALE)
        if "mask" in file and file.endswith(".png"):
            val_masks.append(obj)
        else:
            val_imgs.append(obj)
    
    results = train_dqn_on_images(
        list(zip(train_imgs, train_masks)),
        val_pairs=list(zip(val_imgs, val_masks)),
        num_epochs=20,
        continuity_coef=0.1,
        continuity_decay_factor=0.5,
        seed=123,
        start_epsilon=1.0,
        end_epsilon=0.01,
        epsilon_decay_epochs=15,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    img_test = cv2.imread(os.path.join(val_data_dir, "bush_root_var1_ct.png"), cv2.IMREAD_GRAYSCALE)
    mask_test = cv2.imread(os.path.join(val_data_dir, "bush_root_var1_mask.png"), cv2.IMREAD_GRAYSCALE)

    pred = reconstruct_image(results["policy_net"], img_test, mask_test)

    print(np.unique(pred))
    visualize_result(img_test, mask_test, pred, save_dir="dqn_row_based")

    # Plot training curves with validation
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    
    # Returns (Train vs Val)
    axes[0, 0].plot(results["returns"], alpha=0.3, color='blue', label='Train (per image)')
    # Moving average for train returns
    window = len(train_imgs)
    train_returns_ma = [np.mean(results["returns"][max(0, i-window+1):i+1]) 
                        for i in range(len(results["returns"]))]
    axes[0, 0].plot(train_returns_ma, color='blue', linewidth=2, label='Train (epoch avg)')
    if results["val_returns"]:
        axes[0, 0].plot(np.arange(len(results["val_returns"])) * len(train_imgs), 
                        results["val_returns"], color='red', marker='o', linewidth=2, label='Val')
    axes[0, 0].set_title("Episode Returns")
    axes[0, 0].set_ylabel("Return")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Epsilon
    axes[0, 1].plot(results["epsilons"], color='orange', linestyle='dashed')
    axes[0, 1].set_title("Exploration (Epsilon)")
    axes[0, 1].set_ylabel("ε")
    axes[0, 1].grid(True)
    
    # Loss (Train vs Val)
    axes[0, 2].plot(results["losses"], color='red', marker='o', label='Train')
    if results["val_losses"]:
        axes[0, 2].plot(results["val_losses"], color='darkred', marker='s', label='Val')
    axes[0, 2].set_title("Training Loss")
    axes[0, 2].set_ylabel("MSE Loss")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # IoU
    axes[1, 0].plot(results["train_metrics"]["iou"], label="Train", marker='o')
    if results["val_metrics"]["iou"]:
        axes[1, 0].plot(results["val_metrics"]["iou"], label="Val", marker='s')
    axes[1, 0].set_title("IoU")
    axes[1, 0].set_ylabel("IoU")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # F1 Score
    axes[1, 1].plot(results["train_metrics"]["f1"], label="Train", marker='o')
    if results["val_metrics"]["f1"]:
        axes[1, 1].plot(results["val_metrics"]["f1"], label="Val", marker='s')
    axes[1, 1].set_title("F1 Score")
    axes[1, 1].set_ylabel("F1")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Accuracy
    axes[1, 2].plot(results["train_metrics"]["accuracy"], label="Train", marker='o')
    if results["val_metrics"]["accuracy"]:
        axes[1, 2].plot(results["val_metrics"]["accuracy"], label="Val", marker='s')
    axes[1, 2].set_title("Pixel Accuracy")
    axes[1, 2].set_ylabel("Accuracy")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    # Coverage
    axes[2, 0].plot(results["train_metrics"]["coverage"], label="Train", marker='o')
    if results["val_metrics"]["coverage"]:
        axes[2, 0].plot(results["val_metrics"]["coverage"], label="Val", marker='s')
    axes[2, 0].set_title("Coverage")
    axes[2, 0].set_ylabel("Coverage")
    axes[2, 0].set_xlabel("Epoch")
    axes[2, 0].legend()
    axes[2, 0].grid(True)
    
    # Precision
    if "precision" in results["train_metrics"]:
        axes[2, 1].plot([m for m in results["train_metrics"].get("precision", [])], 
                        label="Train", marker='o')
        if "precision" in results["val_metrics"]:
            axes[2, 1].plot([m for m in results["val_metrics"].get("precision", [])], 
                           label="Val", marker='s')
        axes[2, 1].set_title("Precision")
        axes[2, 1].set_ylabel("Precision")
        axes[2, 1].set_xlabel("Epoch")
        axes[2, 1].legend()
        axes[2, 1].grid(True)
    
    # Recall
    if "recall" in results["train_metrics"]:
        axes[2, 2].plot([m for m in results["train_metrics"].get("recall", [])], 
                        label="Train", marker='o')
        if "recall" in results["val_metrics"]:
            axes[2, 2].plot([m for m in results["val_metrics"].get("recall", [])], 
                           label="Val", marker='s')
        axes[2, 2].set_title("Recall")
        axes[2, 2].set_ylabel("Recall")
        axes[2, 2].set_xlabel("Epoch")
        axes[2, 2].legend()
        axes[2, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{os.getcwd()}/dqn_row_based/results.png", dpi=300)
    plt.show()