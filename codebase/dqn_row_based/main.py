import cv2

import numpy as np
import os
import random
import time
import torch
import torch.optim as optim
import torch.nn as nn

from dqn_row_based.dqn import PerPixelCNNWithHistory, ReplayBuffer
from dqn_row_based.env import PathReconstructionEnv

from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image

CT_LIKE = False
DRIVE_DATASET = True
STARE_DATASET = True

def obs_to_tensor(obs, device):
    """Convert observation dict to tensors on device."""
    row_pixels = torch.FloatTensor(obs["row_pixels"]).unsqueeze(0).to(device)  # (1, W, C)
    prev_preds = torch.FloatTensor(obs["prev_preds"]).unsqueeze(0).to(device)  # (1, history_len, W)
    future_rows = torch.FloatTensor(obs["future_rows"]).unsqueeze(0).to(device)  # (1, future_len, W, C)
    return row_pixels, prev_preds, future_rows

def batch_obs_to_tensor(obs_list, device):
    """Convert list of observation dicts to batched tensors."""
    row_pixels = torch.stack([torch.FloatTensor(o["row_pixels"]) for o in obs_list]).to(device)  # (B, W, C)
    prev_preds = torch.stack([torch.FloatTensor(o["prev_preds"]) for o in obs_list]).to(device)  # (B, history_len, W)
    future_rows = torch.stack([torch.FloatTensor(o["future_rows"]) for o in obs_list]).to(device)  # (B, future_len, W, C)
    return row_pixels, prev_preds, future_rows

def epsilon_greedy_action(q_values, epsilon):
    """Epsilon-greedy action selection.
    
    Args:
        q_values: (W, 2) tensor of Q-values
        epsilon: exploration probability
    
    Returns:
        action: (W,) tensor of binary actions
    """
    if np.random.rand() < epsilon:
        W = q_values.shape[0]
        return torch.randint(0, 2, (W,), device=q_values.device)
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
    
def update_dataset(data_dir, size = (256, 256)):
    for root, _, files in os.walk(data_dir):
        for file in sorted(files):
            file_path = os.path.join(root, file)
            
            # Skip the mask folder (used in other datasets)
            if os.sep + "mask" + os.sep in file_path:
                continue
            
            if "train" in root:
                if "segm" in root:  # Check if in ground_truth folder
                    print(f"Train mask: {file_path}")
                    train_masks_paths.append(file_path)
                    # img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    img = np.array(Image.open(file_path).convert("L").resize(size))
                    if img is not None:
                        train_masks.append(img)
                elif "images" in root:  # Check if in images folder
                    train_imgs_paths.append(file_path)
                    print(f"Train image: {file_path}")
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, size)
                    if img is not None:
                        train_imgs.append(img)
                    
            elif "val" in root:
                if "segm" in root:  # Check if in ground_truth folder
                    val_masks_paths.append(file_path)
                    print(f"Val mask: {file_path}")
                    # img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    img = np.array(Image.open(file_path).convert("L").resize(size))
                    if img is not None:
                        val_masks.append(img)
                elif "images" in root:  # Check if in images folder
                    val_imgs_paths.append(file_path)
                    print(f"Val image: {file_path}")
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, size)  
                    if img is not None:
                        val_imgs.append(img)

def validate(policy_net, val_pairs, device, continuity_coef=0.1, continuity_decay_factor=0.7, future_len=3):
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
                                     future_len=future_len,
                                     device=device)
            metrics = compute_metrics(pred, mask)
            all_metrics.append(metrics)
            
            # Compute validation reward by running through environment
            env = PathReconstructionEnv(image=image,
                                        mask=mask,
                                        continuity_coef=continuity_coef,
                                        continuity_decay_factor=continuity_decay_factor,
                                        history_len=5,
                                        future_len=future_len)
            obs, _ = env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                row_pixels, prev_preds, future_rows = obs_to_tensor(obs, device)
                q = policy_net(row_pixels, prev_preds, future_rows)
                a = q.argmax(dim=-1).cpu().numpy()[0]  # Remove batch dimension
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
    policy_net = PerPixelCNNWithHistory(
        input_channels=C,
        history_len=5,
        future_len=3,
        width=W
    ).to(device)
    
    target_net = PerPixelCNNWithHistory(
        input_channels=C,
        history_len=5,
        future_len=3,
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
    train_losses = []
    val_losses = []

    train_returns = []
    val_returns = []
    base_returns = []
    continuity_returns = []
    
    epsilons = []
    
    # Metrics tracking
    train_metrics_history = {
        "iou": [], "f1": [], "accuracy": [], "coverage": [], "precision": [], "recall": []
    }
    val_metrics_history = {
        "iou": [], "f1": [], "accuracy": [], "coverage": [], "precision": [], "recall": []
    }
    
    # Conn_info tracking
    conn_info_history = {
        "detection_reward": [],
        "thickness_penalty": [],
        "width_bonus": [],
        "confidence_penalty": [],
        "connectivity_reward": [],
        "true_positives": [],
        "false_positives": [],
        "false_negatives": []
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
                                        history_len=5,
                                        future_len=3,
                                        start_from_bottom=True)
            obs, _ = env.reset()

            done = False
            img_return = 0.0
            base_return = 0.0
            continuity_return = 0.0
            
            # Track conn_info for this episode
            episode_conn_info = {
                "detection_reward": [],
                "thickness_penalty": [],
                "width_bonus": [],
                "confidence_penalty": [],
                "connectivity_reward": [],
                "true_positives": [],
                "false_positives": [],
                "false_negatives": []
            }

            while not done:
                row_pixels, prev_preds, future_rows = obs_to_tensor(obs, device)
                with torch.no_grad():
                    q = policy_net(row_pixels, prev_preds, future_rows)  # (1, W, 2)
                    q = q.squeeze(0)  # (W, 2)
                a = epsilon_greedy_action(q, epsilon)  # (W,)

                # Keep action on GPU, only convert to numpy when needed for env
                next_obs, reward, terminated, truncated, info = env.step(a.cpu().numpy())
                pixel_rewards = info["pixel_rewards"]
                base_reward = info["weighted_accuracy"].sum()
                continuity_reward = info["continuity_rewards"].sum()
                
                # Track conn_info metrics
                for key in episode_conn_info.keys():
                    if key in info:
                        episode_conn_info[key].append(info[key])
                
                done = terminated or truncated
                base_return += base_reward
                continuity_return += continuity_reward
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
                    row_pixels_batch, prev_preds_batch, future_rows_batch = batch_obs_to_tensor([t.obs for t in batch], device)
                    act_batch = torch.tensor(np.stack([t.action for t in batch]), dtype=torch.int64, device=device)
                    rew_batch = torch.tensor(np.stack([t.pixel_rewards for t in batch]), dtype=torch.float32, device=device)
                    next_row_pixels_batch, next_prev_preds_batch, next_future_rows_batch = batch_obs_to_tensor([t.next_obs for t in batch], device)
                    done_batch = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device)

                    # Batched forward passes
                    q_s = policy_net(row_pixels_batch, prev_preds_batch, future_rows_batch)  # (B, W, 2)
                    q_s_a = q_s.gather(dim=-1, index=act_batch.unsqueeze(-1)).squeeze(-1)  # (B, W)

                    with torch.no_grad():
                        q_next = target_net(next_row_pixels_batch, next_prev_preds_batch, next_future_rows_batch)  # (B, W, 2)
                        q_next_max = q_next.max(dim=-1).values  # (B, W)

                    not_done = (1.0 - done_batch).unsqueeze(-1)  # (B, 1)
                    target = rew_batch + gamma * not_done * q_next_max  # (B, W)

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
            train_returns.append(img_return)
            epsilons.append(epsilon)
            
            # Aggregate conn_info for this episode (average across rows)
            for key in episode_conn_info.keys():
                if len(episode_conn_info[key]) > 0:
                    conn_info_history[key].append(np.mean(episode_conn_info[key]))
                else:
                    conn_info_history[key].append(0.0)
            
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
                for image, mask in val_pairs:
                    env = PathReconstructionEnv(image=image,
                                                mask=mask,
                                                continuity_coef=continuity_coef,
                                                continuity_decay_factor=continuity_decay_factor,
                                                history_len=5,
                                                future_len=3)
                    obs, _ = env.reset()
                    done = False
                    
                    while not done:
                        row_pixels, prev_preds, future_rows = obs_to_tensor(obs, device)
                        q = policy_net(row_pixels, prev_preds, future_rows)
                        a = q.argmax(dim=-1)
                        
                        next_obs, reward, terminated, truncated, info = env.step(a.cpu().numpy()[0])
                        pixel_rewards = info["pixel_rewards"]
                        done = terminated or truncated
                        
                        # Compute validation loss
                        next_row_pixels, next_prev_preds, next_future_rows = obs_to_tensor(next_obs, device)
                        rew_tensor = torch.tensor(pixel_rewards, dtype=torch.float32, device=device).unsqueeze(0)
                        done_tensor = torch.tensor([done], dtype=torch.float32, device=device)
                        
                        q_s_a = q.gather(dim=-1, index=a.unsqueeze(-1)).squeeze(-1)
                        q_next = target_net(next_row_pixels, next_prev_preds, next_future_rows)
                        q_next_max = q_next.max(dim=-1).values
                        
                        not_done = (1.0 - done_tensor).unsqueeze(-1)
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
        
        train_losses.append(epoch_loss / c if c > 0 else 0)
        dt = time.time() - t0
        if CT_LIKE:
            save_dir = "ct_like"
        elif DRIVE_DATASET:
            save_dir = "drive"
        elif DRIVE_DATASET and STARE_DATASET:
            save_dir = "drive+stare"
        torch.save(target_net.state_dict(), 'dqn_row_based/models/{}/model_cont.pth'.format(save_dir))
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs} | Time: {dt:.1f}s")
        print(f"  Train - Loss: {train_losses[-1]:.4f} | Avg Return: {np.mean(train_returns[-len(image_mask_pairs):]):.2f} | ε: {epsilon:.3f}")
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
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_metrics": train_metrics_history,
        "val_metrics": val_metrics_history,
        "conn_info": conn_info_history,
    }

def reconstruct_image(policy_net, 
                      image, 
                      mask, 
                      continuity_coef=0.2, 
                      continuity_decay_factor=0.7,
                      future_len=3,
                      device = None):
    env = PathReconstructionEnv(image, 
                                mask, 
                                continuity_coef=continuity_coef, 
                                continuity_decay_factor=continuity_decay_factor,
                                history_len=5,
                                future_len=future_len, 
                                start_from_bottom=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    obs, _ = env.reset()

    pred = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    done = False
    policy_net.eval()
    with torch.no_grad():
        while not done:
            row_pixels, prev_preds, future_rows = obs_to_tensor(obs, device)
            q = policy_net(row_pixels, prev_preds, future_rows)  # (1, W, 2)
            a = q.argmax(dim=-1).cpu().numpy()[0]  # Remove batch dim -> (W,)
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
    # plt.show()


if __name__ == "__main__":
    train_imgs = []
    train_masks = []
    
    val_imgs = []
    val_masks = []
    
    train_imgs_paths = []
    train_masks_paths = []
    
    val_imgs_paths = []
    val_masks_paths = []

    size = (384, 384)
    cont_coef = 0.1

    if CT_LIKE:
        update_dataset("../data/ct_like/2d/continuous", size=size)  
    if DRIVE_DATASET:
        update_dataset("../data/DRIVE", size=size)
    if STARE_DATASET:
        update_dataset("../data/STARE", size=size)      
                    
    print(train_imgs_paths[:2])
    print(len(train_imgs_paths))
    print(train_masks_paths[:2])
    print(len(train_masks_paths))
    results = train_dqn_on_images(
        list(zip(train_imgs, train_masks)),
        val_pairs=list(zip(val_imgs, val_masks)),
        num_epochs=30,
        continuity_coef=cont_coef,
        continuity_decay_factor=0.7,
        seed=123,
        start_epsilon=1.0,
        end_epsilon=0.01,
        epsilon_decay_epochs=15,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    img_test = cv2.imread(val_imgs_paths[0], cv2.IMREAD_GRAYSCALE)
    img_test = cv2.resize(img_test, size)
    mask_test = np.array(Image.open(val_masks_paths[0]).convert("L").resize(size))

    pred = reconstruct_image(results["policy_net"], 
                             img_test, 
                             mask_test, 
                             continuity_coef=cont_coef, 
                             continuity_decay_factor=0.7)

    print(np.unique(pred))
    visualize_result(img_test, mask_test, pred, save_dir="dqn_row_based/results")

    # Plot training curves
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Returns (Train vs Val)
    axes[0, 0].plot(results["returns"], alpha=0.3, color='blue', label='Train (per image)')
    # Moving average for train returns
    window = len(train_imgs)
    train_returns_ma = [np.mean(results["returns"][max(0, i-window+1):i+1]) 
                        for i in range(len(results["returns"]))]
    axes[0, 0].plot(train_returns_ma, color='blue', linewidth=2, label='Train (epoch avg)')
    # if results["val_returns"]:
    #     axes[0, 0].plot(np.arange(len(results["val_returns"])) * len(train_imgs), 
    #                     results["val_returns"], color='red', marker='o', linewidth=2, label='Val')
    axes[0, 0].set_title("Episode Returns")
    axes[0, 0].set_ylabel("Return")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Base reward
    axes[0, 1].plot(np.convolve(results["base_returns"], 
                                 np.ones(len(train_imgs))/len(train_imgs), 
                                 mode='valid'), 
                    label="Base Reward", color='green')
    axes[0, 1].set_title("Base Reward (Moving Average)")
    axes[0, 1].set_ylabel("Reward")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].grid(True)

    # Continuity reward
    axes[0, 2].plot(np.convolve(results["continuity_returns"], 
                                 np.ones(len(train_imgs))/len(train_imgs),
                                 mode='valid'), 
                    label="Continuity Reward", color='blue')
    axes[0, 2].set_title("Continuity Reward (Moving Average)")
    axes[0, 2].set_ylabel("Reward")
    axes[0, 2].set_xlabel("Episode")
    axes[0, 2].grid(True)
    
    # Epsilon
    axes[0, 3].plot(results["epsilons"], color='orange', linestyle='dashed')
    axes[0, 3].set_title("Exploration (Epsilon)")
    axes[0, 3].set_ylabel("ε")
    axes[0, 3].grid(True)
    
    # Loss (Train vs Val)
    axes[1, 0].plot(results["train_losses"], color='red', marker='o', label='Train')
    if results["val_losses"]:
        axes[1, 0].plot(results["val_losses"], color='darkred', marker='s', label='Val')
    axes[1, 0].set_title("Model loss per Epoch")
    axes[1, 0].set_ylabel("MSE Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # IoU
    axes[1, 1].plot(results["train_metrics"]["iou"], label="Train", marker='o')
    if results["val_metrics"]["iou"]:
        axes[1, 1].plot(results["val_metrics"]["iou"], label="Val", marker='s')
    axes[1, 1].set_title("IoU")
    axes[1, 1].set_ylabel("IoU")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # F1 Score
    axes[1, 2].plot(results["train_metrics"]["f1"], label="Train", marker='o')
    if results["val_metrics"]["f1"]:
        axes[1, 2].plot(results["val_metrics"]["f1"], label="Val", marker='s')
    axes[1, 2].set_title("F1 Score")
    axes[1, 2].set_ylabel("F1")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    # Accuracy
    axes[1, 3].plot(results["train_metrics"]["accuracy"], label="Train", marker='o')
    if results["val_metrics"]["accuracy"]:
        axes[1, 3].plot(results["val_metrics"]["accuracy"], label="Val", marker='s')
    axes[1, 3].set_title("Pixel Accuracy")
    axes[1, 3].set_ylabel("Accuracy")
    axes[1, 3].set_xlabel("Epoch")
    axes[1, 3].legend()
    axes[1, 3].grid(True)
    
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
    plt.savefig(f"{os.getcwd()}/dqn_row_based/results/results.png", dpi=300)
    # plt.show()
    
    # Plot conn_info metrics
    fig2, axes2 = plt.subplots(2, 4, figsize=(20, 10))
    
    # Moving average window
    window = len(train_imgs)
    
    # Detection reward
    axes2[0, 0].plot(np.convolve(results["conn_info"]["detection_reward"], 
                                  np.ones(window)/window, mode='valid'), 
                     color='green', label='Detection Reward')
    axes2[0, 0].set_title("Detection Reward (TP - FP - FN)")
    axes2[0, 0].set_ylabel("Reward")
    axes2[0, 0].set_xlabel("Episode")
    axes2[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes2[0, 0].grid(True)
    
    # Thickness penalty
    axes2[0, 1].plot(np.convolve(results["conn_info"]["thickness_penalty"], 
                                  np.ones(window)/window, mode='valid'), 
                     color='red', label='Thickness Penalty')
    axes2[0, 1].set_title("Thickness Penalty (Width Control)")
    axes2[0, 1].set_ylabel("Penalty")
    axes2[0, 1].set_xlabel("Episode")
    axes2[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes2[0, 1].grid(True)
    
    # Width bonus
    axes2[0, 2].plot(np.convolve(results["conn_info"]["width_bonus"], 
                                  np.ones(window)/window, mode='valid'), 
                     color='blue', label='Width Bonus')
    axes2[0, 2].set_title("Width Bonus (Matching GT Width)")
    axes2[0, 2].set_ylabel("Bonus")
    axes2[0, 2].set_xlabel("Episode")
    axes2[0, 2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes2[0, 2].grid(True)
    
    # Confidence penalty
    axes2[0, 3].plot(np.convolve(results["conn_info"]["confidence_penalty"], 
                                  np.ones(window)/window, mode='valid'), 
                     color='orange', label='Confidence Penalty')
    axes2[0, 3].set_title("Confidence Penalty (Uncertain Predictions)")
    axes2[0, 3].set_ylabel("Penalty")
    axes2[0, 3].set_xlabel("Episode")
    axes2[0, 3].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes2[0, 3].grid(True)
    
    # Connectivity reward
    axes2[1, 0].plot(np.convolve(results["conn_info"]["connectivity_reward"], 
                                  np.ones(window)/window, mode='valid'), 
                     color='purple', label='Connectivity Reward')
    axes2[1, 0].set_title("Connectivity Reward (Fragmentation)")
    axes2[1, 0].set_ylabel("Reward")
    axes2[1, 0].set_xlabel("Episode")
    axes2[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes2[1, 0].grid(True)
    
    # True positives
    axes2[1, 1].plot(np.convolve(results["conn_info"]["true_positives"], 
                                  np.ones(window)/window, mode='valid'), 
                     color='green', label='True Positives')
    axes2[1, 1].set_title("True Positives (Correct Vessel Pixels)")
    axes2[1, 1].set_ylabel("Count")
    axes2[1, 1].set_xlabel("Episode")
    axes2[1, 1].grid(True)
    
    # False positives vs False negatives
    axes2[1, 2].plot(np.convolve(results["conn_info"]["false_positives"], 
                                  np.ones(window)/window, mode='valid'), 
                     color='red', label='False Positives')
    axes2[1, 2].plot(np.convolve(results["conn_info"]["false_negatives"], 
                                  np.ones(window)/window, mode='valid'), 
                     color='darkorange', label='False Negatives')
    axes2[1, 2].set_title("False Positives vs False Negatives")
    axes2[1, 2].set_ylabel("Count")
    axes2[1, 2].set_xlabel("Episode")
    axes2[1, 2].legend()
    axes2[1, 2].grid(True)
    
    # FP/FN ratio (diagnostic)
    fp_arr = np.array(results["conn_info"]["false_positives"])
    fn_arr = np.array(results["conn_info"]["false_negatives"])
    ratio = np.divide(fp_arr, fn_arr + 1e-8)  # Avoid division by zero
    axes2[1, 3].plot(np.convolve(ratio, np.ones(window)/window, mode='valid'), 
                     color='brown', label='FP/FN Ratio')
    axes2[1, 3].set_title("FP/FN Ratio (Over-prediction Indicator)")
    axes2[1, 3].set_ylabel("Ratio")
    axes2[1, 3].set_xlabel("Episode")
    axes2[1, 3].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Balanced')
    axes2[1, 3].legend()
    axes2[1, 3].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{os.getcwd()}/dqn_row_based/results/conn_info_analysis.png", dpi=300)
    # plt.show()

    