import cv2
import time
import numpy as np
import torch
import random
import torch.optim as optim
import torch.nn as nn

from dqn_row_based.dqn import PerPixelCNN, ReplayBuffer, obs_to_tensor, epsilon_greedy_action
from dqn_row_based.env import PathReconstructionEnv

from l_systems import LSystemGenerator
from matplotlib import pyplot as plt
import os

USE_ARTIFACTS = False

def train_dqn_on_images(
    image_mask_pairs,      # list of (image, mask) tuples
    num_epochs=20,
    buffer_capacity=10000,
    batch_size=64,
    gamma=0.95,
    lr=1e-3,
    target_update_every=500,
    start_epsilon=1.0,
    end_epsilon=0.01,
    epsilon_decay_steps=5000,
    continuity_coef=0.1,
    continuity_decay_factor=0.7,
    seed=42,
    device=None,
):
    import time
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
    epsilon_decay = (start_epsilon - end_epsilon) / max(1, epsilon_decay_steps)

    global_step = 0
    losses = []
    returns = []
    epsilons = []

    for epoch in range(num_epochs):
        random.shuffle(image_mask_pairs)
        epoch_return = 0.0
        t0 = time.time()

        epoch_loss = 0
        c = 0

        for image, mask in image_mask_pairs:
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
                q = policy_net(obs_tensor)  # (W, 2)
                a = epsilon_greedy_action(q, epsilon)  # (W,)

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
                epsilon = max(end_epsilon, epsilon - epsilon_decay)

                # Optimize
                if len(replay) >= batch_size:
                    batch = replay.sample(batch_size)

                    obs_batch = [obs_to_tensor(t.obs, device) for t in batch]
                    act_batch = torch.tensor(np.stack([t.action for t in batch]), dtype=torch.int64, device=device)
                    rew_batch = torch.tensor(np.stack([t.pixel_rewards for t in batch]), dtype=torch.float32, device=device)
                    next_obs_batch = [obs_to_tensor(t.next_obs, device) for t in batch]
                    done_batch = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device)

                    q_s_a = []
                    q_next_max = []

                    for i in range(batch_size):
                        q_s = policy_net(obs_batch[i])  # (W, 2)
                        q_s_a.append(q_s.gather(dim=-1, index=act_batch[i].unsqueeze(-1)).squeeze(-1))  # (W,)

                        with torch.no_grad():
                            q_next = target_net(next_obs_batch[i])  # (W, 2)
                            q_next_max.append(q_next.max(dim=-1).values)  # (W,)

                    q_s_a = torch.stack(q_s_a)  # (B, W)
                    q_next_max = torch.stack(q_next_max)  # (B, W)
                    not_done = (1.0 - done_batch).unsqueeze(-1)  # (B, 1)
                    target = rew_batch + gamma * not_done * q_next_max  # (B, W)

                    loss = torch.nn.functional.mse_loss(q_s_a, target)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=5.0)
                    optimizer.step()
                    epoch_loss += loss.detach().cpu().numpy().item() if device == "cuda" else loss.item()
                    c += 1

                if global_step % target_update_every == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            returns.append(img_return)
            epsilons.append(epsilon)
            
        losses.append(epoch_loss / c)
        dt = time.time() - t0
        torch.save(target_net.state_dict(), 'dqn_row_based/models/model_cont.pth')
        print(f"Epoch {epoch+1}/{num_epochs} | avg loss: {losses[-1]:.3f} | avg return: {np.mean(returns[-len(image_mask_pairs):]):.2f} | epsilon: {epsilon:.3f} | {dt:.1f}s")

    return {
        "policy_net": policy_net,
        "target_net": target_net,
        "returns": returns,
        "epsilons": epsilons,
        "losses": losses,
    }

def reconstruct_image(policy_net, image, mask):
    device = "cpu"
    env = PathReconstructionEnv(image, 
                                mask, 
                                continuity_coef=0.2, 
                                continuity_decay_factor=0.7,
                                history_len=5, 
                                start_from_bottom=True)
    obs, _ = env.reset()

    pred = np.random.randn(image.shape[0], image.shape[1])
    row_ptr = 0  # env returns rows bottom-to-top

    done = False
    while not done:
        x = obs_to_tensor(obs, device)
        with torch.no_grad():
            q = policy_net(x)  # (W, 2)
            a = q.argmax(dim=-1).cpu().numpy()  # (W,)
        next_obs, reward, terminated, truncated, info = env.step(a)
        # Map current row index to image coordinate
        row = info.get("row_index", None)
        if row is not None:
            pred[row] = a.astype(np.uint8)
        obs = next_obs
        done = terminated or truncated
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
    axs[2].set_title("Reconustrucion")
    axs[2].axis("off")

    if save_dir:
        plt.savefig(os.path.join(save_dir, "reconstruction.png"))
    plt.show()


if __name__ == "__main__":
    if USE_ARTIFACTS:
        intermediate_dir = "with_artifacts"
    else:
        intermediate_dir = "noise_only"
        
    train_data_dir = os.path.join("data", intermediate_dir, "train")
    val_data_dir = os.path.join("data", intermediate_dir, "val")
    train_imgs = []
    train_masks = []
    for file in sorted(os.listdir(train_data_dir)):
        obj = cv2.imread(os.path.join(train_data_dir, file), cv2.IMREAD_GRAYSCALE)
        if "mask" in file:
            train_masks.append(obj)
        else:
            train_imgs.append(obj)
    
    results = train_dqn_on_images(
        list(zip(train_imgs, train_masks)),
        num_epochs = 10,
        continuity_coef = 0.2,
        continuity_decay_factor = 0.7,
        seed = 123,
        start_epsilon = 1,
        end_epsilon=0.01,
        epsilon_decay_steps=10e3, 
        device = "cpu"
    )

    img_test = cv2.imread(os.path.join(val_data_dir, "tree_1.png"), cv2.IMREAD_GRAYSCALE)
    mask_test = cv2.imread(os.path.join(val_data_dir, "tree_1_mask.png"), cv2.IMREAD_GRAYSCALE)

    pred = reconstruct_image(results["policy_net"], img_test, mask_test)
    # pred = reconstruct_image(results["policy_net"], train_imgs[0], train_masks[0])

    print(np.unique(pred))
    visualize_result(img_test, mask_test, pred, save_dir = "dqn_row_based")
    # visualize_result(train_imgs[0], train_masks[0], pred, save_dir = "dqn_row_based")

    plt.subplot(3,1,1)
    plt.plot(results["returns"])
    plt.title("Rewards")
    plt.grid(True)

    plt.subplot(3,1,2)
    plt.plot(results["epsilons"], color='orange', linestyle='dashed')
    plt.title("Epsilon")
    plt.grid(True)

    plt.subplot(3,1,3)
    plt.plot(results["losses"], color='red')
    plt.title("Training loss")
    plt.ylabel("L1 loss")
    plt.xlabel("Epoch")
    plt.grid(True)

    plt.savefig(f"{os.getcwd()}/dqn_row_based/results.png", dpi=1200)
    plt.show()