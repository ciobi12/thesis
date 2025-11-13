import os
import cv2
import time
import numpy as np
import random
import torch
import torch.nn.functional as F

from dqn_patch_based.env import PatchClassificationEnv
from dqn_patch_based.dqn import PerPatchCNN, ReplayBuffer, obs_to_tensor, epsilon_greedy_action

from matplotlib import pyplot as plt


def train_dqn_patch(
    image_mask_pairs,
    patch_size=16,
    num_epochs=10,
    buffer_capacity=10000,
    batch_size=64,
    gamma=0.95,
    lr=1e-3,
    target_update_every=500,
    start_epsilon=1.0,
    end_epsilon=0.05,
    epsilon_decay_steps=5000,
    continuity_coef=0.0,
    neighbor_coef=0.1,
    seed=42,
    device=None,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    sample_image, _ = image_mask_pairs[0]
    C = 1 if sample_image.ndim == 2 else sample_image.shape[2]

    policy_net = PerPatchCNN(N=patch_size, C=C).to(device)
    target_net = PerPatchCNN(N=patch_size, C=C).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    replay = ReplayBuffer(buffer_capacity)

    epsilon = start_epsilon
    epsilon_decay = (start_epsilon - end_epsilon) / max(1, epsilon_decay_steps)

    global_step = 0
    returns, losses, epsilons = [], [], []

    for epoch in range(num_epochs):
        random.shuffle(image_mask_pairs)
        t0 = time.time()
        epoch_return = 0.0
        epoch_loss = 0.0
        c_updates = 0

        for image, mask in image_mask_pairs:
            env = PatchClassificationEnv(
                image=image,
                mask=mask,
                patch_size=patch_size,
                continuity_coef=continuity_coef,
                neighbor_coef=neighbor_coef,
                start_from_bottom_left=True,
            )
            obs, _ = env.reset()
            done = False
            img_return = 0.0

            while not done:
                xt = obs_to_tensor(obs, device)
                q = policy_net(xt)  # (2, N, N)
                a = epsilon_greedy_action(q, epsilon)  # (N, N)

                next_obs, reward, terminated, truncated, info = env.step(a.detach().cpu().numpy())
                done = terminated or truncated
                img_return += reward

                pixel_rewards = info["pixel_rewards"]  # (N, N)

                replay.push(obs, a.detach().cpu().numpy(), pixel_rewards.astype(np.float32), next_obs, done)
                obs = next_obs

                global_step += 1
                epsilon = max(end_epsilon, epsilon - epsilon_decay)

                if len(replay) >= batch_size:
                    batch = replay.sample(batch_size)

                    q_s_a_list = []
                    q_next_max_list = []
                    rew_list = []
                    done_list = []

                    for t in batch:
                        o = obs_to_tensor(t.obs, device)
                        q_s = policy_net(o)  # (2, N, N)
                        # gather Q for chosen actions
                        a_idx = torch.tensor(t.action, dtype=torch.int64, device=device)  # (N, N)
                        q_s_a = q_s.permute(1,2,0).gather(dim=-1, index=a_idx.unsqueeze(-1)).squeeze(-1)  # (N, N)

                        with torch.no_grad():
                            o2 = obs_to_tensor(t.next_obs, device)
                            q_next = target_net(o2)  # (2, N, N)
                            q_next_max = q_next.max(dim=0).values  # (N, N)

                        q_s_a_list.append(q_s_a)
                        q_next_max_list.append(q_next_max)
                        rew_list.append(torch.tensor(t.pixel_rewards, dtype=torch.float32, device=device))
                        done_list.append(torch.tensor(1.0 if t.done else 0.0, dtype=torch.float32, device=device))

                    q_s_a = torch.stack(q_s_a_list)           # (B, N, N)
                    q_next_max = torch.stack(q_next_max_list) # (B, N, N)
                    rew = torch.stack(rew_list)               # (B, N, N)
                    done_b = torch.stack(done_list).view(-1, 1, 1)  # (B,1,1)
                    target = rew + gamma * (1.0 - done_b) * q_next_max

                    loss = F.mse_loss(q_s_a, target)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 5.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    c_updates += 1

                if global_step % target_update_every == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            returns.append(img_return)
            epsilons.append(epsilon)

        losses.append(epoch_loss / max(1, c_updates))
        dt = time.time() - t0
        print(f"Epoch {epoch+1}/{num_epochs} | avg loss: {losses[-1]:.3f} | avg return: {np.mean(returns[-len(image_mask_pairs):]):.2f} | eps: {epsilon:.3f} | {dt:.1f}s")

    return {
        "policy_net": policy_net,
        "target_net": target_net,
        "returns": returns,
        "epsilons": epsilons,
        "losses": losses,
    }


def reconstruct_image(policy_net, image, mask, patch_size=16, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    env = PatchClassificationEnv(image=image, mask=mask, patch_size=patch_size)
    obs, _ = env.reset()
    canvas = env.reconstruct_blank()

    done = False
    while not done:
        xt = obs_to_tensor(obs, device)
        with torch.no_grad():
            q = policy_net(xt)  # (2, N, N)
            a = q.argmax(dim=0).cpu().numpy().astype(np.uint8)  # (N, N)
        # write into canvas
        y0, x0 = env._order[env._idx]
        canvas[y0:y0+env.N, x0:x0+env.N] = a
        obs, _, terminated, truncated, _ = env.step(a)
        done = terminated or truncated

    return env.crop_to_original(canvas)


def visualize_result(img, mask, pred, save_dir: str = None) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    axs[0].imshow(img, cmap="gray")
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(mask, cmap="gray")
    axs[1].set_title("Mask")
    axs[1].axis("off")

    axs[2].imshow(pred, cmap="gray")
    axs[2].set_title("Patch-based Reconstruction")
    axs[2].axis("off")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "reconstruction.png"))
    plt.show()


if __name__ == "__main__":
    train_data_dir = os.path.join("data", "train")
    val_data_dir = os.path.join("data", "val")
    train_imgs, train_masks = [], []

    for file in sorted(os.listdir(train_data_dir)):
        obj = cv2.imread(os.path.join(train_data_dir, file), cv2.IMREAD_GRAYSCALE)
        if obj is None:
            continue
        if "mask" in file:
            train_masks.append(obj)
        else:
            train_imgs.append(obj)

    pairs = list(zip(train_imgs, train_masks))
    results = train_dqn_patch(
        pairs,
        patch_size=16,
        num_epochs=5,
        continuity_coef=0.0,
        neighbor_coef=0.0,
        start_epsilon=0.5,
        end_epsilon=0.01,
        epsilon_decay_steps=10000,
        device = "cpu"
    )

    img_test = cv2.imread(os.path.join(val_data_dir, "tree_1.png"), cv2.IMREAD_GRAYSCALE)
    mask_test = cv2.imread(os.path.join(val_data_dir, "tree_1_mask.png"), cv2.IMREAD_GRAYSCALE)

    pred = reconstruct_image(results["policy_net"], img_test, mask_test, patch_size=16, device = "cpu")

    visualize_result(img_test, mask_test, pred, save_dir="dqn_patch_based")

    # Training curves
    plt.figure(figsize=(10,8))
    plt.subplot(3,1,1); plt.plot(results["returns"]); plt.title("Rewards"); plt.grid(True)
    plt.subplot(3,1,2); plt.plot(results["epsilons"], color='orange', linestyle='dashed'); plt.title("Epsilon"); plt.grid(True)
    plt.subplot(3,1,3); plt.plot(results["losses"], color='red'); plt.title("Training loss"); plt.ylabel("MSE loss"); plt.xlabel("Epoch"); plt.grid(True)
    os.makedirs("dqn_patch_based", exist_ok=True)
    plt.savefig(os.path.join("dqn_patch_based", "results.png"), dpi=600)
    plt.show()
