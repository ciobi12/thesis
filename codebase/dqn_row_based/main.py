import time
import numpy as np
import torch
import random
import torch.optim as optim
import torch.nn as nn

from dqn_row_based.dqn import PerPixelDQN, ReplayBuffer, obs_to_tensor, epsilon_greedy_action
from dqn_row_based.env import PathReconstructionEnv

from l_systems import LSystemGenerator
from matplotlib import pyplot as plt
import os

def train_dqn_on_images(
    images,          # list of binary numpy arrays (H, W)
    num_epochs=20,
    steps_per_image=None,  # default: number of rows per image
    buffer_capacity=10000,
    batch_size=64,
    gamma=0.95,
    lr=1e-3,
    target_update_every=500,  # steps
    start_epsilon=1.0,
    end_epsilon=0.05,
    epsilon_decay_steps=5000,
    continuity_coef=0.1,
    seed=42,
    device=None,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Assume all images share width W
    Ws = {img.shape[1] for img in images}
    assert len(Ws) == 1, "All images must have the same width"
    W = next(iter(Ws))

    # Networks
    policy_net = PerPixelDQN(W=W, hidden=256).to(device)
    target_net = PerPixelDQN(W=W, hidden=256).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay = ReplayBuffer(buffer_capacity)

    # Epsilon schedule
    epsilon = start_epsilon
    epsilon_decay = (start_epsilon - end_epsilon) / max(1, epsilon_decay_steps)

    global_step = 0
    losses = []
    returns = []

    for epoch in range(num_epochs):
        random.shuffle(images)
        epoch_return = 0.0
        t0 = time.time()

        for img in images:
            env = PathReconstructionEnv(img, continuity_coef=continuity_coef, start_from_bottom=True)
            obs, _ = env.reset()

            done = False
            img_return = 0.0
            episode_steps = 0
            max_steps = steps_per_image or env.H

            while not done and episode_steps < max_steps:
                # Forward
                x = obs_to_tensor(obs, device).unsqueeze(0)  # (1, 2W+1)
                q = policy_net(x).squeeze(0)  # (W, 2)
                a = epsilon_greedy_action(q, epsilon)  # (W,)

                # Step environment
                next_obs, reward, terminated, truncated, info = env.step(a.detach().cpu().numpy())
                done = terminated or truncated
                pixel_rewards = info["pixel_rewards"]  # (W,)
                img_return += reward

                # Store transition
                replay.push(
                    obs,                               # dict
                    a.detach().cpu().numpy(),          # (W,)
                    pixel_rewards.astype(np.float32),  # (W,)
                    next_obs,                          # dict
                    done
                )

                obs = next_obs
                episode_steps += 1
                global_step += 1

                # Epsilon decay
                epsilon = max(end_epsilon, epsilon - epsilon_decay)

                # Optimize
                if len(replay) >= batch_size:
                    batch = replay.sample(batch_size)

                    # Build tensors
                    obs_batch = torch.stack([obs_to_tensor(t.obs, device) for t in batch], dim=0)  # (B, 2W+1)
                    act_batch = torch.from_numpy(np.stack([t.action for t in batch])).to(device)    # (B, W)
                    rew_batch = torch.from_numpy(np.stack([t.pixel_rewards for t in batch])).to(device)  # (B, W)
                    next_obs_batch = torch.stack([obs_to_tensor(t.next_obs, device) for t in batch], dim=0)  # (B, 2W+1)
                    done_batch = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device)   # (B,)

                    # Q(s,a): gather per pixel
                    q_s = policy_net(obs_batch)  # (B, W, 2)
                    q_s_a = q_s.gather(dim=-1, index=act_batch.unsqueeze(-1)).squeeze(-1)  # (B, W)

                    # Max_a' Q(s', a')
                    with torch.no_grad():
                        q_next = target_net(next_obs_batch)  # (B, W, 2)
                        q_next_max = q_next.max(dim=-1).values  # (B, W)
                        # Broadcast (1 - done) across pixels
                        not_done = (1.0 - done_batch).unsqueeze(-1)  # (B, 1)
                        target = rew_batch + gamma * not_done * q_next_max  # (B, W)

                    loss = nn.MSELoss()(q_s_a, target)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=5.0)
                    optimizer.step()
                    losses.append(loss.item())

                # Target network update
                if global_step % target_update_every == 0:
                    print(f"Updating target network at step {global_step}")
                    target_net.load_state_dict(policy_net.state_dict())

            returns.append(img_return)
        dt = time.time() - t0
        print(f"Epoch {epoch+1}/{num_epochs} | avg return: {np.mean(returns[-len(images):]):.2f} | epsilon: {epsilon:.3f} | {dt:.1f}s ")

    return {
        "policy_net": policy_net,
        "target_net": target_net,
        "returns": returns,
        "losses": losses,
    }

def reconstruct_image(policy_net, image, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    env = PathReconstructionEnv(image, continuity_coef=0.1, start_from_bottom=True)
    obs, _ = env.reset()

    pred = np.zeros_like(image, dtype=np.uint8)
    row_ptr = 0  # env returns rows bottom-to-top

    done = False
    while not done:
        x = obs_to_tensor(obs, device).unsqueeze(0)
        with torch.no_grad():
            q = policy_net(x).squeeze(0)  # (W, 2)
            a = q.argmax(dim=-1).cpu().numpy()  # (W,)
        next_obs, reward, terminated, truncated, info = env.step(a)
        # Map current row index to image coordinate
        row = info.get("row_index", None)
        if row is not None:
            pred[row] = a.astype(np.uint8)
        obs = next_obs
        done = terminated or truncated
    return pred

def visualize_result(mask, pred, save_dir: str = None) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(8,4))
    axs[0].imshow(mask, cmap="gray")
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(pred, cmap="gray")
    axs[1].set_title("Reconustrucion")
    axs[1].axis("off")

    if save_dir:
        plt.savefig(os.path.join(save_dir, "res.png"))
    plt.show()


if __name__ == "__main__":
    lsys_obj = LSystemGenerator(axiom = "X",
                                rules = {"X": "F+[[X]-X]-F[-FX]+X",
                                         "F": "FF"
                                         }
                                )
    
    iterations = 2
    angle = 22.5
    step = 5
    segments = lsys_obj.build_l_sys(iterations = iterations, step = step, angle_deg = angle)
    # lsys_obj.draw_lsystem()
    mask = lsys_obj.build_mask(canvas_size=(64, 128))
    
    results = train_dqn_on_images(
        [mask],
        num_epochs=20,
        continuity_coef=0.1,
        seed=123,
        start_epsilon=0.2
    )
    pred = reconstruct_image(results["policy_net"], mask)
    visualize_result(mask, pred, save_dir=None)

    iterations = 4
    angle = 22.5
    step = 5
    segments = lsys_obj.build_l_sys(iterations = 4, step = step, angle_deg = angle)
    # lsys_obj.draw_lsystem()
    mask_new = lsys_obj.build_mask(canvas_size=(64, 64))

    pred = reconstruct_image(results["policy_net"], mask_new)

    visualize_result(mask_new, pred, save_dir=None)
    plt.plot(results["losses"])
    plt.title("Training Loss")
    plt.xlabel("Optimization Step")
    plt.ylabel("L1 loss")
    plt.grid()
    plt.show()