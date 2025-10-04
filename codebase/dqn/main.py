from dqn.env import PathTraversalEnv
from dqn.q_network import PatchDQN
from dqn.buffer import ReplayBuffer
from l_systems import LSystemGenerator

import cv2
import numpy as np
import torch
import torch.nn as nn
import random
from PIL import Image
import os
from tqdm import tqdm


def train_dqn(env, lsys_iterations = 2, episodes=1000, patch_size=5, batch_size=64, gamma=0.99,
              lr=1e-3, target_update=10, eps_start=1.0, eps_end=0.05, eps_decay=0.995):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_actions = env.action_space.n

    policy_net = PatchDQN(patch_size, n_actions).to(device)
    target_net = PatchDQN(patch_size, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer()

    epsilon = eps_start
    prev_dead_end = False
    rewards = []
    options = {"start_from_seed": True, "reset_global_mask": False}

    for ep in tqdm(range(episodes)):
        if prev_dead_end:
            options["start_from_seed"] = False
        state, _ = env.reset(options = options)
        done = False
        dead_end = False
        total_reward = 0

        while not (done or dead_end):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                    action = int(torch.argmax(q_values).item())

            next_state, reward, dead_end, terminated, truncated, info = env.step(action)
            done = terminated or truncated or dead_end

            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Training step
            if len(buffer) >= batch_size:
                s_batch, a_batch, r_batch, s_next_batch, d_batch = buffer.sample(batch_size)
                s_batch = s_batch.to(device).float() / 2.0
                s_next_batch = s_next_batch.to(device).float() / 2.0
                a_batch = a_batch.to(device)
                r_batch = r_batch.to(device)
                d_batch = d_batch.to(device)

                q_values = policy_net(s_batch).gather(1, a_batch.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    max_next_q = target_net(s_next_batch).max(1)[0]
                    target_q = r_batch + gamma * max_next_q * (1 - d_batch)

                loss = nn.functional.mse_loss(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        rewards.append(total_reward)
        epsilon = max(eps_end, epsilon * eps_decay)

        # Update target network
        if ep % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # if ep % 50 == 0:
        #     print(f"Episode {ep}: reward={total_reward:.1f}, coverage={info.get('total_coverage', 0):.3f}, ε={epsilon:.3f}")
        if env.render_mode == "rgb_array":
            frame = env.render()
            im = Image.fromarray(frame)
            im.save(f"{os.getcwd()}/dqn/episodes_results/lsys_{lsys_iterations}it/ep_{ep+1}_ep_coverage_{info.get('episode_coverage', 0)*100:.2f}_total_coverage_{info.get('total_coverage', 0)*100:.2f}.png")

    return policy_net, rewards

if __name__ == "__main__":
    lsys_obj = LSystemGenerator(axiom = "X",
                                rules = {"X": "F+[[X]-X]-F[-FX]+X",
                                         "F": "FF"
                                         }
                                )
    
    iterations = 2
    angle = 22.5
    step = 5

    os.makedirs(f"{os.getcwd()}/dqn/episodes_results/lsys_{iterations}it", exist_ok = True)

    segments = lsys_obj.build_l_sys(iterations = iterations, step = step, angle_deg = angle)
    # lsys_obj.draw_lsystem()
    mask = lsys_obj.build_mask(canvas_size=(256, 256))
    kernel = np.ones((3, 3), np.uint8)
    mask_dilate = cv2.dilate(mask, kernel, iterations = 1)
    mask_dilate = np.asarray(mask_dilate, dtype = np.uint8)

    env = PathTraversalEnv(path_mask = mask_dilate, 
                           patch_size = 5, 
                           target_coverage = 0.95, 
                           render_mode = "rgb_array")
    train_dqn(env, episodes = 100)