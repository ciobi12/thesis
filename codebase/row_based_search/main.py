import numpy as np 
import os
import matplotlib.pyplot as plt
import random

from row_based_search.env import PixelPathEnv
from l_systems import LSystemGenerator

def visualize_result(env, save_dir: str = None, ep: int = 0) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(8,4))
    axs[0].imshow(env.image, cmap="gray")
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(env.output, cmap="gray")
    axs[1].set_title("Agent's Output")
    axs[1].axis("off")

    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{ep}.png"))
        

def get_Q(Q, state):
    if state not in Q.keys():
        Q[state] = [0.0, 0.0]  # two actions: 0 or 1
    return Q[state]

# # Example toy image
# image = np.random.choice([0, 255], size=(16, 16))
# env = PixelPathEnv(image)

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
mask = lsys_obj.build_mask(canvas_size=(100, 256))

env = PixelPathEnv(mask)

# # Random policy simulation

# obs = env.reset()
# done = False
# while not done:
#     action = env.action_space.sample() 
#     obs, reward, done, _ = env.step(action)

# Hyperparameters
alpha = 0.1       # learning rate
gamma = 0.95       # discount factor
epsilon = 0.01     # exploration rate
episodes = 200

# Q-table as dictionary: Q[state][action]
Q = {}

for ep in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        q_values = get_Q(Q, state)
        if random.random() < epsilon:
            action = random.choice([0,1])
        else:
            action = np.argmax(q_values)

        next_state, reward, done, info = env.step(action)
        total_reward += reward

        if not done:
            next_q = get_Q(Q, next_state)
            q_values[action] += alpha * (reward + gamma * max(next_q) - q_values[action])
        else:
            q_values[action] += alpha * (reward - q_values[action])

        state = next_state

    if ep % 10 == 0:
        print(f"Episode {ep}, total reward = {total_reward}")
        visualize_result(env, "row_based_search/episodes_results/lsys_2it", ep)