import numpy as np 
import os
import matplotlib.pyplot as plt
import random

from row_based_search.env import PixelPathEnv
from l_systems_builder.l_systems_2d.l_systems_2d import LSystem2DGenerator
from time import perf_counter

def visualize_result(env, ep: int = None, show: bool = False, save_path: str = None) -> None:
    fig, axs = plt.subplots(1, 1, figsize=(3, 6))

    axs.imshow(env.output, cmap="gray")
    axs.set_title(f"Episode {ep} | Coverage: {env.coverage()*100:.3f}%", fontsize=10)
    axs.axis("off")
    plt.tight_layout()
    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()  
        

def get_Q(Q, state):
    if state not in Q.keys():
        Q[state] = [0.0, 0.0]  # two actions: 0 or 1
    return Q[state]

# # Example toy image
# image = np.random.choice([0, 255], size=(16, 16))
# env = PixelPathEnv(image)

lsys_obj = LSystem2DGenerator(axiom = "X",
                              rules = {"X": "F+[[X]-X]-F[-FX]+X",
                                       "F": "FF"})

iterations = 2
angle = 22.5
step = 5

segments = lsys_obj.build_l_sys(iterations = iterations, step = step, angle_deg = angle, start_angle = -90)
img, mask = lsys_obj.draw_lsystem_ct_style(canvas_size=(128, 256),
                                               margin = 10,
                                               root_width = 1,
                                               lsys_save_path=f"row_based_search/examples/lsys_{iterations}it.png",
                                               mask_save_path = f"row_based_search/examples/lsys_{iterations}it_mask.png",
                                               add_ct_noise = False,
                                               occlude_root = False,
                                               skip_segments = False
                                               )

# lsys_obj.draw_lsystem()
env = PixelPathEnv(mask, render_mode="array")

# Hyperparameters
alpha = 0.2   # learning rate
gamma = 0.95      
eps_start = 0.2
eps_min = 0.001     
eps_decay = 0.95
episodes = 200

# Q-table as dictionary: Q[state][action]
Q = {}

rewards = []
epsilons = []
coverages = []
eps = eps_start

start = perf_counter()
for ep in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        q_values = get_Q(Q, state)
        if random.random() < eps:
            action = random.choice([0, 1])
        else:
            action = np.argmax(q_values)

        next_state, reward, done, info = env.step(action)
        total_reward += reward / 255

        if not done:
            next_q = get_Q(Q, next_state)
            q_values[action] += alpha * (reward + gamma * max(next_q) - q_values[action])
        else:
            q_values[action] += alpha * (reward - q_values[action])
            eps = max(eps_min, eps * eps_decay)

        state = next_state

    coverage = env.coverage()
    rewards.append(total_reward)
    epsilons.append(eps)
    coverages.append(coverage*100)

    if ep % 5 == 0:
        print(f"Episode {ep}, total reward = {total_reward}, eps = {eps:.3f}")
        visualize_result(env, ep+1,False, f"row_based_search/episodes_results/lsys_{iterations}it/{ep+1}.png")
    if coverage >= 0.99:
        print(f"Reached target coverage in episode {ep}!")
        visualize_result(env, ep+1, False, f"row_based_search/episodes_results/lsys_{iterations}it/best_cov.png")
        break  
    if  ep == episodes - 1:
        visualize_result(env, ep+1, True, f"row_based_search/episodes_results/lsys_{iterations}it/final.png")

end = perf_counter()
print(f"Training completed in {end - start:.2f} seconds.")


fig, axs = plt.subplots(3, 1, figsize=(6, 11))

axs[0].plot(range(len(rewards)), rewards)
axs[0].set_title("Rewards")
axs[0].grid(True)

axs[1].plot(range(len(epsilons)), epsilons, color='orange', linestyle='dashed')
axs[1].set_title("Epsilon")
axs[1].grid(True)

axs[2].plot(range(len(coverages)), coverages, color='red')
axs[2].set_title("Path Coverage")
axs[2].set_ylabel("Coverage (%)")
axs[2].set_xlabel("Episodes")
plt.grid(True)
plt.savefig(f"row_based_search/episodes_results/lsys_{iterations}it/results.png")
plt.show()