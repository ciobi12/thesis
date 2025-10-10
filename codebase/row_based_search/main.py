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
    axs[1].set_title(f"Coverage: {env.coverage()*100:.3f}%")
    axs[1].axis("off")
    fig.suptitle('Episode {}'.format(ep))

    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{ep}.png"))
        plt.close()
        

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
    
iterations = 3
angle = 22.5
step = 5
segments = lsys_obj.build_l_sys(iterations = iterations, step = step, angle_deg = angle)
# lsys_obj.draw_lsystem()
mask = lsys_obj.build_mask(canvas_size=(192, 256))
print(np.unique(mask))
env = PixelPathEnv(mask, render_mode="array")

# Hyperparameters
alpha = 0.2    # learning rate
gamma = 0.9      
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


    if ep % 25 == 0:
        print(f"Episode {ep}, total reward = {total_reward}, eps = {eps:.3f}")
        visualize_result(env, f"row_based_search/episodes_results/lsys_{iterations}it", ep)


plt.subplot(3,1,1)
plt.plot(range(episodes), rewards)
plt.title("Rewards")
plt.grid(True)
plt.subplot(3,1,2)
plt.plot(range(episodes), epsilons, color='orange', linestyle='dashed')
plt.title("Epsilon")
plt.grid(True)
plt.subplot(3,1,3)
plt.plot(range(episodes), coverages, color='red')
plt.title("Path Coverage")
plt.ylabel("Coverage (%)")
plt.grid(True)
plt.savefig(f"{os.getcwd()}/row_based_search/episodes_results/lsys_{iterations}it/results.png")
plt.show()