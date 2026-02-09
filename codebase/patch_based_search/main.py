from patch_based_search.env import PathTraversalEnv
from patch_based_search.qlearner import PatchQLearner
from l_systems_builder.l_systems_2d.l_systems_2d import LSystem2DGenerator

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from time import perf_counter
import os

def get_highest_avg_q_pixel(Q_table, patch_size):
    avg_q = Q_table.mean(axis=1) 
    best_idx = int(np.argmax(avg_q))
    x = best_idx % patch_size
    y = best_idx // patch_size
    return (x, y)

def train(env: PathTraversalEnv, agent: PatchQLearner, lsys_iter = 2, episodes=100):
    rewards = []
    epsilons = []
    path_coverage = []
    options = {"start_from_seed": True, "reset_global_mask": False}
    dead_end = False

    for ep in range(episodes):
        # if ep == 10: 
        #     options["reset_global_mask"] = True
        # else:
        #     options["reset_global_mask"] = False
        obs, _ = env.reset(options = options)
        ep_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated or dead_end):
            patch_idx = env.curr_patch_idx
            s = agent.state_index(obs)
            a = agent.select_action(patch_idx, s)
            obs_next, r, patch_done, dead_end, terminated, truncated, info = env.step(a)

            if dead_end:
                options["start_from_seed"] = False
                
            s_next = agent.state_index(obs_next)
            agent.update(patch_idx, s, a, r, s_next, terminated or truncated or dead_end)
            obs = obs_next
            ep_reward += r
        ep_coverage = info.get("total_coverage", 0)*100

        rewards.append(ep_reward)
        epsilons.append(agent.eps)
        path_coverage.append(ep_coverage)
        
        if env.render_mode == "rgb_array" and ep % 25 == 0:
            frame = env.render()
            fig, ax = plt.subplots(1, 1, figsize=(4, 5))
            ax.imshow(frame)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"Episode: {ep + 1}, Coverage: {ep_coverage:.2f}%")
            plt.savefig(f"{os.getcwd()}/patch_based_search/episodes_results/lsys_{lsys_iter}it/ep_{ep+1}_total_coverage_{ep_coverage:.2f}.png")
            plt.close()
            if terminated:
                frame = env.render()
                fig, ax = plt.subplots(1, 1, figsize=(4, 5))
                ax.imshow(frame, origin="upper")
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"Episode: {ep + 1}, Coverage: {ep_coverage:.2f}%")
                plt.savefig(f"{os.getcwd()}/patch_based_search/episodes_results/lsys_{lsys_iter}it/ep_{ep+1}_total_coverage_{ep_coverage:.2f}.png")
                plt.close()
                return agent, epsilons, rewards, path_coverage, ep      
            
    return agent, epsilons, rewards, path_coverage, ep              

if __name__ == "__main__":
    lsys_obj = LSystem2DGenerator(axiom = "X",
                                rules = {"X": "F+[[X]-X]-F[-FX]+X",
                                         "F": "FF"
                                         }
                                )
    
    iterations = 3
    angle = 22.5
    step = 5

    segments = lsys_obj.build_l_sys(iterations = iterations, step = step, angle_deg = angle, start_angle = -90)
    img, mask = lsys_obj.draw_lsystem_ct_style(canvas_size=(128, 256),
                                               margin = 10,
                                               root_width = 2,
                                               lsys_save_path=f"patch_based_search/examples/lsys_{iterations}it.png",
                                               mask_save_path = f"patch_based_search/examples/lsys_{iterations}it_mask.png",
                                               add_ct_noise = False,
                                               occlude_root = False,
                                               skip_segments = False
                                               )
    env = PathTraversalEnv(path_mask=mask, 
                           patch_size = 3, 
                           target_coverage=0.95, 
                           render_mode = "rgb_array")
    
    episodes = 200
    agent = PatchQLearner(patch_size=env.patch_size, alpha=0.2, gamma=0.9, eps_start = 0.2, eps_min = 0.001, eps_decay=0.95)
    start = perf_counter()
    trained_agent, epsilons, rewards, path_coverage, terminal_ep = train(env, agent, lsys_iter = iterations, episodes = episodes)
    end = perf_counter()
    print(f"Training completed in {end - start:.2f} seconds over {terminal_ep+1} episodes.")

    results_dir = f"patch_based_search/episodes_results/lsys_{iterations}it"
    os.makedirs(results_dir, exist_ok=True)

    # Save .dat files for LaTeX/pgfplots
    episodes_arr = np.arange(1, len(rewards) + 1)
    np.savetxt(f"{results_dir}/rewards.dat",
               np.column_stack([episodes_arr, rewards]),
               header="episode reward", comments="")
    np.savetxt(f"{results_dir}/epsilons.dat",
               np.column_stack([episodes_arr, epsilons]),
               header="episode epsilon", comments="")
    np.savetxt(f"{results_dir}/path_coverage.dat",
               np.column_stack([episodes_arr, path_coverage]),
               header="episode coverage", comments="")

    fig, axs = plt.subplots(1, 3, figsize=(15,5))

    axs[0].plot(range(len(rewards)), rewards)
    axs[0].set_title("Rewards")
    axs[0].grid(True)

    axs[1].plot(range(len(epsilons)), epsilons, color='orange', linestyle='dashed')
    axs[1].set_title("Epsilon")
    axs[1].grid(True)

    axs[2].plot(range(len(path_coverage)), path_coverage, color='red')
    axs[2].set_title("Path Coverage")
    axs[2].set_ylabel("Coverage (%)")
    axs[2].set_xlabel("Episodes")
    plt.grid(True)
    plt.subplots_adjust(top=1.0, bottom=0.0, hspace=0.4)
    plt.savefig(f"{results_dir}/results.png", bbox_inches='tight', pad_inches=0.05)
    # plt.show()