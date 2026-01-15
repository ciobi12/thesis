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
        # print(env.agent_xy)
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

            # if patch_done:
            #     Q_table = agent.Q[env.curr_patch_idx]
            #     x0, y0, _, _ = env.curr_patch_bounds
            #     x_local, y_local = get_highest_avg_q_pixel(Q_table, env.patch_size)
            #     env.agent_xy = (x0 + x_local, y0 + y_local)
            #     env._recenter_patch_to_include(env.agent_xy)
                
            s_next = agent.state_index(obs_next)
            agent.update(patch_idx, s, a, r, s_next, terminated or truncated or dead_end)
            obs = obs_next
            ep_reward += r
        ep_coverage = info.get("total_coverage", 0)*100
        # print('Terminated: ', terminated)
        # print('Truncated:', truncated)
        # print('Dead end:', dead_end)

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
    
    iterations = 4
    angle = 22.5
    step = 5

    segments = lsys_obj.build_l_sys(iterations = iterations, step = step, angle_deg = angle, start_angle = -90)
    img, mask = lsys_obj.draw_lsystem_ct_style(canvas_size=(216, 256),
                                               margin = 20,
                                               root_width = 2,
                                               lsys_save_path=f"patch_based_search/examples/lsys_{iterations}it.png",
                                               mask_save_path = f"patch_based_search/examples/lsys_{iterations}it_mask.png",
                                               add_ct_noise = False,
                                               occlude_root = False,
                                               skip_segments = False
                                               )
    env = PathTraversalEnv(path_mask=mask, 
                           patch_size = 3, 
                           target_coverage=0.99, 
                           render_mode = "rgb_array")
    
    episodes = 200
    agent = PatchQLearner(patch_size=env.patch_size, alpha=0.2, gamma=0.9, eps_start = 0.2, eps_min = 0.001, eps_decay=0.95)
    start = perf_counter()
    trained_agent, epsilons, rewards, path_coverage, terminal_ep = train(env, agent, lsys_iter = iterations, episodes = episodes)
    end = perf_counter()
    print(f"Training completed in {end - start:.2f} seconds over {terminal_ep+1} episodes.")

    plt.subplot(3,1,1)
    plt.plot(range(terminal_ep+1), rewards)
    plt.title("Rewards")
    plt.grid(True)

    plt.subplot(3,1,2)
    plt.plot(range(terminal_ep+1), epsilons, color='orange', linestyle='dashed')
    plt.title("Epsilon")
    plt.grid(True)

    plt.subplot(3,1,3)
    plt.plot(range(terminal_ep+1), path_coverage, color='red')
    plt.title("Path Coverage")
    plt.ylabel("Coverage (%)")
    plt.grid(True)

    plt.savefig(f"{os.getcwd()}/patch_based_search/episodes_results/lsys_{iterations}it/results.png", dpi=1200)
    plt.show()