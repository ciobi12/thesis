from patch_based_search.env import PathTraversalEnv
from patch_based_search.qlearner import PatchQLearner
from l_systems import LSystemGenerator

import numpy as np
from PIL import Image
import os

def get_highest_avg_q_pixel(Q_table, patch_size):
    avg_q = Q_table.mean(axis=1) 
    best_idx = int(np.argmax(avg_q))
    x = best_idx % patch_size
    y = best_idx // patch_size
    return (x, y)

def train(env: PathTraversalEnv, lsys_iter = 2, episodes=100):
    agent = PatchQLearner(patch_size=env.patch_size, eps_start = 0.8, eps_min = 0.1, eps_decay=0.999)
    rewards = []
    options = {"start_from_seed": True, "reset_global_mask": False}
    dead_end = False

    for ep in range(episodes):
        if ep == 20: # reset once after 10 episodes for visualization purposes mostly
            options["reset_global_mask"] = True
        else:
            options["reset_global_mask"] = False
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

        print('Terminated: ', terminated)
        print('Truncated:', truncated)
        print('Dead end:', dead_end)

        rewards.append(ep_reward)
        print(f"Episode {ep+1}: reward={ep_reward:.1f}, coverage={info.get('total_coverage', 0):.3f}, eps={agent.eps:.3f}")
        # if env.render_mode == "rgb_array" and (ep % 50 == 0 or terminated):
        if env.render_mode == "rgb_array":
            frame = env.render()
            im = Image.fromarray(frame)
            im.save(f"{os.getcwd()}/patch_based_search/episodes_results/lsys_{lsys_iter}it/ep_{ep+1}_ep_coverage_{info.get('episode_coverage', 0)*100:.2f}_total_coverage_{info.get('total_coverage', 0)*100:.2f}.png")
            if terminated:
                return agent, rewards            

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
    mask = lsys_obj.build_mask(canvas_size=(256, 256))
    env = PathTraversalEnv(path_mask=mask, 
                           patch_size = 5, 
                           target_coverage=0.95, 
                           render_mode = "rgb_array")
    train(env, lsys_iter = iterations)