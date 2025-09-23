from patch_based_search.env_old import PathTraversalEnv
from patch_based_search.qlearner import PatchQLearner
from l_systems import LSystemGenerator

import numpy as np
from PIL import Image
import os


def train(env: PathTraversalEnv, episodes=2000):
    agent = PatchQLearner(patch_size=env.patch_size, eps_decay=1)
    rewards = []
    prev_dead_end = False
    options = {"start_from_seed": True}
    
    for ep in range(episodes):
        if prev_dead_end:
            options["start_from_seed"] = False
        options["reset_global_mask"] = False
        if ep == 10: # reset once after 10 episodes for visualization purposes mostly
            options["reset_global_mask"] = True
        obs, _ = env.reset(options = options)
        # print(env.agent_xy)
        ep_reward = 0.0
        terminated = False
        truncated = False
        dead_end = False
        # actions = []
        
        c = 0
        while not (terminated or truncated or dead_end):
            c += 1
            patch_idx = env.curr_patch_idx
            s = agent.state_index(obs)
            a = agent.select_action(patch_idx, s)
            # actions.append(a)
            obs_next, r, dead_end, terminated, truncated, info = env.step(a)
            prev_dead_end = dead_end
            s_next = agent.state_index(obs_next)

            agent.update(patch_idx, s, a, r, s_next, terminated or truncated)
            obs = obs_next
            ep_reward += r

        # print(c)

        rewards.append(ep_reward)
        # print(f"Episode {ep+1}: reward={ep_reward:.1f}, coverage={info.get('coverage', 0):.3f}, eps={agent.eps:.3f}")
        if env.render_mode == "rgb_array" and (ep % 100 == 0 or terminated):
            frame = env.render()
            im = Image.fromarray(frame)
            im.save(f"{os.getcwd()}/episodes_results/ep_{ep+1}_ep_coverage_{info.get('episode_coverage', 0)*100:.2f}_total_coverage_{info.get('total_coverage', 0)*100:.2f}.png")
        
        if terminated:
            im.save(f"{os.getcwd()}/episodes_results/ep_{ep+1}_total_coverage_{info.get('total_coverage', 0)*100:.2f}.png")
            return agent, rewards

if __name__ == "__main__":
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
    mask = lsys_obj.build_mask(canvas_size=(256, 256))
    env = PathTraversalEnv(path_mask=mask, 
                           patch_size = 5, 
                           target_coverage=0.95, 
                           render_mode = "rgb_array")
    train(env, episodes=2000)