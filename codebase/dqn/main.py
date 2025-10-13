from dqn.intra_dqn import PatchNavNet, ReplayBuffer, dqn_update
from dqn.inter_dqn import PatchSelNet, InterReplayBuffer, masked_argmax
from dqn.env import PathTraversalEnv
from l_systems import LSystemGenerator

import os
import torch
import torch.nn.functional as F
import numpy as np
import random   
from matplotlib import pyplot as plt

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
mask = lsys_obj.build_mask(canvas_size=(64, 64))
# print(np.unique(mask))
env = PathTraversalEnv(path_mask=mask, 
                       patch_size = 3, 
                       target_coverage=0.99, 
                       render_mode = "rgb_array")
intra = PatchNavNet(N=env.patch_size); intra_tgt = PatchNavNet(N=env.patch_size)
inter = PatchSelNet(grid=3, k_actions=8); inter_tgt = PatchSelNet(grid=3, k_actions=8)
intra_tgt.load_state_dict(intra.state_dict()); inter_tgt.load_state_dict(inter.state_dict())
opt_intra = torch.optim.Adam(intra.parameters(), 1e-3)
opt_inter = torch.optim.Adam(inter.parameters(), 1e-3)
rb_intra, rb_inter = ReplayBuffer(), InterReplayBuffer()
eps_intra, eps_inter = 0.2, 0.3
EPISODES = 10

def select_global_action(env, eps):
    grid = torch.from_numpy(env.patch_grid(grid_size=3)).unsqueeze(0)  # [1,C,H,W]
    px, py = env.curr_patch_idx
    x_norm = px / max(1, (env.W // env.patch_size))
    y_norm = py / max(1, (env.H // env.patch_size))
    ep_cov, total_cov = env.coverage()
    coords_cov = torch.tensor([[x_norm, y_norm, ep_cov, total_cov]], dtype=torch.float32)
    cands, mask = env.frontier_candidates(K=8)
    q = inter(grid.float(), coords_cov)
    if random.random() < eps or mask.sum() == 0:
        a = int(np.random.choice(np.where(mask == 1)[0])) if mask.sum() > 0 else 0
    else:
        a = int(masked_argmax(q, torch.from_numpy(mask)).item())
    return (grid, coords_cov, torch.from_numpy(mask).unsqueeze(0)), cands, a

for ep in range(EPISODES):
    env.reset(options={"start_from_seed": True, "reset_global_mask": True})
    done_global = False
    while not done_global:
        s_inter, cands, a_patch = select_global_action(env, eps_inter)
        cand = cands[a_patch] if len(cands) > 0 else None
        moved = env.select_patch_candidate(cand) if cand is not None else env._advance_to_frontier_patch()
        # Run local episode in the selected patch
        delta_before = int((env.explored_path & env.path_mask).sum())
        for t in range(env.max_steps_per_patch):
            s_local = torch.from_numpy(env.local_state_tensor()).unsqueeze(0)
            if random.random() < eps_intra:
                a = np.random.randint(8)
            else:
                a = int(intra(s_local.float()).argmax(dim=1)[0])
            obs, r, patch_done, dead_end, terminated, truncated, info = env.step(a)
            s_local_next = torch.from_numpy(env.local_state_tensor()).unsqueeze(0)
            rb_intra.push(s_local.squeeze(0), a, r, s_local_next.squeeze(0), float(patch_done))
            if len(rb_intra) > 1000:
                dqn_update(intra, intra_tgt, opt_intra, rb_intra)
            if patch_done or terminated or truncated: 
                break

        delta_after = int((env.explored_path & env.path_mask).sum())
        delta_new = max(0, delta_after - delta_before)
        r_inter = (1 if cand and cand["has_path"] else -1) + 0.01 * delta_new

        # Next inter state
        s2_inter, _, _ = select_global_action(env, eps_inter)  # we only need next state tensors

        rb_inter.push(*s_inter, a_patch, r_inter, *s2_inter, float(terminated))
        if len(rb_inter) > 64:
            s_pg, s_cc, s_mask, a_i, r_i, n_pg, n_cc, n_mask, d_i = rb_inter.sample(64)
            q = inter(s_pg.squeeze().float(), s_cc.squeeze().float()).gather(1, a_i.view(-1,1)).squeeze(1)
            with torch.no_grad():
                q_next = inter(n_pg.squeeze().float(), n_cc.squeeze().float())
                # print(q_next.shape, n_mask.shape)
                a_star = masked_argmax(q_next, n_mask).view(-1,1)
                q_tgt = inter_tgt(n_pg.squeeze().float(), n_cc.squeeze().float()).gather(1, a_star).squeeze(1)
                y = r_i + (1 - d_i) * 0.99 * q_tgt
            loss = F.smooth_l1_loss(q, y)
            opt_inter.zero_grad(); loss.backward(); opt_inter.step()

        if terminated or truncated:
            done_global = True
    if ep % 2 == 0:
        intra_tgt.load_state_dict(intra.state_dict())
        inter_tgt.load_state_dict(inter.state_dict())
    eps_intra = max(0.05, eps_intra * 0.995)
    eps_inter = max(0.05, eps_inter * 0.995)

    ep_coverage = info.get("total_coverage", 0)*100
    if env.render_mode == "rgb_array":
            frame = env.render()
            fig, ax = plt.subplots(1, 1, figsize=(6, 10))
            ax.imshow(frame, origin="upper")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"Coverage: {ep_coverage:.2f}%")
            plt.savefig(f"{os.getcwd()}/dqn/episodes_results/lsys_{iterations}it/ep_{ep+1}_total_coverage_{ep_coverage:.2f}.png")
            plt.close()
    # target sync + epsilon decay
    