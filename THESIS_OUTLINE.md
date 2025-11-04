# Reinforcement Learning for 3D Image Segmentation of Long-Range Connected Structures

This outline is tailored to the current codebase at `thesis/codebase` and the project brief. Each chapter lists concrete artifacts, figures, and where applicable links to existing modules and experiments.

## Front matter
- Abstract (≤250 words)
- Acknowledgements
- Table of Contents
- List of Figures and Tables

## 1. Introduction
- Problem: tracing elongated, branching, low-contrast structures (e.g., vasculature) with continuity.
- Limitations of conventional segmentation (connectivity breaks, noise sensitivity, thresholding).
- Approach: formulate segmentation as sequential decision making from a seed; RL agent navigates and builds a consistent path.
- Contributions (draft):
  1) Synthetic dataset generator for branching paths (L-systems + noise/artifacts).
  2) Patch-based environment for local movement with global coverage objective.
  3) Baselines: row-wise reconstruction, tabular Q-learning, and DQN variants (intra-/inter-patch).
  4) Analysis of reward shaping, state design, and frontier patch selection.
  5) Conceptual and technical framework for scaling to 3D via patch navigation and stitching.
- Thesis structure overview.

## 2. Background and Related Work
- Image segmentation of curvilinear structures: vesselness (Frangi), ridge detectors, active contours/snake models, graph-based tracing, minimal path methods, classical morphological pipelines; deep models (U-Net), centerline extraction and topology preservation.
- Reinforcement Learning for vision and medical imaging; navigation/active search; MDP formalization of tracing.
- Patch-based processing for large 3D volumes: memory constraints, context loss, stitching strategies, hierarchical context.
- Markov Decision Process (MDP) for tracing:
  - State: local patch observation + global context.
  - Actions: 8-neighborhood movement (local) and patch selection (global/frontier).
  - Rewards: path discovery vs. penalties for off-path, revisits, and dead-ends.

## 3. Data and Environment Design
### 3.1 Synthetic Data Generation (2D)
- L-systems for branching curves with parameters: iterations, angles, step (code: `codebase/l_systems.py`, class `LSystemGenerator`).
- Rasterization to images and masks; noise (Gaussian), salt & pepper, random circular artifacts to mimic CT/MicroCT (code: `draw_lsystem`).
- Dataset scripts and presets (code: `codebase/dataset_generator.py`).
- Datasets folder layout: `codebase/data/train`, `codebase/data/val` (current repo structure).
- Figures:
  - Example L-system paths and masks under varying iterations/angles.
  - Noise/artifact augmentation examples.

### 3.2 Environments (MDP)
- Local patch movement env (DQN): `codebase/dqn/env.py` `PathTraversalEnv`.
  - Action space: 8-connected moves.
  - Observations: local coordinates (and local tensor variant via `local_state_tensor()`), explored mask, and agent marker.
  - Patch logic: boundaries, coverage tracking, frontier selection and patch transitions.
  - Rewards: +1 for new-path pixel, −0.1 for revisits, −10 for off-path; termination on coverage.
  - Global context helpers: `patch_grid`, `frontier_candidates`, `coverage`, rendering overlay for qualitative visuals.
- Row-based env: `codebase/row_based_search/env.py` `PathReconstructionEnv`.
  - Row-wise reconstruction with continuity regularization.
- Pixel-path env (early prototype): `codebase/patch_based_search/env.py` `PixelPathEnv`.

- Figures:
  - MDP diagram: state (patch tensor + global grid), actions (move/select patch), rewards.
  - Patch grid visualization and frontier selection concept.

## 4. Methods
### 4.1 Baselines
- Row-wise search: `row_based_search/main.py` with `PathReconstructionEnv`.
- Tabular Q-Learning in patches: `patch_based_search/qlearner.py` `PatchQLearner` with `PixelPathEnv`.

### 4.2 Deep RL Agents (DQN family)
- Intra-patch policy (movement): `dqn/intra_dqn.py` (network `PatchNavNet`, replay buffer, DQN update).
- Inter-patch policy (frontier/next patch selection): `dqn/inter_dqn.py` (network `PatchSelNet`, masked argmax for valid frontier actions).
- Joint training loop coupling intra/inter agents: `dqn/main.py`.
  - Curriculum: start from seed, episodic within-patch rollout with max steps; reward aggregation to inter-agent via delta coverage.
  - Target networks, epsilon decay, replay buffers.
- Row-based DQN: `dqn_row_based/` (if used) for comparison on row-wise formulation.

### 4.3 Reward Design and Ablations
- Components: new-path discovery, revisit penalty, off-path penalty, inter-agent bonus for selecting patches containing path and increasing global coverage.
- Potential extensions: junction preference, curvature smoothness, backtracking cost, dead-end handling.

### 4.4 Implementation Details
- Libraries: Gymnasium, NumPy, PyTorch, Matplotlib, OpenCV, PIL.
- Training setup: patch size, grid size, step limits, learning rates, replay sizes, target sync.
- Reproducibility: seeds, versions, folder conventions (`episodes_results/*`).

## 5. Experiments
### 5.1 Experimental Setup
- Datasets: multiple L-system types (plant, bush, tree, fern, fractal, palm) with varied iterations/noise.
- Train/val splits in `codebase/data`.
- Metrics (2D):
  - Centerline/path coverage: |pred ∧ path| / |path|.
  - Precision/Recall/F1 on path pixels; IoU.
  - Connectivity score: fraction of path components connected in prediction.
  - Length error and branch recall at junctions.
  - Efficiency: steps to coverage, time/episode, memory.

### 5.2 Baseline Comparisons
- Row-wise reconstruction (classical + row-based DQN if applicable).
- Tabular Q-learning vs DQN intra.
- Heuristic frontier selection vs learned inter-policy.

### 5.3 Ablations
- Reward coefficients: revisit penalty, off-path penalty, inter bonus.
- Observation variants: coordinate-only vs `local_state_tensor` channels (path/explored/agent), grid size for inter-policy.
- Patch size and max steps per patch.
- Start-from-seed vs resume-from-last.

### 5.4 Qualitative Results
- Render overlays from `PathTraversalEnv.render()` saved in `dqn/episodes_results/lsys_*`.
- Success/failure cases: weak signal gaps, noisy artifacts, sharp junctions.

## 6. Toward 3D Volumes: Concept and Prototype
- Challenges: memory limits, local context vs global coherence, patch boundaries, re-entry, stitching.
- 3D extension plan:
  - 3D patches and 26-neighborhood movement; hierarchical inter-agent (patches → slabs → volume).
  - Cross-patch memory: visited path voxel map, frontier voxel/super-voxel candidates.
  - Seed handling and multi-seed expansion for tree coverage.
  - Stitching: overlap-consensus, shortest-bridges over small gaps.
- Prototype roadmap: adapt `PathTraversalEnv` to 3D numpy volumes; use slab-based inter-policy.
- Figures:
  - 3D patch tiling, frontier queues, stitching schematic.

## 7. Discussion
- Strengths: continuity-aware, sparse reward shaping aligns with topology, interpretable traces.
- Limitations: exploration at low SNR, junction ambiguity, sensitivity to reward scales and patch hyperparameters.
- Failure modes observed in current runs: off-path attraction in artifact clusters, oscillations at borders.
- Lessons on state/reward/training choices; how baselines inform design.

## 8. Conclusion and Future Work
- Summary of findings on synthetic 2D.
- Path to robust 3D: memory-efficient representations, hierarchical RL, uncertainty-driven frontier selection, integration with vesselness priors.
- Potential evaluation on real data (micro-CT vasculature) and comparison to U-Net + skeletonization.

## References
- Curate after writing; include RL, vessel segmentation, skeletonization, patch-based 3D processing.

## Appendices
- A. Environment and network definitions (key snippets): `dqn/env.py`, `dqn/intra_dqn.py`, `dqn/inter_dqn.py`.
- B. Hyperparameters and training schedules.
- C. Additional result figures and logs (`episodes_results`).
- D. Synthetic data generation details (L-system parameters, noise settings).
- E. Reproducibility checklist (env versions, seeds, hardware).

---

## Mapping from code to thesis sections (quick index)
- Synthetic data: `codebase/l_systems.py`, `codebase/dataset_generator.py` → Sections 3.1, Appendix D.
- Local env and movement: `codebase/dqn/env.py` (state tensor, frontier, coverage, render) → Sections 3.2, 4.2.
- Intra DQN: `codebase/dqn/intra_dqn.py` → Section 4.2.
- Inter DQN: `codebase/dqn/inter_dqn.py` → Section 4.2.
- Joint trainer and results: `codebase/dqn/main.py`, outputs under `codebase/dqn/episodes_results/` → Sections 4.2, 5.4.
- Row-based baseline: `codebase/row_based_search/env.py`, `row_based_search/main.py` → Sections 4.1, 5.2.
- Tabular Q-learning: `codebase/patch_based_search/env.py`, `patch_based_search/qlearner.py` → Sections 4.1, 5.2.
- Row-based DQN (optional): `codebase/dqn_row_based/*` → Sections 4.2, 5.2.

## Proposed figures and tables
- Fig 1: Problem overview and RL tracing concept.
- Fig 2: L-system generation and noise/artifact examples.
- Fig 3: MDP schematic with intra/inter agents and patch grid.
- Fig 4: Frontier candidate selection visualization.
- Fig 5: Qualitative overlays per episode (selected from `episodes_results`).
- Fig 6: 3D extension and stitching schematic.
- Table 1: Dataset summary (L-system variants, sizes, noise levels).
- Table 2: Metrics and definitions (coverage, connectivity, F1, length error).
- Table 3: Baseline vs RL comparisons on 2D synthetic.
- Table 4: Ablation results (reward/obs/patch size).

## Evaluation plan (concise)
- Metrics: coverage, precision/recall/F1, IoU, connectivity, length error; runtime and memory.
- Baselines: row-based reconstruction, tabular Q-learning, heuristic frontier selection; optionally classical vesselness + threshold + skeleton as an external baseline (if feasible).
- Ablations: reward weights, observation channels, patch/grid sizes, seeding strategy.
- Protocol: fixed seeds; 5 runs avg ± std; separate seen/unseen L-system rule sets.

## Writing roadmap (suggested)
- Week 1: Draft Intro + Background; generate dataset visualizations.
- Week 2: Methods — environments and agents; finalize figures for MDP and pipeline.
- Week 3: Run baselines and ablations on 2D set; collect metrics and overlays.
- Week 4: Discussion + Conclusion; draft 3D extension section; compile appendices.

## Notes
- Consider adding a minimal 3D NumPy prototype env to ground Section 6 (optional, small volume) if time permits.
- Keep all experiment configs and outputs under versioned folders; snapshot key seeds in the appendix.
