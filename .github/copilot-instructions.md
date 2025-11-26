# Thesis Codebase: RL for 3D Segmentation of Branching Structures

## Project Overview
Research project applying Reinforcement Learning to segment elongated, branching structures (e.g., root systems, vasculature) in CT-like images. The core idea: formulate segmentation as sequential decision-making where an RL agent navigates and traces paths, maintaining connectivity across occlusions and noise.

## Architecture & Key Components

### 1. Data Generation Pipeline (`l_systems_builder/`)
- **L-System generators** create synthetic branching structures (2D and 3D)
  - `l_systems_2d/l_systems_2d.py`: `LSystem2DGenerator` - builds root-like patterns using axioms + rules
  - Key method: `draw_lsystem_ct_style()` renders CT-realistic images with configurable noise, occlusion, and segment gaps
  - Parameters: `occlusion_strength`, `skip_probability`, `ct_background_intensity`, `root_intensity_range`
- **Dataset structure**: `codebase/data/{ct_like/2d,noise_only,with_artifacts}/{train,val}/`
  - Images: `*_ct.png`, Masks: `*_mask.png` + `.npy`
  - Generated via `dataset_generator.py` using predefined L-system rules (plant, bush, tree, fern, fractal, palm)

### 2. Environment Formulations (Different MDP Strategies)

#### Row-Based (`dqn_row_based/`)
- **env.py**: `PathReconstructionEnv` - processes images row-by-row (bottom→top or top→bottom)
- **Action space**: `MultiBinary(W)` - per-pixel binary decision for each row
- **Observation**: Dict with `row_pixels`, `prev_preds` (history_len previous rows), `row_index`
- **Reward components**:
  - Base: `-|action - ground_truth|`
  - Continuity: exponential decay linking to previous rows (see `MULTI_ROW_LINKING_GUIDE.md`)
  - **5 continuity strategies documented**: Decay (default), Centroid, Alignment, Hybrid, Voting
- **Key hyperparams**: `continuity_coef` (0.1-0.2), `continuity_decay_factor` (0.5-0.9), `history_len` (3-5)

#### Patch-Based (`dqn_patch_based/`)
- **env.py**: `PatchClassificationEnv` - traverses NxN patches (bottom-left → top-right)
- **Action space**: `MultiBinary(N*N)` - per-pixel decision within patch
- **Reward adds neighbor smoothness**: `-neighbor_coef * |action[i,j] - image_intensity[neighbors]|`

#### Slice-Based 3D (`dqn_slice_based/`)
- **env.py**: `SliceReconstructionEnv` - processes 3D volumes slice-by-slice
- **Multi-channel observations**: Current noisy slice + `history_len` previous predictions
- **Network**: Encoder-decoder U-Net variant (`unet.py`) with skip connections
  - `SimpleEncoderDecoderPolicy`: 64x64 input, bottleneck at 8x8
  - `TinyEncoderDecoderPolicy`: Lightweight version for faster training

#### Pixel-Path Exploration (`patch_based_search/`)
- **env.py**: `PathTraversalEnv` - early prototype, local patch navigation with 8-connected moves
- **qlearner.py**: Tabular Q-learning baseline using patch-indexed state-action tables

### 3. Training & Networks

#### Row-Based DQN (`dqn_row_based/`)
- **dqn.py**: `PerPixelCNN` - 1D convolution over row pixels + history
  - Input channels: `C + history_len` (image channels + previous row predictions)
  - Output: `(W, 2)` Q-values (background/path) per pixel
- **Training loop** (`main.py`):
  - Epsilon-greedy exploration (1.0 → 0.01 over 10k steps)
  - Replay buffer with per-pixel transitions
  - Target network updated every 500 steps
  - Per-pixel MSE loss: `|Q(s,a) - (r + γ * max Q(s',a'))|`
- **Data loading**: Reads PNG images + masks from `data/{intermediate_dir}/{train,val}/`
- **Model checkpoints**: Saved to `dqn_row_based/models/model_cont.pth`

#### Slice-Based Agent (`dqn_slice_based/`)
- **dqn_agent.py**: `SlicePolicyAgent` wraps policy network + replay buffer
- **train.py**: Main training loop for 3D volumes
- **evaluate.py**: Metrics computation (IoU, F1, accuracy, coverage)

## Critical Workflows

### Running Training
```bash
# Row-based reconstruction (2D)
cd /home/razva/DTU/thesis/codebase
python -m dqn_row_based.main

# Slice-based reconstruction (3D)
python -m dqn_slice_based.main
```

### Generating New Datasets
```python
# Edit dataset_generator.py to configure L-system rules
lsys_obj = LSystem2DGenerator(axiom="X", rules={"X": "F[-X][X]F[-X]+FX", "F": "FF"})
img, mask = lsys_obj.draw_lsystem_ct_style(
    canvas_size=(128, 256),
    skip_segments=True, skip_probability=0.2,  # 20% gaps
    occlude_root=True, occlusion_strength=0.4,  # moderate occlusion
    ct_background_intensity=80,  # dark CT background
    align_top=True  # roots grow downward
)
```

### Switching Continuity Strategies
See `dqn_row_based/MULTI_ROW_LINKING_GUIDE.md` - replace reward calculation in `env.py` lines 66-76 with desired strategy code block. Strategy selection impacts how vertical path connectivity is enforced.

## Project-Specific Conventions

### File Naming Patterns
- **Images**: `{lsys_type}_{variant}_ct.png` (e.g., `bush_root_var1_ct.png`)
- **Masks**: `{lsys_type}_{variant}_mask.png` + `.npy` binary array
- **Models**: `model_cont.pth` (continuity-based), stored in `{method}/models/`
- **Results**: Episode visualizations in `{method}/episodes_results/lsys_{iterations}it/`

### Coordinate Systems
- **Row-based**: Bottom-up (`start_from_bottom=True`) by default - aligns with root growth direction
- **Patch-based**: Bottom-left to top-right traversal
- **Image coords**: NumPy convention (H, W) or (H, W, C), masks are binary float32 (0.0/1.0)

### Hyperparameter Tuning Principles
- **Continuity vs Accuracy trade-off**: Higher `continuity_coef` (>0.2) enforces smoothness but may ignore ground truth; lower (<0.1) allows more flexibility
- **Decay factor guidelines** (from MULTI_ROW_LINKING_GUIDE.md):
  - 0.9: Long memory for very smooth paths
  - 0.7: Balanced (recommended default)
  - 0.5: Short memory for wiggly paths
- **History length**: 3-5 rows typical; longer for better gap bridging but higher memory

### Reward Components Pattern
All environments follow this structure (sum per pixel, then aggregate):
```python
base_rewards = -|action - ground_truth|  # accuracy
continuity_rewards = -coef * linking_term  # smoothness
neighbor_rewards = -coef * smoothness_term  # spatial coherence (patch-based only)
pixel_rewards = base + continuity [+ neighbor]
total_reward = pixel_rewards.sum()
```

## Common Pitfalls & Solutions

1. **Continuity rewards very negative**: Increase `continuity_coef` or switch to Alignment strategy (handles occlusions better)
2. **Model oscillates at patch/row boundaries**: Increase `history_len` or try Hybrid strategy (decay + alignment)
3. **Dataset not loading**: Check `USE_ARTIFACTS` flag in `main.py` matches folder structure (`noise_only` vs `with_artifacts` vs `ct_like/2d`)
4. **Observation shape mismatch**: Verify `history_len` in env matches network input channels (`C + history_len` for row-based, `1 + history_len` for slice-based)

## Testing & Evaluation
- **Metrics**: Coverage (|pred ∧ path| / |path|), IoU, F1, pixel accuracy
- **Qualitative**: `visualize_result()` in main.py shows {original, mask, reconstruction} side-by-side
- **Monitoring**: Track `continuity_rewards` mean/std during training (should stabilize near 0)

## Future 3D Extensions (Section 6 in thesis)
Planned architecture for full 3D volumes:
- 26-neighborhood movement within patches
- Hierarchical inter-agent: patches → slabs → volume
- Cross-patch memory via visited voxel map
- Stitching via overlap-consensus or shortest-bridges over gaps

## File Organization Logic
- `dqn_*_based/`: Complete training systems (env + network + main loop)
- `*_search/`: Baseline/prototype methods (tabular Q-learning, heuristics)
- `l_systems_builder/`: Data generation tools (decoupled from RL)
- `data/`: All datasets organized by noise type and train/val split
