"""
Quick script to test reward scales and ensure they're reasonable.
"""
import numpy as np
import nibabel as nib
from dqn_slice_based.env_v2 import SliceSamplerEnv
import os

# Load a small sample
train_dir = 'data/ct_like/rapids-p/train'
files = sorted(os.listdir(train_dir))

train_volume = None
train_mask = None


for f in files:
    path = os.path.join(train_dir, f)
    if f.endswith('.npy'):
        data = np.load(path)
        if 'mask' in f.lower():
            train_mask = data
        else:
            train_volume = data
    elif f.endswith('.nii') or f.endswith('.nii.gz'):
        data = nib.load(path).get_fdata()
        if 'mask' in f.lower():
            train_mask = data
        else:
            train_volume = data

if train_volume is None or train_mask is None:
    print("Could not load data")
    exit(1)

print(f"Volume shape: {train_volume.shape}")
print(f"Mask shape: {train_mask.shape}")

# Check foreground ratio
fg_ratio = (train_mask > 0).mean()
print(f"Overall foreground ratio: {fg_ratio*100:.4f}%")

# Test environment
print("\n=== Testing Environment ===")
env = SliceSamplerEnv(
    volume=train_volume,
    mask=train_mask,
    slices_per_episode=64,
    foreground_weight=2.0,
    use_class_weights=True,
    boundary_coef=0.1,
    prefer_foreground=True,
)

obs, _ = env.reset()
print(f"Observation slice shape: {obs['slice_pixels'].shape}")

# Simulate some actions
print("\n=== Testing Reward Scales ===")
H, W = obs['slice_pixels'].shape

# Test 1: All zeros (predict background)
action_zeros = np.zeros((H, W))
env_test = SliceSamplerEnv(train_volume, train_mask, slices_per_episode=64, foreground_weight=2.0)
obs, _ = env_test.reset()
_, reward_zeros, _, _, info = env_test.step(action_zeros.flatten())
print(f"All zeros action: reward = {reward_zeros:.2f}")

# Test 2: All ones (predict foreground)
action_ones = np.ones((H, W))
env_test2 = SliceSamplerEnv(train_volume, train_mask, slices_per_episode=64, foreground_weight=2.0)
obs2, _ = env_test2.reset()
_, reward_ones, _, _, info2 = env_test2.step(action_ones.flatten())
print(f"All ones action: reward = {reward_ones:.2f}")

# Test 3: Random predictions
action_random = np.random.rand(H, W)
env_test3 = SliceSamplerEnv(train_volume, train_mask, slices_per_episode=64, foreground_weight=2.0)
obs3, _ = env_test3.reset()
_, reward_random, _, _, info3 = env_test3.step(action_random.flatten())
print(f"Random action: reward = {reward_random:.2f}")

# Test 4: Perfect prediction
env_test4 = SliceSamplerEnv(train_volume, train_mask, slices_per_episode=64, foreground_weight=2.0)
obs4, _ = env_test4.reset()
slice_idx = info3['slice_index']  # Use same slice
gt = train_mask[slice_idx]
_, reward_perfect, _, _, info4 = env_test4.step(gt.flatten())
print(f"Perfect action: reward = {reward_perfect:.2f}")

print("\n=== Reward Components ===")
print(f"Base rewards range: [{info['base_rewards'].min():.4f}, {info['base_rewards'].max():.4f}]")
print(f"Continuity rewards range: [{info['continuity_rewards'].min():.4f}, {info['continuity_rewards'].max():.4f}]")
print(f"Boundary rewards range: [{info['boundary_rewards'].min():.4f}, {info['boundary_rewards'].max():.4f}]")
print(f"Pixel rewards range: [{info['pixel_rewards'].min():.4f}, {info['pixel_rewards'].max():.4f}]")

print("\n=== Expected Episode Reward ===")
print(f"If predicting zeros all slices: ~{reward_zeros * 64:.0f}")
print(f"If predicting ones all slices: ~{reward_ones * 64:.0f}")
print(f"If random all slices: ~{reward_random * 64:.0f}")
print(f"If perfect all slices: ~{reward_perfect * 64:.0f}")
