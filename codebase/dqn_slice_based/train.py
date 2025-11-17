import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from typing import List

from dqn_slice_based.env import SliceReconstructionEnv
from dqn_slice_based.dqn_agent import SliceDQNAgent


def train_slice_dqn(
    volume_path: str,
    num_episodes: int = 1000,
    target_update: int = 10,
    save_interval: int = 100,
    save_dir: str = 'dqn_slice_based/models',
    results_dir: str = 'dqn_slice_based/results',
    history_len: int = 3
):
    """
    Train DQN agent for slice reconstruction with multi-channel input.
    Each episode processes all slices in the volume sequentially.
    
    Args:
        volume_path: Path to .npy file containing 3D volume
        num_episodes: Number of training episodes (full volume passes)
        target_update: Update target network every N episodes
        save_interval: Save model every N episodes
        save_dir: Directory to save models
        results_dir: Directory to save results
        history_len: Number of previous slices to include in observation
    """
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load volume
    volume = np.load(volume_path)
    print(f"Loaded volume with shape: {volume.shape}")
    
    # Initialize environment and agent
    env = SliceReconstructionEnv(volume, noise_level=0.05, history_len=history_len)
    n_channels = env.get_num_channels()
    print(f"Input channels: {n_channels} (1 current + {history_len} previous)")
    
    agent = SliceDQNAgent(
        state_shape=(n_channels, env.height, env.width),
        n_actions=2,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=0.1,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=32
    )
    
    # Training metrics (per episode)
    episode_rewards = []
    episode_losses = []
    episode_accuracies = []
    episode_ious = []
    episode_f1s = []
    
    # Training loop
    pbar = tqdm(range(num_episodes), desc="Training")
    for episode in pbar:
        # Reset environment to first slice
        state = env.reset()  # Now returns (C, H, W)
        
        # Episode metrics (per slice)
        slice_rewards = []
        slice_losses = []
        slice_accuracies = []
        slice_ious = []
        slice_f1s = []
        
        done = False
        
        # Process all slices in the volume
        while not done:
            # Select action (reconstruct current slice)
            action_map = agent.select_action(state, deterministic=False)
            
            # Take step
            next_state, reward, done, info = env.step(action_map)
            
            # Store transition
            agent.store_transition(state, action_map, reward, next_state, done)
            
            # Train
            loss = agent.train_step()
            
            # Record slice metrics
            slice_rewards.append(reward)
            slice_losses.append(loss)
            slice_accuracies.append(info['accuracy'])
            slice_ious.append(info['iou'])
            slice_f1s.append(info['f1_score'])
            
            # Move to next slice
            state = next_state
        
        # Update target network
        if episode % target_update == 0:
            agent.update_target_network()
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Record episode metrics (average over all slices)
        episode_rewards.append(np.mean(slice_rewards))
        episode_losses.append(np.mean(slice_losses))
        episode_accuracies.append(np.mean(slice_accuracies))
        episode_ious.append(np.mean(slice_ious))
        episode_f1s.append(np.mean(slice_f1s))
        
        # Update progress bar
        pbar.set_postfix({
            'avg_reward': f'{episode_rewards[-1]:.3f}',
            'avg_acc': f'{episode_accuracies[-1]:.3f}',
            'avg_iou': f'{episode_ious[-1]:.3f}',
            'eps': f'{agent.epsilon:.3f}',
            'slices': len(slice_rewards)
        })
        
        # Save model
        if (episode + 1) % save_interval == 0:
            model_path = os.path.join(save_dir, f'model_episode_{episode+1}.pth')
            agent.save(model_path)
            print(f"\nModel saved to {model_path}")
            
            # Visualize some results
            visualize_reconstruction(env, agent, results_dir, episode + 1)
    
    # Final save
    agent.save(os.path.join(save_dir, 'model_final.pth'))
    
    # Plot training curves
    plot_training_curves(
        episode_rewards, episode_losses, episode_accuracies, 
        episode_ious, episode_f1s, results_dir
    )
    
    return agent, env


def visualize_reconstruction(env, agent, save_dir, episode):
    """Visualize reconstruction on evenly spaced slices"""
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Select 3 evenly spaced slices
    slice_indices = np.linspace(0, env.depth - 1, 3, dtype=int)
    
    for i, slice_idx in enumerate(slice_indices):
        # Get noisy and target slices
        noisy_slice = env.noisy_volume[slice_idx]
        target_slice = env.target_volume[slice_idx]
        
        # Build observation (for visualization, use zeros for history)
        obs_channels = [noisy_slice] + [np.zeros_like(noisy_slice) for _ in range(env.history_len)]
        obs = np.stack(obs_channels, axis=0)
        
        # Reconstruct
        action_map = agent.select_action(obs, deterministic=True)
        
        # Plot
        axes[i, 0].imshow(noisy_slice, cmap='gray')
        axes[i, 0].set_title(f'Noisy Input (slice {slice_idx})')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(action_map, cmap='gray')
        axes[i, 1].set_title('Reconstruction')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(target_slice, cmap='gray')
        axes[i, 2].set_title('Target')
        axes[i, 2].axis('off')
        
        # Difference map
        diff = np.abs(action_map - target_slice)
        axes[i, 3].imshow(diff, cmap='hot')
        axes[i, 3].set_title('Difference')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'reconstruction_ep{episode}.png'))
    plt.close()


def plot_training_curves(rewards, losses, accuracies, ious, f1s, save_dir):
    """Plot training metrics (averaged per episode)"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    episodes = range(len(rewards))
    
    # Rewards
    axes[0, 0].plot(episodes, rewards)
    axes[0, 0].set_title('Average Reward per Episode')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)
    
    # Losses
    axes[0, 1].plot(episodes, losses)
    axes[0, 1].set_title('Average Loss per Episode')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True)
    
    # Accuracy
    axes[0, 2].plot(episodes, accuracies)
    axes[0, 2].set_title('Average Accuracy per Episode')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Accuracy')
    axes[0, 2].grid(True)
    
    # IoU
    axes[1, 0].plot(episodes, ious)
    axes[1, 0].set_title('Average IoU per Episode')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].grid(True)
    
    # F1 Score
    axes[1, 1].plot(episodes, f1s)
    axes[1, 1].set_title('Average F1 Score per Episode')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('F1')
    axes[1, 1].grid(True)
    
    # Moving average
    window = 50
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[1, 2].plot(range(len(moving_avg)), moving_avg)
        axes[1, 2].set_title(f'Moving Average Reward (window={window})')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Avg Reward')
        axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()