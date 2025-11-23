import numpy as np
import torch
import matplotlib.pyplot as plt
from dqn_slice_based.env import SliceReconstructionEnv
from dqn_slice_based.dqn_agent import SlicePolicyAgent


def evaluate_agent(volume_path: str, model_path: str, history_len: int = 3, save_dir: str = 'dqn_slice_based/results'):
    """Evaluate trained policy network on a volume"""
    
    # Load volume
    volume = np.load(volume_path)
    print(f"Loaded volume with shape: {volume.shape}")
    
    # Initialize environment
    env = SliceReconstructionEnv(volume, noise_level=0.3, history_len=history_len)
    n_channels = env.get_num_channels()
    
    # Initialize agent
    agent = SlicePolicyAgent(state_shape=(n_channels, env.height, env.width))
    agent.load(model_path)
    agent.epsilon = 0.0  # No exploration during evaluation
    print(f"Loaded model from {model_path}")
    
    # Run through entire volume
    state = env.reset()
    done = False
    
    all_accuracies = []
    all_ious = []
    all_f1s = []
    reconstructed_volume = np.zeros_like(volume)
    
    while not done:
        # Get action (deterministic)
        prob_map = agent.select_action(state, deterministic=True)
        action_map = (prob_map > 0.5).astype(np.float32)
        
        # Store reconstruction
        slice_idx = env._slice_order[env.current_slice_idx]
        reconstructed_volume[slice_idx] = action_map
        
        # Take step
        next_state, reward, done, info = env.step(action_map)
        
        all_accuracies.append(info['accuracy'])
        all_ious.append(info['iou'])
        all_f1s.append(info['f1_score'])
        
        state = next_state
    
    # Print results
    print("\n=== Evaluation Results ===")
    print(f"Average Accuracy: {np.mean(all_accuracies):.4f} ± {np.std(all_accuracies):.4f}")
    print(f"Average IoU: {np.mean(all_ious):.4f} ± {np.std(all_ious):.4f}")
    print(f"Average F1: {np.mean(all_f1s):.4f} ± {np.std(all_f1s):.4f}")
    
    # Visualize results
    visualize_full_reconstruction(env, reconstructed_volume, all_accuracies, all_ious, save_dir)
    
    return reconstructed_volume, all_accuracies, all_ious, all_f1s


def visualize_full_reconstruction(env, reconstructed_volume, accuracies, ious, save_dir):
    """Visualize full volume reconstruction"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot metrics over slices
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    slice_indices = range(len(accuracies))
    axes[0].plot(slice_indices, accuracies, label='Accuracy', marker='o', markersize=3)
    axes[0].plot(slice_indices, ious, label='IoU', marker='s', markersize=3)
    axes[0].set_xlabel('Slice Index')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Metrics per Slice')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Histogram of accuracies
    axes[1].hist(accuracies, bins=20, alpha=0.7, label='Accuracy')
    axes[1].hist(ious, bins=20, alpha=0.7, label='IoU')
    axes[1].set_xlabel('Score')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Score Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'evaluation_metrics.png'), dpi=150)
    plt.close()
    
    # Visualize sample slices
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    slice_indices = np.linspace(0, env.depth - 1, 4, dtype=int)
    
    for i, slice_idx in enumerate(slice_indices):
        noisy = env.noisy_volume[slice_idx]
        target = env.target_volume[slice_idx]
        recon = reconstructed_volume[slice_idx]
        
        axes[i, 0].imshow(noisy, cmap='gray')
        axes[i, 0].set_title(f'Noisy (slice {slice_idx})')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(recon, cmap='gray')
        axes[i, 1].set_title(f'Reconstructed')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(target, cmap='gray')
        axes[i, 2].set_title('Target')
        axes[i, 2].axis('off')
        
        diff = np.abs(recon - target)
        axes[i, 3].imshow(diff, cmap='hot')
        axes[i, 3].set_title(f'Difference (Acc={accuracies[slice_idx]:.3f})')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sample_reconstructions.png'), dpi=150)
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--volume_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--history_len', type=int, default=3)
    parser.add_argument('--save_dir', type=str, default='dqn_slice_based/results')
    
    args = parser.parse_args()
    
    evaluate_agent(args.volume_path, args.model_path, args.history_len, args.save_dir)