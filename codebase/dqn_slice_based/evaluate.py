import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from tqdm import tqdm

from .env import SliceReconstructionEnv
from .dqn_agent import SliceDQNAgent


def evaluate_volume_reconstruction(
    volume_path: str,
    model_path: str,
    results_dir: str = 'dqn_slice_based/results/evaluation'
):
    """
    Evaluate trained agent by reconstructing entire volume slice by slice.
    
    Args:
        volume_path: Path to target volume
        model_path: Path to trained model
        results_dir: Directory to save evaluation results
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Load volume
    volume = np.load(volume_path)
    depth, height, width = volume.shape
    print(f"Loaded volume with shape: {volume.shape}")
    
    # Initialize environment and agent
    env = SliceReconstructionEnv(volume, noise_level=0.3)
    agent = SliceDQNAgent(state_shape=(height, width))
    agent.load(model_path)
    agent.epsilon = 0.0  # No exploration during evaluation
    
    # Reconstruct entire volume
    reconstructed_volume = np.zeros_like(volume)
    slice_metrics = []
    
    print("Reconstructing volume slice by slice...")
    for slice_idx in tqdm(range(depth)):
        # Reset to specific slice
        state = env.reset(slice_idx=slice_idx)
        
        # Reconstruct slice
        action_map = agent.select_action(state, deterministic=True)
        
        # Evaluate
        _, reward, _, info = env.step(action_map)
        
        # Store reconstruction
        reconstructed_volume[slice_idx] = action_map
        slice_metrics.append(info)
    
    # Compute overall metrics
    avg_accuracy = np.mean([m['accuracy'] for m in slice_metrics])
    avg_iou = np.mean([m['iou'] for m in slice_metrics])
    avg_f1 = np.mean([m['f1_score'] for m in slice_metrics])
    
    print(f"\n=== Overall Metrics ===")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")
    
    # Save reconstructed volume
    volume_save_path = os.path.join(results_dir, 'reconstructed_volume.npy')
    np.save(volume_save_path, reconstructed_volume)
    print(f"Reconstructed volume saved to {volume_save_path}")
    
    # Visualize results
    visualize_volume_slices(volume, reconstructed_volume, results_dir)
    plot_slice_metrics(slice_metrics, results_dir)
    
    # Save metrics
    metrics_save_path = os.path.join(results_dir, 'metrics.npz')
    np.savez(metrics_save_path,
             accuracy=avg_accuracy,
             iou=avg_iou,
             f1=avg_f1,
             slice_metrics=slice_metrics)
    
    return reconstructed_volume, slice_metrics


def visualize_volume_slices(original, reconstructed, save_dir, num_slices=6):
    """Visualize random slices comparing original and reconstruction"""
    depth = original.shape[0]
    slice_indices = np.linspace(0, depth-1, num_slices, dtype=int)
    
    fig, axes = plt.subplots(3, num_slices, figsize=(3*num_slices, 9))
    
    for i, idx in enumerate(slice_indices):
        # Original
        axes[0, i].imshow(original[idx], cmap='gray')
        axes[0, i].set_title(f'Original (z={idx})')
        axes[0, i].axis('off')
        
        # Reconstructed
        axes[1, i].imshow(reconstructed[idx], cmap='gray')
        axes[1, i].set_title(f'Reconstructed')
        axes[1, i].axis('off')
        
        # Difference
        diff = np.abs(original[idx].astype(float) - reconstructed[idx].astype(float))
        axes[2, i].imshow(diff, cmap='hot')
        axes[2, i].set_title(f'Difference')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'volume_slices_comparison.png'), dpi=150)
    plt.close()


def plot_slice_metrics(metrics, save_dir):
    """Plot metrics for each slice"""
    slice_indices = [m['slice_idx'] for m in metrics]
    accuracies = [m['accuracy'] for m in metrics]
    ious = [m['iou'] for m in metrics]
    f1s = [m['f1_score'] for m in metrics]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(slice_indices, accuracies, marker='o', markersize=3)
    axes[0].set_xlabel('Slice Index')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy per Slice')
    axes[0].grid(True)
    
    axes[1].plot(slice_indices, ious, marker='o', markersize=3)
    axes[1].set_xlabel('Slice Index')
    axes[1].set_ylabel('IoU')
    axes[1].set_title('IoU per Slice')
    axes[1].grid(True)
    
    axes[2].plot(slice_indices, f1s, marker='o', markersize=3)
    axes[2].set_xlabel('Slice Index')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('F1 Score per Slice')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'slice_metrics.png'), dpi=150)
    plt.close()