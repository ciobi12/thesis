import argparse
import numpy as np
from dqn_slice_based.train import train_slice_dqn
from dqn_slice_based.evaluate import evaluate_volume_reconstruction

def main():
    parser = argparse.ArgumentParser(description='Slice-based DQN for 3D Volume Reconstruction')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                       help='Train or evaluate mode')
    parser.add_argument('--volume_path', type=str, required=True,
                       help='Path to .npy file containing 3D volume')
    parser.add_argument('--num_episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to saved model (for evaluation)')
    parser.add_argument('--save_dir', type=str, default='dqn_slice_based/models',
                       help='Directory to save models')
    parser.add_argument('--results_dir', type=str, default='dqn_slice_based/results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Starting training...")
        agent, env = train_slice_dqn(
            volume_path=args.volume_path,
            num_episodes=args.num_episodes,
            save_dir=args.save_dir,
            results_dir=args.results_dir,
            history_len = 1
        )
        print("Training completed!")
        
    elif args.mode == 'eval':
        if args.model_path is None:
            raise ValueError("--model_path required for evaluation mode")
        
        print("Starting evaluation...")
        evaluate_volume_reconstruction(
            volume_path=args.volume_path,
            model_path=args.model_path,
            results_dir=args.results_dir
        )
        print("Evaluation completed!")

if __name__ == '__main__':
    main()