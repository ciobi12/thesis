import argparse
from dqn_slice_based.train import train_slice_dqn

def main():
    parser = argparse.ArgumentParser(description='Train Policy Network for Slice Reconstruction')
    parser.add_argument('--volume_path', type=str, required=True, help='Path to .npy volume file')
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--target_update', type=int, default=50, help='Target network update frequency')
    parser.add_argument('--save_interval', type=int, default=100, help='Model save frequency')
    parser.add_argument('--save_dir', type=str, default='dqn_slice_based/models', help='Model save directory')
    parser.add_argument('--results_dir', type=str, default='dqn_slice_based/results', help='Results save directory')
    parser.add_argument('--history_len', type=int, default=3, help='Number of previous slices in history')
    
    args = parser.parse_args()
    
    agent, env = train_slice_dqn(
        volume_path=args.volume_path,
        num_episodes=args.num_episodes,
        target_update=args.target_update,
        save_interval=args.save_interval,
        save_dir=args.save_dir,
        results_dir=args.results_dir,
        history_len=args.history_len
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main()