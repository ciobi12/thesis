import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Tuple

from dqn_slice_based.unet import SimpleEncoderDecoderPolicy


class ReplayBuffer:
    """Experience replay buffer for storing transitions"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class SlicePolicyAgent:
    """Policy Network Agent for slice reconstruction (outputs probabilities directly)"""
    
    def __init__(
        self,
        state_shape: Tuple[int, int, int],  # (C, H, W)
        learning_rate: float = 1e-4,
        gamma: float = 0.95,
        epsilon_start: float = 0.3,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.99,
        buffer_size: int = 50000,
        batch_size: int = 16,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            state_shape: Shape of state (n_channels, height, width)
                        n_channels = 1 (current) + history_len (previous predictions)
        """
        self.state_shape = state_shape
        self.n_channels = state_shape[0]
        self.height = state_shape[1]
        self.width = state_shape[2]
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.device = device
        
        # Policy network (outputs probabilities)
        self.policy_net = SimpleEncoderDecoderPolicy(n_channels=self.n_channels).to(device)
        
        # Target network for stability (optional, but helps)
        self.target_net = SimpleEncoderDecoderPolicy(n_channels=self.n_channels).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Loss functions
        self.bce_loss = nn.BCELoss(reduction='none')  # Per-pixel BCE
        self.mse_loss = nn.MSELoss()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select action map for entire slice.
        
        Args:
            state: Multi-channel array (C, H, W) where C = 1 + history_len
            deterministic: If True, always take best action (no exploration)
            
        Returns:
            action_map: 2D continuous array (H, W) with values in [0, 1]
        """
        if not deterministic and random.random() < self.epsilon:
            # Random action: random binary map
            return np.random.randint(0, 2, size=(self.height, self.width)).astype(np.float32)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, C, H, W)
            probs = self.policy_net(state_tensor)  # (1, H, W)
            action_map = probs.squeeze(0).cpu().numpy()  # (H, W)
            action_map = np.clip(action_map, 0, 1)
        
        return action_map.astype(np.float32)
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self) -> float:
        """Perform one training step using policy gradient + reconstruction loss"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)  # (B, C, H, W)
        actions = torch.FloatTensor(actions).to(self.device)  # (B, H, W)
        rewards = torch.FloatTensor(rewards).to(self.device)  # (B,)
        next_states = torch.FloatTensor(next_states).to(self.device)  # (B, C, H, W)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Forward pass: get predicted probabilities
        pred_probs = self.policy_net(states)  # (B, H, W)
        
        # LOSS 1: Reconstruction loss (BCE between prediction and action taken)
        # We want the network to predict the action that was taken
        reconstruction_loss = self.bce_loss(pred_probs, actions).mean()
        
        # LOSS 2: Reward-weighted policy gradient loss
        # Higher rewards should encourage the predicted probabilities
        with torch.no_grad():
            # Get value estimate from target network
            target_probs = self.target_net(next_states)  # (B, H, W)
            # Compute advantage (simplified)
            advantages = rewards.view(-1, 1, 1)  # (B, 1, 1)
        
        # Policy gradient: log prob * advantage
        log_probs = torch.log(pred_probs + 1e-8) * actions + torch.log(1 - pred_probs + 1e-8) * (1 - actions)
        policy_loss = -(log_probs * advantages).mean()
        
        # LOSS 3: Value loss (encourage accurate reward prediction)
        # Predict expected reward from current probabilities
        pred_reward = pred_probs.mean(dim=(1, 2))  # Simple reward estimate
        target_reward = rewards + (1 - dones) * self.gamma * target_probs.mean(dim=(1, 2))
        value_loss = self.mse_loss(pred_reward, target_reward)
        
        # Combined loss
        loss = reconstruction_loss + 0.5 * policy_loss + 0.1 * value_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str):
        """Save model"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']