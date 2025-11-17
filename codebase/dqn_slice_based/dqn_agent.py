import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Tuple

from dqn_slice_based.unet import SimpleEncoderDecoderDQN

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


class SliceDQNAgent:
    """DQN Agent for slice reconstruction using multi-channel encoder-decoder"""
    
    def __init__(
        self,
        state_shape: Tuple[int, int, int],  # Now (C, H, W) instead of (H, W)
        n_actions: int = 2,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 32,
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
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.device = device
        
        # Networks (now with multi-channel input)
        self.policy_net = SimpleEncoderDecoderDQN(
            n_channels=self.n_channels, 
            n_actions=n_actions
        ).to(device)
        self.target_net = SimpleEncoderDecoderDQN(
            n_channels=self.n_channels, 
            n_actions=n_actions
        ).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select action map for entire slice.
        
        Args:
            state: Multi-channel array (C, H, W) where C = 1 + history_len
            deterministic: If True, always take best action (no exploration)
            
        Returns:
            action_map: 2D binary array (H, W)
        """
        # Epsilon-greedy for exploration
        if not deterministic and random.random() < self.epsilon:
            # Random action: random binary map
            return np.random.randint(0, 2, size=(self.height, self.width)).astype(np.float32)
        
        # Greedy action from Q-network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, C, H, W)
            q_values = self.policy_net(state_tensor)  # (1, n_actions, H, W)
            
            # Select action with highest Q-value for each pixel
            action_map = torch.argmax(q_values, dim=1).squeeze(0).cpu().numpy()
            
        return action_map.astype(np.float32)
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self) -> float:
        """Perform one training step"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)  # (B, C, H, W)
        actions = torch.LongTensor(actions).to(self.device)  # (B, H, W)
        rewards = torch.FloatTensor(rewards).to(self.device)  # (B,)
        next_states = torch.FloatTensor(next_states).to(self.device)  # (B, C, H, W)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.policy_net(states)  # (B, n_actions, H, W)
        
        # Gather Q values for taken actions
        actions_expanded = actions.unsqueeze(1)  # (B, 1, H, W)
        current_q = current_q_values.gather(1, actions_expanded).squeeze(1)  # (B, H, W)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states)  # (B, n_actions, H, W)
            max_next_q = next_q_values.max(dim=1)[0]  # (B, H, W)
            
            # Target: reward + gamma * max_next_q (if not done)
            target_q = rewards.view(-1, 1, 1) + (1 - dones.view(-1, 1, 1)) * self.gamma * max_next_q
        
        # Compute loss
        loss = self.criterion(current_q, target_q)
        
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