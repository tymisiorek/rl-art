# src/dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_channels, action_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)  # Output: 32x16x16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Output: 64x16x16
        self.fc1 = nn.Linear(64 * 16 * 16, 512)  # Adjusted for 16x16 canvas
        self.fc2 = nn.Linear(512, action_size)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class DQNAgent:
    def __init__(
        self, 
        state_size, 
        action_size, 
        lr=1e-3, 
        gamma=0.99, 
        epsilon_start=1.0, 
        epsilon_end=0.1, 
        epsilon_decay=1000,  # Increased decay for more exploration
        batch_size=32,
        memory_size=10000,    # Increased memory size
        target_update=10
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

        self.device = self._device()

        # Assuming the canvas size is 16x16
        self.policy_net = DQN(input_channels=1, action_size=action_size).to(self.device)
        self.target_net = DQN(input_channels=1, action_size=action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.target_update = target_update
        self.episode_count = 0

        self.loss_history = []  # Track loss

    def _device(self):
        return torch.device("cpu")  # Forces CPU usage

    def select_action(self, state):
        """
        Selects an action using epsilon-greedy policy.
        """
        self.steps_done += 1
        eps_threshold = self.epsilon_end + (self.epsilon - self.epsilon_end) * \
            np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.epsilon = max(self.epsilon_end, eps_threshold)

        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                # Convert state to tensor and add batch and channel dimensions
                state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)  # Shape: [1,1,16,16]
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """
        Stores a transition in the replay memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def sample_batch(self):
        """
        Samples a batch of transitions from memory.
        """
        return random.sample(self.memory, self.batch_size)

    def optimize_model(self):
        """
        Samples a batch from memory and performs a single optimization step.
        """
        if len(self.memory) < self.batch_size:
            return

        transitions = self.sample_batch()
        batch = list(zip(*transitions))

        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch[0])).unsqueeze(1).to(self.device)  # Shape: [batch, 1,16,16]
        action_batch = torch.LongTensor(batch[1]).unsqueeze(1).to(self.device)  # Shape: [batch,1]
        reward_batch = torch.FloatTensor(batch[2]).unsqueeze(1).to(self.device)  # Shape: [batch,1]
        next_state_batch = torch.FloatTensor(np.array(batch[3])).unsqueeze(1).to(self.device)  # Shape: [batch,1,16,16]
        done_batch = torch.FloatTensor(batch[4]).unsqueeze(1).to(self.device)  # Shape: [batch,1]

        # Compute current Q values
        current_q = self.policy_net(state_batch).gather(1, action_batch)  # Shape: [batch,1]

        # Double DQN: action selection is from policy_net, Q-value from target_net
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).argmax(1).unsqueeze(1)  # Shape: [batch,1]
            max_next_q = self.target_net(next_state_batch).gather(1, next_actions)  # Shape: [batch,1]
            target_q = reward_batch + (self.gamma * max_next_q * (1 - done_batch))  # Shape: [batch,1]

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        # Track loss
        self.loss_history.append(loss.item())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        """
        Updates the target network by copying weights from the policy network.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())
