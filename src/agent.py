import numpy as np

class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = {}

    def get_action(self, state):
        """Epsilon-greedy strategy."""
        state_key = self._state_to_key(state)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)  # Explore
        return np.argmax(self.q_table.get(state_key, np.zeros(self.action_size)))  # Exploit

    def update_q_value(self, state, action, reward, next_state):
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)

        best_next_action = np.max(self.q_table[next_state_key])
        self.q_table[state_key][action] = (
            (1 - self.alpha) * self.q_table[state_key][action]
            + self.alpha * (reward + self.gamma * best_next_action)
        )

    def decay_epsilon(self):
        self.epsilon = max(0.1, self.epsilon * self.epsilon_decay)

    def _state_to_key(self, state):
        """Convert a 2D array into a hashable tuple for Q-table storage."""
        return tuple(state.flatten())
