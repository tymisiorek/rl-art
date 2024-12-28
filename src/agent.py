import numpy as np

class QLearningAgent:
    def __init__(
        self, 
        state_size, 
        action_size, 
        alpha=0.05, 
        gamma=0.9, 
        epsilon=1.0, 
        epsilon_decay=0.999
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha        # Learning rate
        self.gamma = gamma        # Discount factor
        self.epsilon = epsilon    # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.q_table = {}

    def get_action(self, state):
        """
        Epsilon-greedy strategy:
        - With probability epsilon, pick a random action.
        - Otherwise, pick the action with max Q-value.
        """
        state_key = self._state_to_key(state)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        return np.argmax(self.q_table.get(state_key, np.zeros(self.action_size)))

    def update_q_value(self, state, action, reward, next_state):
        """
        Q-learning update rule:
            Q(s,a) <- Q(s,a) + alpha * [reward + gamma * max Q(s',a') - Q(s,a)]
        """
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)

        current_q = self.q_table[state_key][action]
        best_next_q = np.max(self.q_table[next_state_key])

        # Q-learning formula
        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * best_next_q)
        self.q_table[state_key][action] = new_q

    def decay_epsilon(self):
        """Slowly reduce exploration, down to a minimum of 0.1."""
        self.epsilon = max(0.1, self.epsilon * self.epsilon_decay)

    def _state_to_key(self, state):
        """
        Flatten the 2D canvas into a 1D tuple for dictionary key usage.
        """
        return tuple(state.flatten())
