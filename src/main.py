import os
import matplotlib.pyplot as plt

from environment import SymmetryEnv
from agent import QLearningAgent
from utils import create_animation

# Ensure the folder for saved patterns exists
os.makedirs("data/saved_patterns", exist_ok=True)

# Initialize environment and agent
env = SymmetryEnv(size=32)  # Larger canvas for more interesting designs
agent = QLearningAgent(
    state_size=32 * 32, 
    action_size=len(SymmetryEnv.ACTIONS),
    alpha=0.05,        # Learning rate
    gamma=0.9,         # Discount factor
    epsilon=1.0,       # Start fully exploring
    epsilon_decay=0.995
)

rewards = []
num_episodes = 100
max_steps_per_episode = 50

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0.0

    for t in range(max_steps_per_episode):
        # Choose an action
        action_index = agent.get_action(state)
        action = SymmetryEnv.ACTIONS[action_index]

        # Take the action in the environment
        next_state, reward, done = env.step(action)

        # Update Q-values
        agent.update_q_value(state, action_index, reward, next_state)

        state = next_state
        total_reward += reward
        
        if done:
            break

    # Store total episode reward
    rewards.append(total_reward)

    # Decay exploration rate
    agent.decay_epsilon()

    # Save a PNG of the environment every 10 episodes
    if episode % 10 == 0:
        filename = f"data/saved_patterns/episode_{episode}.png"
        env.render(title=f"Episode {episode}", filename=filename)

# Plot reward progress
plt.figure()
plt.plot(rewards, label="Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward Progress")
plt.legend()
plt.savefig("data/reward_progress.png")
plt.show()

# Create an animation from the saved PNGs
create_animation("data/saved_patterns", output_file="training_progress.gif")
print("Animation saved as training_progress.gif")
