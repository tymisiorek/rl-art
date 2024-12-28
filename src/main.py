import os
import matplotlib.pyplot as plt

from environment import SymmetryEnv
from agent import QLearningAgent
from utils import create_animation

# Ensure the folder for saved patterns exists
os.makedirs("data/saved_patterns", exist_ok=True)

# Initialize environment and agent
env = SymmetryEnv(size=32)
agent = QLearningAgent(state_size=32*32, action_size=len(SymmetryEnv.ACTIONS))

rewards = []
num_episodes = 50
max_steps_per_episode = 20

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    for t in range(max_steps_per_episode):
        action_index = agent.get_action(state)  
        action = SymmetryEnv.ACTIONS[action_index]  # Retrieve the corresponding action dict
        next_state, reward, done = env.step(action)

        # Update Q-learning
        agent.update_q_value(state, action_index, reward, next_state)
        state = next_state
        total_reward += reward
        
        if done:
            break

    rewards.append(total_reward)
    agent.decay_epsilon()

    # Save a PNG of the environment every 10 episodes
    if episode % 10 == 0:
        filename = f"data/saved_patterns/episode_{episode}.png"
        env.render(title=f"Episode {episode}", filename=filename)

# Plot reward progress
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward Progress")
plt.savefig("data/reward_progress.png")
plt.show()

# Create an animation from the saved PNGs
create_animation("data/saved_patterns", output_file="training_progress.gif")
print("Animation saved as training_progress.gif")
