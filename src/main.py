# src/main.py
import os
import matplotlib.pyplot as plt
import numpy as np

from environment import ArtEnv
from dqn_agent import DQNAgent
from utils import create_animation

# Ensure the folder for saved patterns exists
os.makedirs("data/saved_patterns", exist_ok=True)

# Initialize environment and agent
env = ArtEnv(size=16)  # Increased to 16x16 canvas
state_size = env.size * env.size  # 256
action_size = len(env.ACTIONS)  # Expanded action set
agent = DQNAgent(
    state_size=state_size,
    action_size=action_size,
    lr=5e-4,                # Lowered learning rate
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay=1000,     # Increased for more exploration
    batch_size=32,
    memory_size=10000,      # Increased memory size
    target_update=10
)

rewards = []
coverage_rewards = []
shape_rewards = []
losses = []
num_episodes = 500
max_steps_per_episode = 100

for episode in range(num_episodes):
    state = env.reset()  # Do NOT flatten the state
    total_reward = 0.0

    for t in range(max_steps_per_episode):
        # Choose an action
        action_index = agent.select_action(state)
        action = env.ACTIONS[action_index]  # Access ACTIONS from the instance

        # Step in the environment
        next_state, reward, done = env.step(action)

        # Store the transition in memory
        agent.store_transition(state, action_index, reward, next_state, done)

        # Move to the next state
        state = next_state
        total_reward += reward

        # Perform optimization
        agent.optimize_model()

        if done:
            break

    # Update the target network periodically
    if episode % agent.target_update == 0:
        agent.update_target_network()

    # Store total reward
    rewards.append(total_reward)

    # Log coverage and shape rewards
    coverage = np.sum(env.canvas) / (env.size * env.size)
    shape = env._compute_shape_measure()
    coverage_rewards.append(coverage)
    shape_rewards.append(shape)

    # Log loss
    if agent.loss_history:
        losses.append(agent.loss_history[-1])
    else:
        losses.append(0.0)

    # Save images periodically to visualize progress
    if episode % 25 == 0:
        filename = f"data/saved_patterns/episode_{episode}.png"
        env.render(title=f"Episode {episode}", filename=filename)

    # Print progress every 50 episodes
    if episode % 50 == 0:
        avg_loss = np.mean(agent.loss_history[-50:]) if len(agent.loss_history) >= 50 else np.mean(agent.loss_history)
        avg_reward = np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards)
        avg_coverage = np.mean(coverage_rewards[-50:]) if len(coverage_rewards) >= 50 else np.mean(coverage_rewards)
        avg_shape = np.mean(shape_rewards[-50:]) if len(shape_rewards) >= 50 else np.mean(shape_rewards)
        print(f"Episode {episode}: Avg Reward={avg_reward:.2f}, Avg Coverage={avg_coverage:.2f}, "
              f"Avg Shape={avg_shape:.2f}, Avg Loss={avg_loss:.4f}, Epsilon={agent.epsilon:.3f}")

# Plot reward and loss progress
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(rewards, label="Total Reward per Episode", alpha=0.6)
plt.plot(coverage_rewards, label="Coverage", alpha=0.6)
plt.plot(shape_rewards, label="Shape", alpha=0.6)
plt.xlabel("Episode")
plt.ylabel("Reward / Coverage / Shape")
plt.title("Training Progress (Coverage + Shape)")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(losses, label="Loss", color='red', alpha=0.6)
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("data/reward_progress_and_loss.png")
plt.show()

# Create animation
create_animation("data/saved_patterns", output_file="training_progress.gif")
print("Animation saved as training_progress.gif")
