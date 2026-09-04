import gymnasium as gym
import numpy as np
env = gym.make("CartPole-v1")
episode_rewards = []
for episode in range(100):
    observation, info = env.reset(seed=episode)
    total_reward = 0.0
    while True:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    episode_rewards.append(total_reward)
mean_reward = np.mean(episode_rewards)
min_reward = np.min(episode_rewards)
max_reward = np.max(episode_rewards)
median_reward = np.median(episode_rewards)
std_reward = np.std(episode_rewards)
print(f"Mean reward: {mean_reward:.2f}")
print(f"Min reward: {min_reward:.2f}")
print(f"Max reward: {max_reward:.2f}")
print(f"Median reward: {median_reward:.2f}")
print(f"Std reward: {std_reward:.2f}")
env.close()