import gymnasium as gym
import numpy as np
env = gym.make("CartPole-v1")
episode_rewards = []
episode_lengths = []
for episode in range(100):
    observation, info = env.reset(seed=episode)
    total_reward = 0.0
    length = 0
    while True:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        length += 1
        if terminated or truncated:
            break
    episode_rewards.append(total_reward)
    episode_lengths.append(length)
best_episode = np.argmax(episode_rewards)
best_reward = episode_rewards[best_episode]
best_length = episode_lengths[best_episode]
print("Best episode:", best_episode + 1)
print("Best reward:", best_reward)
print("Best length:", best_length)
env.close()