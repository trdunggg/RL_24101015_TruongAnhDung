import gymnasium as gym
import numpy as np
def experiment(seed, n_episodes):
    env = gym.make("CartPole-v1")
    episode_rewards = []
    env.action_space.seed(seed)
    for episode in range(n_episodes):
        observation, info = env.reset(seed=seed + episode)
        total_reward = 0.0
        for step in range(500):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        episode_rewards.append(total_reward)
    env.close()
    return {
        "seed": seed,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "min_reward": np.min(episode_rewards)
    }
seeds = [42, 100, 200, 300, 400]
for seed in seeds:
    result = experiment(seed, 100)
    print(result)