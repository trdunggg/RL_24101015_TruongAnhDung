import gymnasium as gym
import numpy as np
def run_experiment(seed, num_episodes=20):
    env = gym.make("CartPole-v1")
    rewards = []
    for episode in range(num_episodes):
        observation, info = env.reset(seed=seed + episode)
        total_reward = 0.0
        while True:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)
    env.close()
    return rewards
rewards_42 = run_experiment(42)
rewards_100 = run_experiment(100)
mean_42 = np.mean(rewards_42)
mean_100 = np.mean(rewards_100)
print("Seed 42")
print("Rewards:", rewards_42)
print(f"Mean reward: {mean_42:.2f}")
print("\nSeed 100")
print("Rewards:", rewards_100)
print(f"Mean reward: {mean_100:.2f}")
print("\nComparison:")
if mean_42 > mean_100:
    print("Seed 42 has a higher mean reward.")
elif mean_100 > mean_42:
    print("Seed 100 has a higher mean reward.")
else:
    print("Both seeds have the same mean reward.")