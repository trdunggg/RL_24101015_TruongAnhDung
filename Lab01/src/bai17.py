import gymnasium as gym
import matplotlib.pyplot as plt
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
env.close()
plt.figure(figsize=(10, 5))
plt.plot(range(1, 101), episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward per Episode")
plt.grid(True)
plt.savefig("Lab01/figures/reward.png", dpi=150)
plt.show()