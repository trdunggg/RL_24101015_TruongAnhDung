import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
def moving_average(values, window_size):
    result = []
    for i in range(len(values)):
        start = max(0, i - window_size + 1)
        window = values[start:i + 1]
        result.append(np.mean(window))
    return np.array(result)
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
window_size = 10
moving_avg = moving_average(episode_rewards, window_size)
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, label="Raw reward")
plt.plot(moving_avg, label="Moving average")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward and Moving Average")
plt.legend()
plt.grid(True)
plt.savefig("Lab01/figures/moving_average.png", dpi=150)
plt.show()