import gymnasium as gym
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
print("Episode rewards:")
print(episode_rewards)
env.close()