import gymnasium as gym
env = gym.make("CartPole-v1")
for episode in range(1, 11):
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
    print(f"{episode:<8} | {total_reward:<6.1f} | {length}")

env.close()