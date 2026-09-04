import gymnasium as gym
env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42) 
while True:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    episode_finished = terminated or truncated
    if episode_finished:
        if terminated:
            print("Termination")
        elif truncated:
            print("Truncation")
        break
env.close            