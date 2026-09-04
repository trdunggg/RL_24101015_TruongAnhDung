import gymnasium as gym 
env = gym.make("CartPole-v1")
observation_space = env.observation_space
print("Observation space", observation_space)
print("Shape:", observation_space.shape)
print("Data type:", observation_space.dtype)
print("Lower  bound:", observation_space.low)
print("Upper bound", observation_space.high)
env.close()