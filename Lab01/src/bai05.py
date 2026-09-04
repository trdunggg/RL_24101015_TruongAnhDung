import gymnasium as gym 
import numpy as np
env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42)
print("Observation:", observation)
print("Type:", type(observation))
print("Shape:", observation.shape)
print("Info:", info)
env.close()