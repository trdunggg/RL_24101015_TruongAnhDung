import gymnasium as gym
import numpy as np
from mdp_utils import action_values

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
V = np.zeros(env.observation_space.n)
print(action_values(env, V, 0, 0.99))
env.close()
