import gymnasium as gym
import numpy as np
from mdp_utils import value_iteration_sweep

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
V = np.zeros(env.observation_space.n)
new_V = value_iteration_sweep(env, V, 0.99)
print(new_V)
env.close()
