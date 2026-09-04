import gymnasium as gym
import numpy as np
from mdp_utils import greedy_policy_from_value

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
V = np.zeros(env.observation_space.n)
policy = greedy_policy_from_value(env, V)
print(policy)
env.close()
