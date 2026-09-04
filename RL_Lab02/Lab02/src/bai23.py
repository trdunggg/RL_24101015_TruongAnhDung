import gymnasium as gym
import numpy as np
from mdp_utils import policy_evaluation_sweep

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
V = np.zeros(env.observation_space.n)
print(policy_evaluation_sweep(env, policy, V, 0.99))
env.close()
