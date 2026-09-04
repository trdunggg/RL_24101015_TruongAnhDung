import gymnasium as gym
import numpy as np
from mdp_utils import q_from_v

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
V = np.zeros(env.observation_space.n)
for action in range(env.action_space.n):
    print("Q(0,", action, ") =", q_from_v(env, V, 0, action, 0.99))
env.close()
