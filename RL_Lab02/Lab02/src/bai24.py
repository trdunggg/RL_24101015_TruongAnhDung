import gymnasium as gym
import numpy as np
from mdp_utils import policy_evaluation

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
V, n_iterations = policy_evaluation(env, policy, gamma=0.99, theta=1e-8)
print("V =", V)
print("Iterations =", n_iterations)
env.close()
