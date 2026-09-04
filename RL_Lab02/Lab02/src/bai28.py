import gymnasium as gym
import numpy as np
from mdp_utils import policy_evaluation, greedy_policy_from_value

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
old_policy = np.zeros(env.observation_space.n, dtype=int)
V, _ = policy_evaluation(env, np.eye(env.action_space.n)[old_policy])
new_policy = greedy_policy_from_value(env, V)

print("Old policy:", old_policy)
print("New policy:", new_policy)
print("Changed states:", np.sum(old_policy != new_policy))
env.close()
