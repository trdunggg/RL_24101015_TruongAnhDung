import gymnasium as gym
from mdp_utils import policy_iteration

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
policy, V, n_policy_iterations = policy_iteration(env)
print("Policy:", policy)
print("V:", V)
print("Policy Iteration converged after", n_policy_iterations, "iterations.")
env.close()
