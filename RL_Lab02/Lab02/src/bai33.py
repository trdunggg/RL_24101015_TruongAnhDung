import gymnasium as gym
from mdp_utils import value_iteration, greedy_policy_from_value, print_frozenlake_policy

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
V, n_iterations, _ = value_iteration(env)
optimal_policy = greedy_policy_from_value(env, V)

print("Optimal state values:")
print(V.reshape(4, 4))
print("Optimal policy:")
print_frozenlake_policy(env, optimal_policy)
env.close()
