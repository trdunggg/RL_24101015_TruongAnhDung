import gymnasium as gym
from mdp_utils import value_iteration

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
V, n_iterations, deltas = value_iteration(env)
print("V =", V)
print("Iterations =", n_iterations)
print("Last delta =", deltas[-1])
env.close()
