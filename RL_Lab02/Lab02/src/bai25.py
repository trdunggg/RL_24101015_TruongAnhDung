import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from mdp_utils import policy_evaluation_with_deltas

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
V, n_iterations, deltas = policy_evaluation_with_deltas(env, policy)

plt.plot(range(1, len(deltas) + 1), deltas, label="delta")
plt.title("Policy Evaluation Convergence")
plt.xlabel("Iteration")
plt.ylabel("Delta")
plt.legend()
plt.grid()
plt.savefig("../figures/policy_iteration_convergence.png", dpi=150)
plt.show()
print("Iterations:", n_iterations)
env.close()
