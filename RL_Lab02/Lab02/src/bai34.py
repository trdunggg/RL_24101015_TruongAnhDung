import gymnasium as gym
import numpy as np
from mdp_utils import evaluate_policy_by_simulation, value_iteration, policy_iteration

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
random_policy = np.zeros(env.observation_space.n, dtype=int)
V, _, _ = value_iteration(env)
value_policy = np.argmax(
    np.array([[
        sum(p * (r + (0 if term else 0.99 * V[ns]))
            for p, ns, r, term in env.unwrapped.P[s][a])
        for a in range(env.action_space.n)
    ] for s in range(env.observation_space.n)]), axis=1
)
policy_policy, _, _ = policy_iteration(env)

for name, policy in [
    ("Random", random_policy),
    ("Value Iteration", value_policy),
    ("Policy Iteration", policy_policy)
]:
    result = evaluate_policy_by_simulation(env, policy, 1000, 42)
    print(name, result)

env.close()
