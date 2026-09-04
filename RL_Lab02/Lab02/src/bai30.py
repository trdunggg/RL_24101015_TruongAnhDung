import gymnasium as gym
from mdp_utils import policy_iteration, action_values

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
policy, V, n = policy_iteration(env)

stable = True
for state in range(env.observation_space.n):
    best_action = int(action_values(env, V, state, 0.99).argmax())
    if best_action != policy[state]:
        stable = False

print("Policy stable:", stable)
print("Iterations:", n)
env.close()
