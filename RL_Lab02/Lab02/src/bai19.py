import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
valid = True

for state in range(env.observation_space.n):
    for action in range(env.action_space.n):
        probabilities = [t[0] for t in env.unwrapped.P[state][action]]
        if not np.isclose(sum(probabilities), 1.0):
            valid = False
            print("Invalid:", state, action)

print("All transitions valid:", valid)
env.close()
