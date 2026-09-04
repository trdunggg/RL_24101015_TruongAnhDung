import numpy as np

policy = np.array([0, 1])

def print_policy(policy):
    for state, action in enumerate(policy):
        print("State", state, "-> Action", action)

print_policy(policy)
