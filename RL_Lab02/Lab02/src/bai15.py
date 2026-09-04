import numpy as np

n_states = 2
n_actions = 2
policy = np.ones((n_states, n_actions)) / n_actions

print(policy)
print("Tổng xác suất action:", policy.sum(axis=1))
print("Hợp lệ:", np.allclose(policy.sum(axis=1), 1.0))
