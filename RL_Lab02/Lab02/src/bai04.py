import numpy as np

P = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.5, 0.2],
    [0.2, 0.3, 0.5]
])

def state_distribution(p0, P, n_steps):
    p = np.array(p0, dtype=float)
    for _ in range(n_steps):
        p = p @ P
    return p

p0 = np.array([1.0, 0.0, 0.0])
for n in [1, 2, 5, 10, 50]:
    print("t =", n, state_distribution(p0, P, n))
