import numpy as np

P = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.5, 0.2],
    [0.2, 0.3, 0.5]
])

def sample_next_state(current_state, P, rng):
    return int(rng.choice(len(P), p=P[current_state]))

rng = np.random.default_rng(42)
state = 0
states = [state]
for _ in range(30):
    state = sample_next_state(state, P, rng)
    states.append(state)
print(states)
