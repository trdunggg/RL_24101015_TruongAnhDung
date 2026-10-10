"""Bài 31. Incremental mean"""
from collections import defaultdict

import numpy as np

from mc_utils import update_action_value_incremental

# update_action_value_incremental (mc_utils.py):
#   N(s,a) = N(s,a) + 1;  Q(s,a) = Q(s,a) + (G - Q(s,a)) / N(s,a)
Q = defaultdict(lambda: np.zeros(2))
N = defaultdict(lambda: np.zeros(2))
state, action = (15, 10, False), 1

rng = np.random.default_rng(0)
returns = rng.choice([-1.0, 0.0, 1.0], size=1000)
for G in returns:
    update_action_value_incremental(Q, N, state, action, G)

print("Incremental mean:", Q[state][action])
print("np.mean         :", returns.mean())
assert np.isclose(Q[state][action], returns.mean())
print("-> Hai kết quả bằng nhau, nhưng incremental mean không cần lưu toàn bộ return.")
