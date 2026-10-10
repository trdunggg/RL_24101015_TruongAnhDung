"""Bài 28. Kiểm tra epsilon-greedy với Q giả = [1.0, 5.0], epsilon = 0.1"""
from collections import Counter

import numpy as np

from mc_utils import epsilon_greedy_action

rng = np.random.default_rng(0)
Q = {"s": np.array([1.0, 5.0])}
epsilon, n = 0.1, 10000

counts = Counter(epsilon_greedy_action(Q, "s", 2, epsilon, rng) for _ in range(n))
print(f"action 0: {counts[0]} ({counts[0] / n:.3f})")
print(f"action 1 (greedy): {counts[1]} ({counts[1] / n:.3f})")
# Nhận xét: kỳ vọng P(greedy) = 1 - epsilon + epsilon/2 = 0.95, P(action 0) = epsilon/2 = 0.05.
print("Lý thuyết: P(greedy) =", 1 - epsilon + epsilon / 2)
