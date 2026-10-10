"""Bài 27. Epsilon-greedy policy"""
import numpy as np

from mc_utils import epsilon_greedy_action

# epsilon_greedy_action (mc_utils.py):
#   - xác suất epsilon    : chọn ngẫu nhiên đều trong mọi action (khám phá)
#   - xác suất 1 - epsilon: chọn greedy action theo Q (khai thác)
rng = np.random.default_rng(0)
Q = {"s": np.array([0.2, 0.8])}          # state giả "s", action 1 là tốt nhất
for epsilon in (0.0, 0.5, 1.0):
    actions = [epsilon_greedy_action(Q, "s", 2, epsilon, rng) for _ in range(10)]
    print(f"epsilon={epsilon}: {actions}")
