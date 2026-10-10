"""Bài 30. Khởi tạo Q, returns_sum, returns_count"""
from collections import defaultdict

import gymnasium as gym
import numpy as np

env = gym.make("Blackjack-v1")
n_actions = env.action_space.n

Q = defaultdict(lambda: np.zeros(env.action_space.n))
returns_sum = defaultdict(lambda: np.zeros(n_actions))      # tổng return theo (s, a)
returns_count = defaultdict(lambda: np.zeros(n_actions))    # số lần gặp (s, a)

print("Q ban đầu của một state bất kỳ:", Q[(15, 10, False)])
print("returns_sum:", returns_sum[(15, 10, False)], "| returns_count:", returns_count[(15, 10, False)])
env.close()
