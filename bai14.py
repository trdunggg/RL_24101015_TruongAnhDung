"""Bài 14. Ước lượng V(s) = trung bình return"""
from collections import defaultdict

import gymnasium as gym
import numpy as np

from mc_utils import compute_returns, generate_episode, stick_on_20_policy

env = gym.make("Blackjack-v1")
returns = defaultdict(list)
for i in range(10000):
    episode = generate_episode(env, stick_on_20_policy, seed=0 if i == 0 else None)
    G = compute_returns([r for _, _, r in episode], gamma=1.0)
    for t, (state, _, _) in enumerate(episode):
        returns[state].append(G[t])

V = {}
for state in returns:
    V[state] = np.mean(returns[state])

print("V(s) của 10 state xuất hiện nhiều nhất:")
for state in sorted(V, key=lambda s: -len(returns[s]))[:10]:
    print(f"  V{state} = {V[state]:+.3f}  (n={len(returns[state])})")
env.close()
