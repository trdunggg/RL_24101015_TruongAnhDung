"""Bài 13. Thu thập return theo state (chưa phân biệt first/every visit)"""
from collections import defaultdict

import gymnasium as gym

from mc_utils import compute_returns, generate_episode, stick_on_20_policy

env = gym.make("Blackjack-v1")
returns = defaultdict(list)          # state -> danh sách return quan sát được

for i in range(1000):
    episode = generate_episode(env, stick_on_20_policy, seed=0 if i == 0 else None)
    G = compute_returns([r for _, _, r in episode], gamma=1.0)
    for t, (state, _, _) in enumerate(episode):
        returns[state].append(G[t])  # lưu return tại MỌI lần state xuất hiện

print("Số state đã gặp:", len(returns))
for state in sorted(returns, key=lambda s: -len(returns[s]))[:5]:
    print(f"state={state}: {len(returns[state])} return, 5 đầu = {returns[state][:5]}")
env.close()
