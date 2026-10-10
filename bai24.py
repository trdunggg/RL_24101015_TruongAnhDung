"""Bài 24. Lưu (state, action) -> return"""
from collections import defaultdict

import gymnasium as gym

from mc_utils import compute_returns, generate_episode, make_random_policy

env = gym.make("Blackjack-v1")
policy = make_random_policy(env.action_space.n, seed=0)    # random để mọi action đều được thử

returns = defaultdict(list)          # (state, action) -> danh sách return
for i in range(20000):
    episode = generate_episode(env, policy, seed=0 if i == 0 else None)
    G = compute_returns([r for _, _, r in episode], gamma=1.0)
    for t, (state, action, _) in enumerate(episode):
        returns[(state, action)].append(G[t])

print("Số cặp (state, action) đã gặp:", len(returns))
for state in [(20, 10, False), (13, 2, False)]:
    for action in (0, 1):
        r = returns[(state, action)]
        print(f"{state}, action={action}: n={len(r)}")
env.close()
