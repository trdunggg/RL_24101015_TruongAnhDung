"""Bài 15. Theo dõi estimate V(s) của một state cụ thể khi tăng số episode"""
from collections import defaultdict

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from mc_utils import compute_returns, generate_episode, save_figure, stick_on_20_policy

env = gym.make("Blackjack-v1")
target_state = (20, 10, False)
checkpoints = [100, 500, 1000, 5000, 10000]

returns = defaultdict(list)
estimates = []
for ep in range(1, max(checkpoints) + 1):
    episode = generate_episode(env, stick_on_20_policy, seed=0 if ep == 1 else None)
    G = compute_returns([r for _, _, r in episode], gamma=1.0)
    for t, (state, _, _) in enumerate(episode):
        returns[state].append(G[t])
    if ep in checkpoints:
        est = np.mean(returns[target_state]) if returns[target_state] else np.nan
        estimates.append(est)
        print(f"Sau {ep:5d} episode: V{target_state} ≈ {est:+.4f} "
              f"(đã gặp {len(returns[target_state])} lần)")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(checkpoints, estimates, marker="o")
ax.set_xscale("log")
ax.set_xlabel("Số episode")
ax.set_ylabel(f"Estimate V{target_state}")
ax.set_title("Bài 15: Convergence curve của một state")
ax.grid(alpha=0.3)
save_figure(fig, "v_target_state_convergence.png")
env.close()
