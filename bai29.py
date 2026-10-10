"""Bài 29. So sánh epsilon: tần suất chọn greedy action + learning curve"""
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from mc_utils import epsilon_greedy_action, moving_average, on_policy_mc_control, save_figure

epsilons = [0.01, 0.05, 0.10, 0.20, 0.50]
rng = np.random.default_rng(0)
Q = {"s": np.array([1.0, 5.0])}

# (a) Tần suất chọn greedy action với Q giả
freq = []
for eps in epsilons:
    actions = [epsilon_greedy_action(Q, "s", 2, eps, rng) for _ in range(10000)]
    freq.append(np.mean(np.array(actions) == 1))
    print(f"epsilon={eps:<5}: tần suất greedy = {freq[-1]:.3f}  (lý thuyết {1 - eps + eps / 2:.3f})")

# (b) Ảnh hưởng của epsilon lên quá trình học (MC control, 100k episode mỗi epsilon)
env = gym.make("Blackjack-v1")
curves = {}
for eps in epsilons:
    _, _, rewards = on_policy_mc_control(env, 100_000, epsilon=eps, seed=42)
    curves[eps] = moving_average(rewards, window=1000)
env.close()

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].bar([str(e) for e in epsilons], freq)
axes[0].plot([str(e) for e in epsilons], [1 - e + e / 2 for e in epsilons],
             "r.--", label="lý thuyết")
axes[0].set_xlabel("epsilon")
axes[0].set_ylabel("Tần suất chọn greedy action")
axes[0].set_ylim(0, 1.05)
axes[0].legend()
for eps, curve in curves.items():
    axes[1].plot(curve, label=f"epsilon={eps}")
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Mean reward (moving avg, window=1000)")
axes[1].set_title("Learning curve theo epsilon (reward lúc train, có exploration)")
axes[1].legend()
axes[1].grid(alpha=0.3)
fig.tight_layout()
save_figure(fig, "epsilon_comparison.png")
