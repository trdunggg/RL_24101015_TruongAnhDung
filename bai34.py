"""Bài 34. Đánh giá policy: random vs fixed vs MC learned"""
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from mc_utils import (N_TRAIN, evaluate_policy, make_policy_from_dict, make_random_policy,
                      save_figure, stick_on_20_policy, train_or_load_control)

_, learned_dict, _ = train_or_load_control(n_episodes=N_TRAIN, epsilon=0.1)
learned_policy = make_policy_from_dict(learned_dict)       # chỉ dùng greedy, KHÔNG explore

env = gym.make("Blackjack-v1")
policies = {
    "random": make_random_policy(env.action_space.n, seed=7),
    "fixed (stick on 20)": stick_on_20_policy,
    "MC learned": learned_policy,
}
n_eval = 100_000      # episode MỚI, seed khác lúc train
results = {name: evaluate_policy(env, pol, n_episodes=n_eval, seed=123)
           for name, pol in policies.items()}
env.close()

print(f"{'policy':<22}{'win':>8}{'loss':>8}{'draw':>8}{'mean reward':>14}")
for name, r in results.items():
    print(f"{name:<22}{r['win_rate']:>8.3f}{r['loss_rate']:>8.3f}{r['draw_rate']:>8.3f}"
          f"{r['mean_reward']:>+10.4f} ± {r['std_error']:.4f}")

names = list(results)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].bar(names, [results[n]["mean_reward"] for n in names],
            yerr=[results[n]["std_error"] for n in names], capsize=4)
axes[0].set_ylabel("Mean reward / episode")
axes[0].set_title("Mean reward")
axes[0].axhline(0, color="k", linewidth=0.8)
x = np.arange(len(names))
for i, key in enumerate(["win_rate", "draw_rate", "loss_rate"]):
    axes[1].bar(x + (i - 1) * 0.25, [results[n][key] for n in names], 0.25,
                label=key.replace("_rate", ""))
axes[1].set_xticks(x)
axes[1].set_xticklabels(names)
axes[1].set_title("Win / Draw / Loss rate")
axes[1].legend()
fig.tight_layout()
save_figure(fig, "policy_performance.png")
