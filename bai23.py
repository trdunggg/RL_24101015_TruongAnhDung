"""Bài 23. Biểu đồ hội tụ First-Visit vs Every-Visit"""
import gymnasium as gym
import matplotlib.pyplot as plt

from mc_utils import (make_random_policy, mc_prediction_with_history, save_figure,
                      stick_on_20_policy)

checkpoints = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]


def run_both(env, policy_factory, states, gamma, n_episodes=50000):
    """Chạy first/every-visit với cùng seed và policy; trả về history của các state."""
    out = {}
    for name, first in [("First-Visit", True), ("Every-Visit", False)]:
        _, _, hist = mc_prediction_with_history(
            env, policy_factory(), n_episodes, gamma, first_visit=first, seed=0,
            track_states=states, checkpoints=checkpoints)
        out[name] = hist
    return out


# --- Hàng 1: Blackjack, 3 state ---
bj = gym.make("Blackjack-v1")
bj_states = [(20, 10, False), (18, 6, False), (13, 2, False)]
bj_hist = run_both(bj, lambda: stick_on_20_policy, bj_states, gamma=1.0)

# --- Hàng 2: FrozenLake, 3 state hay gặp nhất (chọn bằng một lần chạy thử) ---
fl = gym.make("FrozenLake-v1")
fl_policy = lambda: make_random_policy(fl.action_space.n, seed=1)
_, counts, _ = mc_prediction_with_history(fl, fl_policy(), 2000, 0.95, seed=99)
fl_states = sorted(counts, key=lambda s: -counts[s])[:3]
fl_hist = run_both(fl, fl_policy, fl_states, gamma=0.95)

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for row, (title, states, hist) in enumerate([("Blackjack", bj_states, bj_hist),
                                             ("FrozenLake", fl_states, fl_hist)]):
    for col, s in enumerate(states):
        ax = axes[row, col]
        ax.plot(checkpoints, hist["First-Visit"][s], marker="o", label="First-Visit")
        ax.plot(checkpoints, hist["Every-Visit"][s], marker="x", linestyle="--",
                label="Every-Visit")
        ax.set_xscale("log")
        ax.set_title(f"{title}: V{s}")
        ax.set_xlabel("Số episode")
        ax.set_ylabel("Estimate V(s)")
        ax.grid(alpha=0.3)
        ax.legend()
fig.suptitle("First-Visit vs Every-Visit MC (Blackjack: trùng nhau; FrozenLake: khác nhau)")
fig.tight_layout()
save_figure(fig, "first_vs_every_visit.png")
bj.close()
fl.close()
