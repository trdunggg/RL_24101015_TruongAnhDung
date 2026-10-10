"""Bài 22. Sai khác giữa First-Visit và Every-Visit"""
import gymnasium as gym
import numpy as np

from mc_utils import (every_visit_mc_prediction, first_visit_mc_prediction,
                      make_random_policy, stick_on_20_policy)


def mean_abs_difference(V_first, V_every):
    """Mean |V_first[s] - V_every[s]| trên các state chung."""
    common = [s for s in V_first if s in V_every]
    diffs = [abs(V_first[s] - V_every[s]) for s in common]
    return float(np.mean(diffs)), common


# --- Blackjack ---
env = gym.make("Blackjack-v1")
kw = dict(n_episodes=50000, gamma=1.0, seed=0)
V_f, _ = first_visit_mc_prediction(env, stick_on_20_policy, **kw)
V_e, _ = every_visit_mc_prediction(env, stick_on_20_policy, **kw)
mad, common = mean_abs_difference(V_f, V_e)
print(f"Blackjack : {len(common)} state chung, mean |V_first - V_every| = {mad:.6f}")
env.close()

# --- FrozenLake (state có lặp lại trong episode nên hai phương pháp khác nhau) ---
fl = gym.make("FrozenLake-v1")
kw = dict(n_episodes=20000, gamma=0.95, seed=0)
V_f, _ = first_visit_mc_prediction(fl, make_random_policy(4, seed=1), **kw)
V_e, _ = every_visit_mc_prediction(fl, make_random_policy(4, seed=1), **kw)
mad, common = mean_abs_difference(V_f, V_e)
print(f"FrozenLake: {len(common)} state chung, mean |V_first - V_every| = {mad:.6f}")
for s in sorted(common)[:6]:
    print(f"  state {s:2d}: first={V_f[s]:.4f}, every={V_e[s]:.4f}, |diff|={abs(V_f[s]-V_e[s]):.4f}")
fl.close()
