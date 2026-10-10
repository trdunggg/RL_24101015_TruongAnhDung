"""Bài 19. Visualize V(s) của Blackjack (heatmap player_sum x dealer_card)"""
import gymnasium as gym

from mc_utils import (blackjack_grid, first_visit_mc_prediction, plot_blackjack_grid,
                      stick_on_20_policy)

env = gym.make("Blackjack-v1")
V, _ = first_visit_mc_prediction(env, stick_on_20_policy, n_episodes=200_000, seed=0)

for usable_ace, name in [(False, "no_usable_ace"), (True, "usable_ace")]:
    grid = blackjack_grid(V, usable_ace)
    if not usable_ace:                       # in bảng text đơn giản cho usable_ace = False
        print("V(s), usable_ace=False  (hàng: player_sum 12..21, cột: dealer A,2..10)")
        for ps, row in zip(range(12, 22), grid):
            print(f"{ps:2d} | " + " ".join(f"{v:+.2f}" for v in row))
    plot_blackjack_grid(grid, f"V(s) - stick on 20 - usable_ace={usable_ace}",
                        f"blackjack_value_{name}.png", vmin=-1, vmax=1)
env.close()
