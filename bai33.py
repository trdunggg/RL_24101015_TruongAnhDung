"""Bài 33. Học policy cho Blackjack (>= 100000 episode)"""
import numpy as np

from mc_utils import (BLACKJACK_ACTION_NAMES, N_TRAIN, blackjack_grid, plot_blackjack_grid,
                      train_or_load_control)

# Train (kết quả được cache trong data/ để các bài 34, 35 dùng lại).
Q, policy, episode_rewards = train_or_load_control(n_episodes=N_TRAIN, epsilon=0.1)
print(f"Đã train {N_TRAIN} episode, policy có {len(policy)} state.")

# Policy được học từ Q (không hard-code): action = argmax_a Q(s, a).
for state in [(20, 10, False), (18, 6, False), (13, 2, False), (18, 6, True)]:
    action = int(np.argmax(Q[state]))
    print(f"state={state}: Q={np.round(Q[state], 3)} -> policy = {BLACKJACK_ACTION_NAMES[action]}")

# Hình minh họa policy (0 = stick, 1 = hit)
for usable_ace, name in [(False, "no_usable_ace"), (True, "usable_ace")]:
    grid = blackjack_grid(policy, usable_ace)
    plot_blackjack_grid(grid, f"Learned policy (0=stick, 1=hit), usable_ace={usable_ace}",
                        f"blackjack_policy_{name}.png", cmap="coolwarm", vmin=0, vmax=1,
                        colorbar_label="action")
