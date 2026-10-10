"""Bài 35. Learning curve (moving average, window = 1000)"""
import matplotlib.pyplot as plt

from mc_utils import N_TRAIN, moving_average, save_figure, train_or_load_control

_, _, episode_rewards = train_or_load_control(n_episodes=N_TRAIN, epsilon=0.1)
window = 1000
curve = moving_average(episode_rewards, window)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(range(window, len(episode_rewards) + 1), curve)
ax.set_xlabel("Episode")
ax.set_ylabel(f"Mean reward (moving average, window={window})")
ax.set_title("On-policy MC control - learning curve (Blackjack-v1, epsilon=0.1)")
ax.grid(alpha=0.3)
save_figure(fig, "mc_convergence.png")
print(f"Mean reward {window} episode đầu : {sum(episode_rewards[:window]) / window:+.4f}")
print(f"Mean reward {window} episode cuối: {sum(episode_rewards[-window:]) / window:+.4f}")
