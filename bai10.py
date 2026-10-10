"""Bài 10. So sánh gamma: mean G_0 theo gamma"""
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from mc_utils import compute_returns, generate_episode, make_random_policy, save_figure

env = gym.make("Blackjack-v1")
policy = make_random_policy(env.action_space.n, seed=0)    # cùng một policy cho mọi gamma
n_episodes = 5000

# Sinh episode một lần, rồi tính return với từng gamma để so sánh công bằng.
all_rewards = [[r for _, _, r in generate_episode(env, policy, seed=0 if i == 0 else None)]
               for i in range(n_episodes)]

gammas = [0.5, 0.8, 0.9, 0.99, 1.0]
mean_g0 = []
for gamma in gammas:
    g0 = [compute_returns(rewards, gamma)[0] for rewards in all_rewards]
    mean_g0.append(float(np.mean(g0)))
    print(f"gamma={gamma:<5}: mean G_0 = {mean_g0[-1]:.4f}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(gammas, mean_g0, marker="o")
ax.set_xlabel("gamma")
ax.set_ylabel("mean G_0")
ax.set_title("Bài 10: mean G_0 theo gamma (random policy, Blackjack)")
ax.grid(alpha=0.3)
save_figure(fig, "gamma_comparison.png")
env.close()
