"""Bài 32. On-policy First-Visit MC Control"""
import gymnasium as gym

from mc_utils import on_policy_mc_control

# Quy trình (xem on_policy_mc_control trong mc_utils.py):
# khởi tạo Q -> sinh episode bằng epsilon-greedy -> tính return ->
# first-visit update Q(s,a) -> cập nhật policy -> lặp lại.
env = gym.make("Blackjack-v1")
Q, policy, episode_rewards = on_policy_mc_control(env, n_episodes=50000, gamma=1.0,
                                                  epsilon=0.1, seed=42)
print("Số state có Q   :", len(Q))
print("Số state có policy:", len(policy))
print("Số episode_rewards:", len(episode_rewards))
print("Mean reward 1000 episode cuối (lúc train):", sum(episode_rewards[-1000:]) / 1000)
env.close()
