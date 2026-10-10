"""Bài 7. Tính return từ cuối episode"""
import gymnasium as gym

from mc_utils import compute_returns, generate_episode, make_random_policy

# compute_returns duyệt ngược: G_t = R_{t+1} + gamma * G_{t+1}  (xem mc_utils.py)
env = gym.make("Blackjack-v1")
episode = generate_episode(env, make_random_policy(env.action_space.n, seed=0), seed=1)
rewards = [reward for _, _, reward in episode]

for gamma in (1.0, 0.9):
    print(f"gamma={gamma}: rewards={rewards} -> returns={compute_returns(rewards, gamma)}")
env.close()
