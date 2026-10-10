"""Bài 9. Gắn return vào trajectory: (state, action, reward, G_t)"""
import gymnasium as gym

from mc_utils import compute_returns, generate_episode, make_random_policy

env = gym.make("Blackjack-v1")
episode = generate_episode(env, make_random_policy(env.action_space.n, seed=0), seed=3)

returns = compute_returns([r for _, _, r in episode], gamma=1.0)
episode_with_return = [(s, a, r, G) for (s, a, r), G in zip(episode, returns)]

for t, step in enumerate(episode_with_return):
    print(f"t={t}: (state, action, reward, G_t) = {step}")
env.close()
