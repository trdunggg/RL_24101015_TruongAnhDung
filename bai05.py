"""Bài 5. Lưu episode: generate_episode(env, policy, seed)"""
import gymnasium as gym

from mc_utils import generate_episode, make_random_policy   # generate_episode nằm trong mc_utils.py

env = gym.make("Blackjack-v1")
random_policy = make_random_policy(env.action_space.n, seed=0)

# Mỗi phần tử của episode: (state, action, reward)
episode = generate_episode(env, random_policy, seed=1)
print("Số bước:", len(episode))
for t, (state, action, reward) in enumerate(episode):
    print(f"t={t}: state={state}, action={action}, reward={reward}")
env.close()
