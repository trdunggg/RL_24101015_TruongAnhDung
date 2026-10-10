"""Bài 6. Chuỗi reward"""
import gymnasium as gym

from mc_utils import generate_episode, make_random_policy

env = gym.make("Blackjack-v1")
episode = generate_episode(env, make_random_policy(env.action_space.n, seed=0), seed=1)

rewards = [reward for _, _, reward in episode]
print("Episode length:", len(episode))
print("Rewards       :", rewards)
print("Total reward  :", sum(rewards))
env.close()
