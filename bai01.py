"""Bài 1. Tạo Blackjack-v1"""
import gymnasium as gym

env = gym.make("Blackjack-v1")
observation, info = env.reset(seed=42)

print("observation      :", observation)
print("info             :", info)
print("observation_space:", env.observation_space)
print("action_space     :", env.action_space)
env.close()
