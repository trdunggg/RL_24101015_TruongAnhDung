import gymnasium as gym
env = gym.make(
    "FrozenLake-v1",
    is_slippery=False
)
print(env.observation_space)
print(env.action_space)
print("Number of states:", env.observation_space.n)
print("Number of actions:", env.action_space.n)
env.close()