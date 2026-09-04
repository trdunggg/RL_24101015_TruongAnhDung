import gymnasium as gym
env = gym.make(
    "FrozenLake-v1",
    render_mode="ansi",
    is_slippery=False
)
observation, info = env.reset()
print(env.render())
env.close()