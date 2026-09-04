import gymnasium as gym
env = gym.make(
    "FrozenLake-v1",
    is_slippery=False
)
ACTION_NAMES = {
    0: "LEFT",
    1: "DOWN",
    2: "RIGHT",
    3: "UP"
}
action = env.action_space.sample()
print(f"Action {action} -> {ACTION_NAMES[action]}")
env.close()