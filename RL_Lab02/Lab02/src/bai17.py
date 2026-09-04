import gymnasium as gym

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
P = env.unwrapped.P

for action in range(env.action_space.n):
    print("Action:", action)
    for probability, next_state, reward, terminated in P[0][action]:
        print("Probability:", probability,
              "Next state:", next_state,
              "Reward:", reward,
              "Terminated:", terminated)
env.close()
