import gymnasium as gym
env = gym.make("CartPole-v1")
env.action_space.seed(42)
actions_1 = []
for i in range(20):
    action = env.action_space.sample()
    actions_1.append(action)
env.action_space.seed(42)
actions_2 = []
for i in range(20):
    action = env.action_space.sample()
    actions_2.append(action)
print("Actions lan 1:")
print(actions_1)
print("\nActions lan 2:")
print(actions_2)
print("\nHai danh sach giong nhau:", actions_1 == actions_2)
env.close()