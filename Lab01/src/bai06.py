import gymnasium as gym
env = gym.make("CartPole-v1")
actions = []
for i in range(20):
    action = env.action_space.sample()
    actions.append(action)
print("Danh sach 20 action:")
print(actions)
print("\nTan suat xuat hien:")
for action in sorted(set(actions)):
    count = actions.count(action)
    print(f"Action {action}: {count} lan")
env.close()