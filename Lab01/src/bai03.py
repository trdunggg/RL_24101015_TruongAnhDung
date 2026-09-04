import gymnasium as gym 
env = gym.make("CartPole-v1")
print("Action sapce:" , env.action_space)
print("Number of action:" , env.action_space.n)
env.close()