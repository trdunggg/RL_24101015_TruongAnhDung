import gymnasium as gym
env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42)
state_before = observation
action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)
print("State before action:", state_before)
print("Action:", action)
print("State after action:", observation)
print("Reward:", reward)
print("Terminated:", terminated)
print("Trundcated:", truncated)
print("Info:", info)
env.close