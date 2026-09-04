import gymnasium as gym 
def run_one_step(env, action):
    observation, reward, terminated, truncated, info = env.step(action)
    return observation, reward, terminated, truncated, info 
env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42)
action = env.action_space.sample()
result = run_one_step(env, action)
print("Observation:", result[0])
print("Reward:", result[1])
print("Terminated:", result[2])
print("Truncated:", result[3])
print("Info:", result[4])
env.close