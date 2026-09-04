import gymnasium as gym
import numpy as np
observations = []
for i in range(10):
    env = gym.make("CartPole-v1")
    observation, info = env.reset(seed=42)
    observations.append(observation)
    print(f"Environment {i + 1}: {observation}")
    env.close()
same = all(np.array_equal(observations[0], obs)
           for obs in observations[1:])
print("\nAll observations are the same:", same)
