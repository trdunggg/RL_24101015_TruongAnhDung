import gymnasium as gym

for slippery in [False, True]:
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=slippery)
    transitions = env.unwrapped.P[0][2]
    print("is_slippery =", slippery)
    print("Number of transitions:", len(transitions))
    for t in transitions:
        print(t)
    env.close()

print("Kết luận: deterministic thường có một transition, còn stochastic có thể có nhiều transition với các xác suất khác nhau.")
