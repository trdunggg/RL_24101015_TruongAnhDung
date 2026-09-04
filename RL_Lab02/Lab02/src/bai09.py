def discounted_returns(rewards, gamma):
    returns = [0.0] * len(rewards)
    G = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        G = rewards[i] + gamma * G
        returns[i] = G
    return returns

rewards = [0, 0, 0, 1]
print(discounted_returns(rewards, 0.9))
