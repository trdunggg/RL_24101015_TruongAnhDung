def compute_return(rewards, gamma):
    total = 0.0
    for reward in rewards:
        total = total * gamma + reward
    return total

rewards = [1, 1, 1, 1, 1]
print(compute_return(rewards, 1.0))
