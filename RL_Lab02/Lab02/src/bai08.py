def compute_return(rewards, gamma):
    total = 0.0
    for reward in rewards:
        total = total * gamma + reward
    return total

rewards = [1, 1, 1, 1, 1]
for gamma in [0.0, 0.5, 0.9, 0.99, 1.0]:
    print(f"{gamma:.2f} {compute_return(rewards, gamma):.6f}")
