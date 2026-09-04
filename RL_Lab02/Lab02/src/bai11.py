def compute_return(rewards, gamma):
    G = 0.0
    for reward in rewards:
        G = reward + gamma * G
    return G

sequence_A = [5, 0, 0, 0, 0]
sequence_B = [0, 0, 0, 0, 10]

for gamma in [i / 100 for i in range(101)]:
    a = compute_return(sequence_A, gamma)
    b = compute_return(sequence_B, gamma)
    if b > a:
        print("B > A tại gamma =", gamma)
