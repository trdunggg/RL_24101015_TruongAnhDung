import numpy as np
import matplotlib.pyplot as plt

rewards = [0, 0, 0, 0, 10]
gammas = np.linspace(0, 1, 101)

def compute_return(rewards, gamma):
    G = 0.0
    for reward in rewards:
        G = reward + gamma * G
    return G

returns = [compute_return(rewards, gamma) for gamma in gammas]
plt.plot(gammas, returns, label="G0")
plt.title("Ảnh hưởng của gamma đến G0")
plt.xlabel("Gamma")
plt.ylabel("G0")
plt.legend()
plt.grid()
plt.savefig("../figures/gamma_comparison.png", dpi=150)
plt.show()
