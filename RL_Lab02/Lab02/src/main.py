import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from mdp_utils import (
    value_iteration,
    policy_iteration,
    evaluate_policy_by_simulation,
    greedy_policy_from_value,
    print_frozenlake_policy,
)

def create_environment(is_slippery=True, map_name="4x4"):
    return gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery)

def main():
    gamma = 0.99
    theta = 1e-8
    max_iterations = 10000
    n_episodes = 1000

    env = create_environment(is_slippery=True)

    start = perf_counter()
    V_vi, vi_iterations, vi_deltas = value_iteration(
        env, gamma, theta, max_iterations
    )
    vi_runtime = perf_counter() - start
    vi_policy = greedy_policy_from_value(env, V_vi, gamma)

    start = perf_counter()
    pi_policy, V_pi, pi_iterations = policy_iteration(
        env, gamma, theta, 1000
    )
    pi_runtime = perf_counter() - start

    vi_result = evaluate_policy_by_simulation(env, vi_policy, n_episodes)
    pi_result = evaluate_policy_by_simulation(env, pi_policy, n_episodes)

    print("Value Iteration")
    print("V =", V_vi.reshape(4, 4))
    print_frozenlake_policy(env, vi_policy)
    print(vi_result)
    print("Runtime:", vi_runtime)

    print("\nPolicy Iteration")
    print("V =", V_pi.reshape(4, 4))
    print_frozenlake_policy(env, pi_policy)
    print(pi_result)
    print("Runtime:", pi_runtime)

    plt.plot(range(1, len(vi_deltas) + 1), vi_deltas, label="Value Iteration")
    plt.title("Dynamic Programming Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Delta")
    plt.legend()
    plt.grid()
    plt.savefig("../figures/value_iteration_convergence.png", dpi=150)
    plt.show()

    print("\nComparison")
    print("Value Iteration:", vi_iterations, vi_runtime,
          vi_result["success_rate"], vi_result["mean_reward"])
    print("Policy Iteration:", pi_iterations, pi_runtime,
          pi_result["success_rate"], pi_result["mean_reward"])

    env.close()

if __name__ == "__main__":
    main()
