import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from mdp_utils import value_iteration, policy_iteration, evaluate_policy_by_simulation

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)

start = perf_counter()
V, vi_iterations, deltas = value_iteration(env)
vi_time = perf_counter() - start
vi_policy = np.argmax(np.array([[
    sum(p * (r + (0 if term else 0.99 * V[ns]))
        for p, ns, r, term in env.unwrapped.P[s][a])
    for a in range(env.action_space.n)
] for s in range(env.observation_space.n)]), axis=1)
vi_result = evaluate_policy_by_simulation(env, vi_policy)

start = perf_counter()
pi_policy, pi_V, pi_iterations = policy_iteration(env)
pi_time = perf_counter() - start
pi_result = evaluate_policy_by_simulation(env, pi_policy)

print("Algorithm | Iterations | Time | Success rate | Mean reward")
print("Value Iteration |", vi_iterations, "|", vi_time, "|",
      vi_result["success_rate"], "|", vi_result["mean_reward"])
print("Policy Iteration |", pi_iterations, "|", pi_time, "|",
      pi_result["success_rate"], "|", pi_result["mean_reward"])

plt.bar(["Value Iteration", "Policy Iteration"], [vi_time, pi_time])
plt.title("Algorithm Runtime Comparison")
plt.xlabel("Algorithm")
plt.ylabel("Time (seconds)")
plt.grid(axis="y")
plt.savefig("../figures/algorithm_comparison.png", dpi=150)
plt.show()

print("Nhận xét:")
print("1. Cả hai thuật toán đều sử dụng model chuyển trạng thái của FrozenLake.")
print("2. Value Iteration cập nhật trực tiếp giá trị tối ưu.")
print("3. Policy Iteration luân phiên đánh giá và cải thiện policy.")
print("4. Số vòng lặp của hai thuật toán có thể khác nhau.")
print("5. Runtime phụ thuộc cấu hình máy và ngưỡng hội tụ.")
print("6. Với cùng môi trường, policy cuối thường cho kết quả tương đương.")
print("7. Success rate phụ thuộc tính ngẫu nhiên của FrozenLake.")
print("8. Có thể thay đổi gamma và theta để nghiên cứu thêm.")
env.close()
