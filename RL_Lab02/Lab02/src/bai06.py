import numpy as np

P = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.5, 0.2],
    [0.2, 0.3, 0.5]
])
rng = np.random.default_rng(42)
state = 0
counts = np.zeros(3, dtype=int)

for _ in range(100000):
    state = int(rng.choice(3, p=P[state]))
    counts[state] += 1

simulation = counts / counts.sum()
theory = np.array([1.0, 0.0, 0.0]) @ np.linalg.matrix_power(P, 1000)

print("Simulation:", simulation)
print("Theory after 1000 steps:", theory)
print("Difference:", np.abs(simulation - theory))
print("Nhận xét: kết quả mô phỏng gần với phân phối lý thuyết ở trạng thái ổn định.")
print("Sai khác nhỏ xuất hiện do lấy mẫu ngẫu nhiên.")
print("Số transition càng lớn thì tần suất mô phỏng thường càng ổn định.")
