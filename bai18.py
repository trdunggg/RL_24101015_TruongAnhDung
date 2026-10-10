"""Bài 18. Chạy First-Visit MC với 100 / 1000 / 10000 / 50000 episode"""
import gymnasium as gym

from mc_utils import first_visit_mc_prediction, stick_on_20_policy

env = gym.make("Blackjack-v1")
for n_episodes in [100, 1000, 10000, 50000]:
    V, counts = first_visit_mc_prediction(env, stick_on_20_policy, n_episodes, seed=0)
    print(f"{n_episodes:6d} episode -> {len(V):3d} state được ước lượng, "
          f"tổng số lần cập nhật = {sum(counts.values())}")
env.close()
