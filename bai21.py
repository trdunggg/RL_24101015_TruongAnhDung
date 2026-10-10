"""Bài 21. So sánh First-Visit và Every-Visit (cùng policy, gamma, n_episodes, seed)"""
import gymnasium as gym

from mc_utils import every_visit_mc_prediction, first_visit_mc_prediction, stick_on_20_policy

env = gym.make("Blackjack-v1")
kwargs = dict(policy=stick_on_20_policy, n_episodes=50000, gamma=1.0, seed=0)
V_first, count_first = first_visit_mc_prediction(env, **kwargs)
V_every, count_every = every_visit_mc_prediction(env, **kwargs)

common = [s for s in V_first if s in V_every]
top10 = sorted(common, key=lambda s: -count_first[s])[:10]
print(f"{'state':<18}{'V_first':>10}{'V_every':>10}{'n_first':>9}{'n_every':>9}")
for s in top10:
    print(f"{str(s):<18}{V_first[s]:>+10.4f}{V_every[s]:>+10.4f}"
          f"{count_first[s]:>9}{count_every[s]:>9}")
# Nhận xét: trong Blackjack một state hầu như không lặp lại trong cùng episode
# nên first-visit và every-visit cho kết quả giống nhau (xem Bài 22).
env.close()
