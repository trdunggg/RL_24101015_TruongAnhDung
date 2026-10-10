"""Bài 20. Every-Visit MC Prediction"""
import gymnasium as gym

from mc_utils import every_visit_mc_prediction, stick_on_20_policy

env = gym.make("Blackjack-v1")
V, returns_count = every_visit_mc_prediction(env, stick_on_20_policy, n_episodes=10000,
                                             gamma=1.0, seed=0)

print("Số state ước lượng được:", len(V))
for state in sorted(V, key=lambda s: -returns_count[s])[:10]:
    print(f"  V{state} = {V[state]:+.3f}  (n={returns_count[state]})")
env.close()
