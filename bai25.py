"""Bài 25. Ước lượng Q(s,a) bằng MC"""
import gymnasium as gym

from mc_utils import BLACKJACK_ACTION_NAMES, make_random_policy, mc_action_value_prediction

env = gym.make("Blackjack-v1")
# Dùng random policy để cả hai action đều được thử ở mọi state (Q[state][action]).
Q = mc_action_value_prediction(env, make_random_policy(env.action_space.n, seed=0),
                               n_episodes=200_000, gamma=1.0, seed=0)

print("Số state có Q:", len(Q))
for state in [(20, 10, False), (18, 6, False), (13, 2, False), (18, 6, True)]:
    q = Q[state]
    print(f"Q{state}: stick={q[0]:+.3f}, hit={q[1]:+.3f}")
env.close()
