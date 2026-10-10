"""Bài 26. Greedy action từ Q"""
import gymnasium as gym
import numpy as np

from mc_utils import BLACKJACK_ACTION_NAMES, make_random_policy, mc_action_value_prediction


def greedy_action(Q, state, n_actions):
    """Trả về action có Q(s,a) lớn nhất bằng np.argmax; state chưa gặp -> 0."""
    if state not in Q:
        return 0
    return int(np.argmax(Q[state][:n_actions]))


env = gym.make("Blackjack-v1")
n_actions = env.action_space.n
Q = mc_action_value_prediction(env, make_random_policy(n_actions, seed=0),
                               n_episodes=200_000, gamma=1.0, seed=0)
for state in [(20, 10, False), (18, 6, False), (13, 2, False), (18, 6, True)]:
    a = greedy_action(Q, state, n_actions)
    print(f"state={state}: Q={np.round(Q[state], 3)} -> greedy = {BLACKJACK_ACTION_NAMES[a]}")
env.close()
