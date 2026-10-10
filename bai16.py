"""Bài 16. Phát hiện first visit của mỗi state trong một episode"""
import gymnasium as gym

from mc_utils import first_visit_indices, generate_episode, make_random_policy, stick_on_20_policy

# first_visit_indices(states) (trong mc_utils.py) duyệt episode theo thứ tự thời gian
# và ghi lại chỉ số t đầu tiên của mỗi state - không dùng thư viện RL nào.

# 1) Blackjack: player_sum chỉ tăng nên state gần như không lặp lại trong một episode.
bj = gym.make("Blackjack-v1")
episode = generate_episode(bj, stick_on_20_policy, seed=5)
states = [s for s, _, _ in episode]
print("Blackjack states:", states)
print("First-visit idx :", first_visit_indices(states))
bj.close()

# 2) FrozenLake (random policy): state lặp lại nhiều lần -> first/every visit khác nhau.
fl = gym.make("FrozenLake-v1")
policy = make_random_policy(fl.action_space.n, seed=0)
for seed in range(50):                                   # tìm một episode có state lặp lại
    episode = generate_episode(fl, policy, seed=seed)
    states = [s for s, _, _ in episode]
    if len(set(states)) < len(states):
        break
first = first_visit_indices(states)
print("\nFrozenLake states      :", states)
print("First-visit idx        :", first)
for s in first:
    all_idx = [t for t, x in enumerate(states) if x == s]
    print(f"  state {s}: first-visit t={first[s]}, every-visit t={all_idx}")
fl.close()
