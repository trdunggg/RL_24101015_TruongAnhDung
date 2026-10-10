"""Bài 3. Phân tích action"""
import gymnasium as gym

env = gym.make("Blackjack-v1")
print("action_space:", env.action_space, "| số action:", env.action_space.n)

# Action của Blackjack-v1: 0 = stick (dừng, không rút thêm), 1 = hit (rút thêm một lá).
ACTION_NAMES = {
    0: "stick (dừng)",
    1: "hit (rút bài)",
}
for action_id, name in ACTION_NAMES.items():
    print(f"action {action_id} = {name}")
env.close()
