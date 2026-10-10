"""Bài 2. Phân tích observation"""
import gymnasium as gym

env = gym.make("Blackjack-v1")

# observation = (player_sum, dealer_card, usable_ace)
#   player_sum  : tổng điểm bài hiện tại của người chơi (Ace tính 11 nếu không bị quá 21).
#   dealer_card : lá bài dealer đang ngửa (1 = Ace, 2..10; J/Q/K tính là 10).
#   usable_ace  : người chơi có Ace đang được tính là 11 hay không
#                 (Gymnasium trả về 0/1 nên in ra dạng bool cho dễ đọc).
observations = []
for episode_id in range(12):                       # 12 episode khác nhau
    obs, _ = env.reset(seed=episode_id)
    observations.append(obs)

for i, obs in enumerate(observations, start=1):
    player_sum, dealer_card, usable_ace = obs
    print(f"{i:2d}. obs={obs} -> player_sum={player_sum:2d}, "
          f"dealer_card={dealer_card:2d}, usable_ace={bool(usable_ace)}")
env.close()
