"""Bài 12. Sinh 100 episode với policy stick-on-20 và thống kê win/loss/draw"""
import gymnasium as gym

from mc_utils import generate_episode, stick_on_20_policy

env = gym.make("Blackjack-v1")
n_episodes = 100
win = loss = draw = 0

for i in range(n_episodes):
    episode = generate_episode(env, stick_on_20_policy, seed=0 if i == 0 else None)
    total_reward = sum(r for _, _, r in episode)
    if total_reward > 0:
        win += 1
    elif total_reward < 0:
        loss += 1
    else:
        draw += 1

print(f"win : {win:3d} ({win / n_episodes:.1%})")
print(f"loss: {loss:3d} ({loss / n_episodes:.1%})")
print(f"draw: {draw:3d} ({draw / n_episodes:.1%})")
env.close()
