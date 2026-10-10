"""Bài 4. Một episode ngẫu nhiên (random policy)"""
import gymnasium as gym

from mc_utils import BLACKJACK_ACTION_NAMES

env = gym.make("Blackjack-v1")
env.action_space.seed(0)

state, _ = env.reset(seed=0)
done, step = False, 0
while not done:
    action = env.action_space.sample()             # random policy
    next_state, reward, terminated, truncated, _ = env.step(action)
    step += 1
    print(f"Bước {step}: state={state}, action={BLACKJACK_ACTION_NAMES[action]}, "
          f"reward={reward}, next_state={next_state}, "
          f"terminated={terminated}, truncated={truncated}")
    state = next_state
    done = terminated or truncated
env.close()
