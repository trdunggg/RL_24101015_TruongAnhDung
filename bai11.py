"""Bài 11. Policy cố định: stick on 20"""
from mc_utils import BLACKJACK_ACTION_NAMES


def stick_on_20_policy(state):
    """Stick (0) nếu player_sum >= 20, ngược lại hit (1)."""
    player_sum, dealer_card, usable_ace = state

    if player_sum >= 20:
        return 0
    return 1


# Kiểm tra lại ý nghĩa action trước khi dùng: 0 = stick, 1 = hit (xem Bài 3).
for state in [(21, 5, False), (20, 10, False), (19, 10, False), (12, 6, True)]:
    action = stick_on_20_policy(state)
    print(f"state={state} -> action {action} = {BLACKJACK_ACTION_NAMES[action]}")
