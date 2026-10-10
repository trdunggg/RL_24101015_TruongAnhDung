"""Bài 8. Kiểm tra return với episode giả"""
import numpy as np

from mc_utils import compute_returns

rewards = [0, 0, 1]          # R1=0, R2=0, R3=1
for gamma in (1.0, 0.9, 0.5):
    G = compute_returns(rewards, gamma)
    # Kết quả mong đợi: [gamma^2, gamma, 1]
    expected = [gamma ** 2, gamma, 1.0]
    assert np.allclose(G, expected), (G, expected)
    print(f"gamma={gamma}: G = {G}  (đúng với [gamma^2, gamma, 1])")
