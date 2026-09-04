import numpy as np

P = [
    [
        [(1.0, 0, 1.0, False)],
        [(0.7, 1, 0.0, False), (0.3, 0, 0.5, False)]
    ],
    [
        [(0.4, 0, 0.0, False), (0.6, 1, 1.0, False)],
        [(1.0, 0, 2.0, True)]
    ]
]

for state in range(2):
    for action in range(2):
        print("state =", state, "action =", action, P[state][action])
