from mdp_utils import validate_mdp

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

print(validate_mdp(P, 2, 2))
