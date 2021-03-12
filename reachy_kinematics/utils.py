import numpy as np

def minjerk(initial_position, goal_position, duration, initial_velocity=0, initial_acceleration=0, final_velocity=0, final_acceleration=0):
    a0 = initial_position
    a1 = initial_velocity
    a2 = initial_acceleration / 2

    d1, d2, d3, d4, d5 = [duration ** i for i in range(1, 6)]

    A = np.array((
        (d3, d4, d5),
        (3 * d2, 4 * d3, 5 * d4),
        (6 * d1, 12 * d2, 20 * d3)
    ))
    B = np.array((
        goal_position - a0 - (a1 * d1) - (a2 * d2),
        final_velocity - a1 - (2 * a2 * d1),
        final_acceleration - (2 * a2)
    ))
    X = np.linalg.solve(A, B)

    coeffs = [
        a0,
        a1,
        a2,
        X[0],
        X[1],
        X[2]
    ]

    t = np.linspace(0, duration, int(100 * duration))
    return np.sum([
        c * t ** i
        for i, c in enumerate(coeffs)
    ], axis=0)

