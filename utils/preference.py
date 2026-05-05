import numpy as np


def circle_points(r, n):
    """Generate evenly distributed preference vectors for two tasks.

    Samples n points on the arc of radius r in the first quadrant (0 to pi/2),
    spanning the full trade-off spectrum between the two objectives.
    """
    circles = []
    for r_val, n_val in zip(r, n):
        t = np.linspace(0, 0.5 * np.pi, n_val)
        x = r_val * np.cos(t)
        y = r_val * np.sin(t)
        circles.append(np.c_[x, y])
    return circles
