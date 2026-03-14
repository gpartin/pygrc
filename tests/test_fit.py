import pygrc
import numpy as np
import math


def function(x, *args):
    return args[0] * x + args[1]


# Least-squares fit test against synthetic data
def test_fit_lsq():
    rng = np.random.default_rng(42)

    x = np.linspace(0, 10, 10)
    noise = rng.random(10)
    y = 2 * x + noise

    m = pygrc.Fit(x, y, 1, 1).fit_lsq(
        function,
        [(-1, 10), (-2, 10)],
        0.1
    )

    assert math.fabs(m.values["x0"] - 2.0) < 0.1, \
        "Least Square slope (x0) did not converge to true value"

    assert math.fabs(m.values["x1"] - 0.5) < 1.0, \
        "Least Square intercept (x1) not within reasonable range"