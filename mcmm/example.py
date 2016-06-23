r"""
This module generates trajectories of a simple two dimensional toy model for testing purposes.
"""

import numpy as np

__all__ = ['generate_test_data']

def _gradient(x, y):
    return (x * x - 1.0) * 4.0 * x + 0.5, (4.0 * y * y - 7.0) * y

def _bd(x0, y0, length, dt=0.005):
    coeff_A = dt
    coeff_B = np.sqrt(2.0 * dt)
    x = [x0]
    y = [y0]
    for _i in range(1, length):
        dx, dy = _gradient(x[-1], y[-1])
        x.append(x[-1] - coeff_A * dx + coeff_B * np.random.normal())
        y.append(y[-1] - coeff_A * dy + coeff_B * np.random.normal())
    return np.array([[_x, _y] for _x, _y in zip(x, y)], dtype=np.float64)

def generate_test_data(traj_length=20000, num_trajs=5):
    r"""
    This functions handles the test data generation via Brownian dynamics simulations with
    randomized starting configurations.

    Parameters
    ----------
    traj_length : int, optional, default=20000
        Length of a single trajectory.
    num_trajs : int, optional, default=5
        Number of independent trajectories.

    Returns
    -------
    trajs : list of numpy.ndarray(shape=(traj_length, 2), dtype=numpy.float64) objects
        Time series of configurations of the toy model.
    """
    trajs = []
    for _i in range(num_trajs):
        trajs.append(_bd(3.0 * np.random.rand() - 1.5, 3.0 * np.random.rand() - 1.5, traj_length))
    return trajs
