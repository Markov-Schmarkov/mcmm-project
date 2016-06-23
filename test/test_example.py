r"""
This test script should illustrate how unit tests should be written. Here, we basically just check
whether mcmm.example.generate_test_data produces the correct number of independent trajectories and
that all trajectories have the correct length, dimension, and floating point type.
"""

import mcmm
import numpy as np
from nose.tools import assert_true

def check_trajs(trajs, traj_length, num_trajs):
    r"""
    This function handles the checks that we need in all other tests.
    """
    assert_true(isinstance(trajs, list))
    assert_true(len(trajs) == num_trajs)
    for traj in trajs:
        assert_true(isinstance(traj, np.ndarray))
        assert_true(traj.dtype == np.float64)
        assert_true(traj.ndim == 2)
        assert_true(traj.shape[0] == traj_length)
        assert_true(traj.shape[1] == 2)

def test_interface_default():
    r"""
    Does the default setting produce the expected data?
    """
    trajs = mcmm.example.generate_test_data()
    check_trajs(trajs, 20000, 5)

def test_interface_custom():
    r"""
    Does a custom setting produce the requested data?
    """
    traj_length = 10000
    num_trajs = 7
    trajs = mcmm.example.generate_test_data(traj_length, num_trajs)
    check_trajs(trajs, traj_length, num_trajs)

def test_interface_random():
    r"""
    Does a randomly chosen setting produce the requested data?
    """
    traj_length = np.random.randint(500, 25000)
    num_trajs = np.random.randint(3, 9)
    trajs = mcmm.example.generate_test_data(traj_length, num_trajs)
    check_trajs(trajs, traj_length, num_trajs)
