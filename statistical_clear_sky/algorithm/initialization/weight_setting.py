"""
This module defines a class for Weight Setting Algorithm.
"""
import numpy as np
import cvxpy as cvx
from statistical_clear_sky.solver_type import SolverType
from solardatatools import find_clear_days

class WeightSetting(object):
    """
    Delegate class.
    Weight Setting Algorithm:
    Two metrics are calculated and normalized to the interval [0, 1],
    and then the geometric mean is taken.
    Metric 1: daily smoothness
    Metric 2: seasonally weighted daily energy
    After calculating the geometric mean of these two values, weights below
    """

    def __init__(self, solver_type=SolverType.ecos):
        self._solver_type = solver_type

    def obtain_weights(self, power_signals_d):
        weights = find_clear_days(power_signals_d, boolean_out=False)
        return weights
