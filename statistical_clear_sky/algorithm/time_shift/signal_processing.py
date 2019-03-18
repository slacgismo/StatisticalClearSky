"""
This module defines a subclass of AbstractTimeShifts,
which uses signal processing to perform time shifts.
"""

import numpy as np
import cvxpy as cvx
from statistical_clear_sky.algorithm.time_shift.abstract_time_shift\
import AbstractTimeShift
from statistical_clear_sky.solver_type import SolverType
from statistical_clear_sky.algorithm.utilities.time_shift import fix_time_shifts

class SignalProcessingTimeShift(AbstractTimeShift):

    def __init__(self, power_signals_d, weight=30, tolerance=5e-2,
                 solver_type=SolverType.ecos):
        self._power_signals_d = power_signals_d
        self._weight = weight
        self._tolerance = tolerance
        self._solver_type = solver_type

    def fix_time_shifts(self, verbose=False):
        return fix_time_shifts(self._power_signals_d, w=self._weight,
            tol=self._tolerance, solver_type=self._solver_type, verbose=False)
