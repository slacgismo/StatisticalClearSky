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
    """
    Class for time shift using signal processing to fix time shift.
    Adaptor (wrapper) class to eliminate the need to modify the code in
    IterativeFitting when underlying algorithm for time shift fix is changed.
    """

    def __init__(self, power_signals_d, weight=30, tolerance=5e-2,
                 solver_type=SolverType.ecos):
        """
        Arguments
        ---------
        power_signals_d : numpy array
            Representing a matrix with row for dates and colum for time of day,
            containing input power signals.

        Keyword arguments
        -----------------
        weight : integer
            Weight used in underlying algorithm.
        tolerance : float
            The difference under this value is considered not to have time
            shift.
        """
        super().__init__(power_signals_d)
        self._weight = weight
        self._tolerance = tolerance
        self._solver_type = solver_type

    def fix_time_shifts(self, verbose=False):
        """
        Adaptor (wrapper) method to eliminate the need to modify the code
        in IterativeFitting when underlying algorithm for time shift fix is
        changed.
        If the format of the input arguments of the underlying method changes,
        the translation from the arguments from IterativeFitting to those input
        arguments is performed in here.

        Keyword arguments
        -----------------
        verbose : boolean
            If True, verbose message is printed out.

        Returns
        -------
        numpy array
            Representing a matrix with row for dates and colum for time of day,
            containing power signals with fixed time shift.
        """
        return fix_time_shifts(self._power_signals_d, w=self._weight,
            tol=self._tolerance, solver_type=self._solver_type, verbose=False)
