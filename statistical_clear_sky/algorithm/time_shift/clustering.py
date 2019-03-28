"""
This module defines a subclass of AbstractTimeShifts,
which uses KDE-based clustering to perform time shifts.
"""
from statistical_clear_sky.algorithm.time_shift.abstract_time_shift\
import AbstractTimeShift
from solardatatools import fix_time_shifts

class ClusteringTimeShift(AbstractTimeShift):
    """
    Class for time shift using clustering to fix time shift.
    Adaptor (wrapper) class to eliminate the need to modify the code in
    IterativeFitting when underlying algorithm for time shift fix is changed.
    """

    def __init__(self, power_signals_d, return_ixs=False):
        """
        Arguments
        ---------
        power_signals_d : numpy array
            Representing a matrix with row for dates and column for time of day,
            containing input power signals.

        Keyword arguments
        -----------------
        return_ixs : boolean
            If it's set to True, index set is returned along with the power
            signals with fixed time shift.
        """
        super().__init__(power_signals_d)
        self._return_ixs = return_ixs

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
        verbose :boolean
            If True, verbose message is printed out.

        Returns
        -------
        numpy array
            Representing a matrix with row for dates and colum for time of day,
            containing power signals with fixed time shift.
        """
        return fix_time_shifts(self._power_signals_d,
            return_ixs=self._return_ixs, verbose=False)
