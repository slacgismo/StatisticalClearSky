"""
This module defines abstract class for time shift.
"""
from abc import abstractmethod

class AbstractTimeShift():
    """
    Abstract class for time shift.
    Adaptor (wrapper) class to eliminate the need to modify the code in
    IterativeFitting when underlying algorithm for time shift fix is changed.
    """

    def __init__(self, power_signals_d):
        """
        Arguments
        ---------
        power_signals_d : numpy array
            Representing a matrix with row for dates and column for time of day,
            containing input power signals.
        """
        self._power_signals_d = power_signals_d

    @abstractmethod
    def fix_time_shifts(verbose=False):
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
        pass
