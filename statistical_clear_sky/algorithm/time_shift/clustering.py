"""
This module defines a subclass of AbstractTimeShifts,
which uses KDE-based clustering to perform time shifts.
"""
from statistical_clear_sky.algorithm.time_shift.abstract_time_shift\
import AbstractTimeShift
from solardatatools import fix_time_shifts

class ClusteringTimeShift(AbstractTimeShift):

    def __init__(self, power_signals_d, return_ixs=False):
        self._power_signals_d = power_signals_d
        self._return_ixs = return_ixs

    def fix_time_shifts(self, verbose=False):
        return fix_time_shifts(self._power_signals_d,
            return_ixs=self._return_ixs, verbose=False)
