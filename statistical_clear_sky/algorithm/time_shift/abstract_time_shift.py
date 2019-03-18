'''
This module defines abstract class for time shift.
'''
from abc import abstractmethod

class AbstractTimeShift():

    @abstractmethod
    def fix_time_shifts(verbose=False):
        pass
