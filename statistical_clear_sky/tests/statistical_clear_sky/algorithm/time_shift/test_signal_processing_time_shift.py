import unittest
import os
import sys
import numpy as np
from statistical_clear_sky.algorithm.time_shift.signal_processing\
import SignalProcessingTimeShift

class TestSignalProcessingTimeShift(unittest.TestCase):

    def setUp(self):
        np.set_printoptions(threshold=sys.maxsize)
        self.maxDiff = None

    def test_fix_time_shifts(self):

        input_power_signals_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                         "../../fixtures/power_signals_d_1.csv"))
        with open(input_power_signals_file_path) as file:
            power_signals_d = np.loadtxt(file, delimiter=',')

        input_power_signals_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                     "../../fixtures/time_shifts/power_signals_d_fix_1.csv"))
        with open(input_power_signals_file_path) as file:
            expected_power_signals_d_fix = np.loadtxt(file, delimiter=',')

        time_shift = SignalProcessingTimeShift(
            power_signals_d, weight=30, tolerance=5e-2)
        actual_power_signals_d_fix = time_shift.fix_time_shifts()

        np.testing.assert_array_equal(actual_power_signals_d_fix,
                                      expected_power_signals_d_fix)
