import unittest
import os
import sys
import numpy as np
from statistical_clear_sky.algorithm.utilities.time_shift\
 import fix_time_shifts

class TestTimeShift(unittest.TestCase):

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

        actual_power_signals_d_fix = fix_time_shifts(power_signals_d)

        np.testing.assert_array_equal(actual_power_signals_d_fix,
                                      expected_power_signals_d_fix)
