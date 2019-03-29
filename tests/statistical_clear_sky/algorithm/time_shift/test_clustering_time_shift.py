'''
Unit test should be written in solar-data-tools project.
This test is written in order to verify the impact of switching from
signal processing based time shifts to clustering based time shifts.
'''
import unittest
import os
import sys
import numpy as np
import cvxpy as cvx
from statistical_clear_sky.algorithm.time_shift.clustering\
import ClusteringTimeShift

class TestClusteringTimeShift(unittest.TestCase):

    def setUp(self):
        np.set_printoptions(threshold=sys.maxsize)
        self.maxDiff = None

    def test_fix_time_shifts(self):

        input_power_signals_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                     "../../fixtures/time_shifts",
                     "one_year_power_signals_d_1.csv"))
        with open(input_power_signals_file_path) as file:
            power_signals_d = np.loadtxt(file, delimiter=',')

        output_power_signals_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
            "../../fixtures/time_shifts/power_signals_d_fix_clustering_1.csv"))
        with open(output_power_signals_file_path) as file:
            expected_power_signals_d_fix = np.loadtxt(file, delimiter=',')

        time_shift = ClusteringTimeShift(power_signals_d)

        # Underling solar-data-tools uses MOSEK solver and
        # if it's not used, try with ECOS solver.
        # However, fails with ECOS solver and raises cvx.SolverError.
        try:
            actual_power_signals_d_fix = time_shift.fix_time_shifts()
        except (cvx.SolverError, ValueError):
            self.skipTest("This test uses MOSEK solver"
                + "because default ECOS solver fails with large data. "
                + "Unless MOSEK is installed, this test fails.")
        else:
            np.testing.assert_array_equal(actual_power_signals_d_fix,
                                          expected_power_signals_d_fix)
