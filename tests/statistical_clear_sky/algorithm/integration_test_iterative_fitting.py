import unittest
import os
import numpy as np
import cvxpy as cvx
from statistical_clear_sky.algorithm.iterative_fitting import IterativeFitting
from statistical_clear_sky.solver_type import SolverType

class IntegrationTestIterativeFitting(unittest.TestCase):

    def test_execute(self):

        input_power_signals_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                         "../fixtures/power_signals_d_1.csv"))
        with open(input_power_signals_file_path) as file:
            power_signals_d = np.loadtxt(file, delimiter=',')

        rank_k = 6

        clear_sky_signals_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                         "../fixtures/clear_sky_signals_result_1.csv"))
        with open(clear_sky_signals_file_path) as file:
            expected_clear_sky_signals = np.loadtxt(file, delimiter=',')
        expected_degradation_rate = np.array(-0.04215127)

        try: # try block for solver usage at initialization.
            iterative_fitting = IterativeFitting(power_signals_d, rank_k=rank_k,
                                                 solver_type=SolverType.mosek)
            iterative_fitting.execute(mu_l=5e2, mu_r=1e3, tau=0.9,
                                      max_iteration=10)
        except cvx.SolverError:
            self.skipTest("This test uses MOSEK solver"
                + "because default ECOS solver fails with large data. "
                + "Unless MOSEK is installed, this test fails.")
        else:
            # Handling solver error coming from minimization.
            if iterative_fitting.state_data.is_solver_error is True:
                self.skipTest("This test uses MOSEK solver"
                    + "because default ECOS solver fails with large data. "
                    + "Unless MOSEK is installed, this test fails.")
            else:
                actual_clear_sky_signals = iterative_fitting.clear_sky_signals()
                actual_degradation_rate = iterative_fitting.degradation_rate()

                np.testing.assert_array_equal(actual_clear_sky_signals,
                                              expected_clear_sky_signals)
                # np.testing.assert_array_equal(actual_degradation_rate,
                #                               expected_degradation_rate)
                np.testing.assert_almost_equal(actual_degradation_rate,
                                               expected_degradation_rate,
                                               decimal=8)
