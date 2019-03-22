import unittest
from unittest.mock import Mock
import os
import numpy as np
import cvxpy as cvx
from statistical_clear_sky.algorithm.iterative_fitting import IterativeFitting
from statistical_clear_sky.solver_type import SolverType
from statistical_clear_sky.algorithm.minimization.left_matrix\
 import LeftMatrixMinimization
from statistical_clear_sky.algorithm.minimization.right_matrix\
 import RightMatrixMinimization

class TestIterativeFittingExecute(unittest.TestCase):

    def setUp(self):

        left_matrix_minimize_return_values = []

        for i in range(8):

            l_cs_value_left_matrix_file_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__),
                "../fixtures/for_mock",
                ("l_cs_value_after_left_matrix_minimization_iteration_{}.csv"
                 .format(i+1))))
            with open(l_cs_value_left_matrix_file_path) as file:
                l_cs_value_left_matrix = np.loadtxt(file, delimiter=',')
            r_cs_value_left_matrix_file_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__),
                "../fixtures/for_mock",
                ("r_cs_value_after_left_matrix_minimization_iteration_{}.csv"
                 .format(i+1))))
            with open(r_cs_value_left_matrix_file_path) as file:
                r_cs_value_left_matrix = np.loadtxt(file, delimiter=',')
            beta_value_left_matrix_file_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__),
                "../fixtures/for_mock",
                ("beta_value_after_left_matrix_minimization_iteration_{}.csv"
                 .format(i+1))))
            with open(beta_value_left_matrix_file_path) as file:
                beta_value_left_matrix = np.loadtxt(file, delimiter=',')

            left_matrix_minimize_return_values.append(
                (l_cs_value_left_matrix, r_cs_value_left_matrix,
                 beta_value_left_matrix))

        self.mock_left_matrix_minimization = Mock(spec=LeftMatrixMinimization)
        self.mock_left_matrix_minimization.minimize.side_effect =\
            left_matrix_minimize_return_values

        right_matrix_minimize_return_values = []

        for i in range(8):

            l_cs_value_right_matrix_file_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__),
                "../fixtures/for_mock",
                ("l_cs_value_after_right_matrix_minimization_iteration_{}.csv"
                 .format(i+1))))
            with open(l_cs_value_right_matrix_file_path) as file:
                l_cs_value_right_matrix = np.loadtxt(file, delimiter=',')
            r_cs_value_right_matrix_file_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__),
                "../fixtures/for_mock",
                ("r_cs_value_after_right_matrix_minimization_iteration_{}.csv"
                 .format(i+1))))
            with open(r_cs_value_right_matrix_file_path) as file:
                r_cs_value_right_matrix = np.loadtxt(file, delimiter=',')
            beta_value_right_matrix_file_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__),
                "../fixtures/for_mock",
                ("beta_value_after_right_matrix_minimization_iteration_{}.csv"
                 .format(i+1))))
            with open(beta_value_right_matrix_file_path) as file:
                beta_value_right_matrix = np.loadtxt(file, delimiter=',')

            right_matrix_minimize_return_values.append(
                (l_cs_value_right_matrix, r_cs_value_right_matrix,
                 beta_value_right_matrix))

        self.mock_right_matrix_minimization = Mock(spec=RightMatrixMinimization)
        self.mock_right_matrix_minimization.minimize.side_effect =\
            right_matrix_minimize_return_values

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
            iterative_fitting.set_left_matrix_minimization(
                self.mock_left_matrix_minimization)
            iterative_fitting.set_right_matrix_minimization(
                self.mock_right_matrix_minimization)
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
