import unittest
import os
import numpy as np
import cvxpy as cvx
from statistical_clear_sky.solver_type import SolverType
from statistical_clear_sky.algorithm.minimization.left_matrix\
 import LeftMatrixMinimization

class TestLeftMatrixMinimization(unittest.TestCase):

    def test_minimize(self):
        # Data from Example_02 Jupyter notebook.
        # From 144th to 147th element of outer array,
        # from 183th to 187th element of inner array.
        # Since in power signals matrix, row is time of day
        # and column is day number.
        # Since the number of rows is 288,
        # 144 to 147 are from 12:00 noon to 12:15pm (5 minutes interval).
        # 213th to 216th are from August 1 to 4 (summer).
        # power_signals_d = np.array([[1.24810004, 1.30900002, 0.56239998,
        #                              1.30690002],
        #                             [1.33389997, 1.40310001, 0.67150003,
        #                              0.77249998],
        #                             [1.42349994, 1.51800001, 1.43809998,
        #                              1.20449996],
        #                             [1.52020001, 1.45150006, 1.84809995,
        #                              0.99949998]])
        # Note: In order to make the second constraint satisfied (<= 1e-5),
        #       made the first element values 0
        power_signals_d = np.array([[0.0, 0.0, 0.0, 0.0],
                                    [1.33389997, 1.40310001, 0.67150003,
                                     0.77249998],
                                    [1.42349994, 1.51800001, 1.43809998,
                                     1.20449996],
                                    [1.52020001, 1.45150006, 1.84809995,
                                     0.99949998]])
        rank_k = 4
        # Data from Example_02 Jupyter notebook.
        # From 144th to 147th element.
        weights = np.array([0.0, 0.0, 0.97073243, 0.97243198])
        tau = 0.9
        mu_l = 5e2
        solver_type=SolverType.ecos
        initial_l_cs_value = np.array([[0.12227644, -0.05536519,
                                        -0.02796016, 0.11115515],
                                       [0.12183656, -0.06418167,
                                        -0.03631565, 0.09248578],
                                       [0.12190038, -0.07035386,
                                        -0.03077544, 0.06306368],
                                       [0.12185763, -0.0822263,
                                        -0.02468169,  0.03843156]])
        initial_r_cs_value = np.array([[7.81948013, 11.26965908,
                                        11.43521789, 8.89706298],
                                       [0.18783052, -1.17162576,
                                        -1.68541257, -1.14962881],
                                       [0.97275831, 0.99957452,
                                        0.92734892, 0.453427],
                                       [-0.86265428, -3.28835462,
                                        -4.00326343, -1.76664483]])
        initial_beta_value = 0.0
        initial_component_r0 = np.array([1.36527916, 2.70624333, 4.04720749,
                                         5.38817165])

        # Note: In Example_02, rank is specified to be 6. Better data needed.
        # expected_l_cs_value = np.array([[1.25425950e-01, -4.29673887e-02,
        #                                  1.35640294e-02, 8.16029351e-02],
        #                                 [1.25222358e-01, -5.45503992e-02,
        #                                  6.91829367e-03, 5.43180337e-02],
        #                                 [1.24787127e-01, -6.62930310e-02,
        #                                  1.27108315e-04, 2.63508908e-02],
        #                                 [1.24914255e-01, -7.74757429e-02,
        #                                  -6.51881582e-03, -6.79405903e-05]])
        # Note: Since power_signals_d is modified to satisfy a constraint,
        #       the result value is diffrent from above.
        expected_l_cs_value = np.array([[2.610888e-14, -1.027025e-14,
                                         1.481367e-14, -1.786423e-14],
                                        [6.769088e-02, -5.028329e-14,
                                         -4.090143e-14, 1.891483e-13],
                                        [1.353818e-01, -8.877942e-14,
                                         4.614613e-15, -1.047267e-14],
                                        [2.030726e-01, 1.495160e-13,
                                         1.955246e-14, -1.573292e-13]])
        expected_r_cs_value = initial_r_cs_value
        expected_beta_value = initial_beta_value

        left_matrix_minimization = LeftMatrixMinimization(power_signals_d,
            rank_k, weights, tau, mu_l, solver_type=SolverType.ecos)

        actual_l_cs_value, actual_r_cs_value, actual_beta_value =\
            left_matrix_minimization.minimize(initial_l_cs_value,
                                              initial_r_cs_value,
                                              initial_beta_value,
                                              initial_component_r0)

        # np.testing.assert_array_equal(actual_l_cs_value, expected_l_cs_value)
        np.testing.assert_almost_equal(actual_l_cs_value, expected_l_cs_value,
                                       decimal=6)
        np.testing.assert_array_equal(actual_r_cs_value, expected_r_cs_value)
        np.testing.assert_array_equal(actual_beta_value, expected_beta_value)

    def test_minimize_with_example_02_data_iteration_1(self):

        input_power_signals_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                         "../../fixtures/power_signals_d_1.csv"))
        with open(input_power_signals_file_path) as file:
            power_signals_d = np.loadtxt(file, delimiter=',')

        rank_k = 6

        weights_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                         "../../fixtures/weights_1.csv"))
        with open(weights_file_path) as file:
            weights = np.loadtxt(file, delimiter=',')

        tau = 0.9
        mu_l = 5e2

        initial_l_cs_value_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
            "../../fixtures/left_matrix_minimization/initial_l_cs_value_1.csv"))
        with open(initial_l_cs_value_file_path) as file:
            initial_l_cs_value = np.loadtxt(file, delimiter=',')

        initial_r_cs_value_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
            "../../fixtures/left_matrix_minimization/initial_r_cs_value_1.csv"))
        with open(initial_r_cs_value_file_path) as file:
            initial_r_cs_value = np.loadtxt(file, delimiter=',')

        initial_beta_value = 0.0

        l_cs_value_after_iteration_1_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
            "../../fixtures/left_matrix_minimization",
            "l_cs_value_after_iteration_1_1.csv"))
        with open(l_cs_value_after_iteration_1_file_path) as file:
            expected_l_cs_value = np.loadtxt(file, delimiter=',')

        r_cs_value_after_iteration_1_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
            "../../fixtures/left_matrix_minimization",
            "r_cs_value_after_iteration_1_1.csv"))
        with open(r_cs_value_after_iteration_1_file_path) as file:
            expected_r_cs_value = np.loadtxt(file, delimiter=',')

        expected_beta_value = initial_beta_value

        initial_r0_value_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
            "../../fixtures/right_matrix_minimization/initial_r0_value_1.csv"))
        with open(initial_r0_value_file_path) as file:
            initial_component_r0_value = np.loadtxt(file, delimiter=',')

        left_matrix_minimization = LeftMatrixMinimization(power_signals_d,
            rank_k, weights, tau, mu_l, solver_type=SolverType.mosek)

        try:
            actual_l_cs_value, actual_r_cs_value, actual_beta_value =\
                left_matrix_minimization.minimize(initial_l_cs_value,
                                                  initial_r_cs_value,
                                                  initial_beta_value,
                                                  initial_component_r0_value)
        except cvx.SolverError:
            self.skipTest("This test uses MOSEK solver"
                + "because default ECOS solver fails with large data. "
                + "Unless MOSEK is installed, this test fails.")
        else:
            np.testing.assert_array_equal(actual_l_cs_value,
                                          expected_l_cs_value)
            np.testing.assert_array_equal(actual_r_cs_value,
                                          expected_r_cs_value)
            np.testing.assert_array_equal(actual_beta_value,
                                          expected_beta_value)

    def test_minimize_with_example_02_data_iteration_2(self):

        input_power_signals_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                         "../../fixtures/power_signals_d_1.csv"))
        with open(input_power_signals_file_path) as file:
            power_signals_d = np.loadtxt(file, delimiter=',')

        rank_k = 6

        weights_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                         "../../fixtures/weights_1.csv"))
        with open(weights_file_path) as file:
            weights = np.loadtxt(file, delimiter=',')

        tau = 0.9
        mu_l = 5e2

        initial_l_cs_value_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
            "../../fixtures/left_matrix_minimization",
            "l_cs_value_before_iteration_2_1.csv"))
        with open(initial_l_cs_value_file_path) as file:
            initial_l_cs_value = np.loadtxt(file, delimiter=',')

        initial_r_cs_value_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
            "../../fixtures/left_matrix_minimization",
            "r_cs_value_before_iteration_2_1.csv"))
        with open(initial_r_cs_value_file_path) as file:
            initial_r_cs_value = np.loadtxt(file, delimiter=',')

        initial_beta_value = 0.0

        l_cs_value_after_iteration_1_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
            "../../fixtures/left_matrix_minimization",
            "l_cs_value_after_iteration_2_1.csv"))
        with open(l_cs_value_after_iteration_1_file_path) as file:
            expected_l_cs_value = np.loadtxt(file, delimiter=',')

        r_cs_value_after_iteration_1_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
            "../../fixtures/left_matrix_minimization",
            "r_cs_value_after_iteration_2_1.csv"))
        with open(r_cs_value_after_iteration_1_file_path) as file:
            expected_r_cs_value = np.loadtxt(file, delimiter=',')

        expected_beta_value = initial_beta_value

        initial_r0_value_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
            "../../fixtures/right_matrix_minimization/initial_r0_value_1.csv"))
        with open(initial_r0_value_file_path) as file:
            initial_component_r0_value = np.loadtxt(file, delimiter=',')

        left_matrix_minimization = LeftMatrixMinimization(power_signals_d,
            rank_k, weights, tau, mu_l, solver_type=SolverType.mosek)

        try:
            actual_l_cs_value, actual_r_cs_value, actual_beta_value =\
                left_matrix_minimization.minimize(initial_l_cs_value,
                                                  initial_r_cs_value,
                                                  initial_beta_value,
                                                  initial_component_r0_value)
        except cvx.SolverError:
            self.skipTest("This test uses MOSEK solver"
                + "because default ECOS solver fails with large data. "
                + "Unless MOSEK is installed, this test fails.")
        else:
            np.testing.assert_array_equal(actual_l_cs_value,
                                          expected_l_cs_value)
            np.testing.assert_array_equal(actual_r_cs_value,
                                          expected_r_cs_value)
            np.testing.assert_array_equal(actual_beta_value,
                                          expected_beta_value)
