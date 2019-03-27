import unittest
import os
import numpy as np
import cvxpy as cvx
from statistical_clear_sky.algorithm.iterative_fitting import IterativeFitting
from statistical_clear_sky.solver_type import SolverType

class TestIterativeFitting(unittest.TestCase):

    def test_initialization(self):

        # Data from Example_02 Jupyter notebook.
        # From 100th to 103th element of outer array,
        # first 4 elements of inner array.
        power_signals_d = np.array([[3.65099996e-01, 0.00000000e+00,
                                     0.00000000e+00, 2.59570003e+00],
                                    [6.21100008e-01, 0.00000000e+00,
                                     0.00000000e+00, 2.67740011e+00],
                                    [8.12500000e-01, 0.00000000e+00,
                                     0.00000000e+00, 2.72729993e+00],
                                    [9.00399983e-01, 0.00000000e+00,
                                     0.00000000e+00, 2.77419996e+00]])
        rank_k = 4
        solver_type = SolverType.ecos

        iterative_fitting = IterativeFitting(power_signals_d, rank_k=rank_k,
            solver_type=SolverType.ecos, auto_fix_time_shifts=False)

    def test_calculate_objective_before_iteration_1(self):

        input_power_signals_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                         "../fixtures/power_signals_d_1.csv"))
        with open(input_power_signals_file_path) as file:
            power_signals_d = np.loadtxt(file, delimiter=',')

        rank_k = 6

        mu_l = 5e2
        mu_r = 1e3
        tau = 0.9

        initial_l_cs_value_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
            "../fixtures/initial_l_cs_value_1.csv"))
        with open(initial_l_cs_value_file_path) as file:
            l_cs_value = np.loadtxt(file, delimiter=',')

        initial_r_cs_value_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
            "../fixtures/initial_r_cs_value_1.csv"))
        with open(initial_r_cs_value_file_path) as file:
            r_cs_value = np.loadtxt(file, delimiter=',')

        beta_value = 0.0

        weights_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                         "../fixtures/weights_1.csv"))
        with open(weights_file_path) as file:
            weights = np.loadtxt(file, delimiter=',')

        expected_objective_values = np.array([2171.1759047049295,
            183.68448929381688, 388776.7729609732, 137624.12552820993])

        iterative_fitting = IterativeFitting(power_signals_d, rank_k=rank_k,
                                             auto_fix_time_shifts=False)

        actual_objective_values = iterative_fitting._calculate_objective(
            mu_l, mu_r, tau, l_cs_value, r_cs_value,
            beta_value, weights, sum_components=False)

        # Note: With Python 3.7, assertion passes
        #       but with Python 3.6, there is a minor discrepancy.
        # np.testing.assert_array_equal(actual_objective_values,
        #                               expected_objective_values)
        np.testing.assert_almost_equal(actual_objective_values,
                                       expected_objective_values,
                                       decimal=6)

    def test_calculate_objective_iteration_1(self):

        input_power_signals_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                         "../fixtures/power_signals_d_1.csv"))
        with open(input_power_signals_file_path) as file:
            power_signals_d = np.loadtxt(file, delimiter=',')

        rank_k = 6

        mu_l = 5e2
        mu_r = 1e3
        tau = 0.9

        initial_l_cs_value_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
            "../fixtures/objective_calculation",
            "l_cs_value_after_iteration_1_1.csv"))
        with open(initial_l_cs_value_file_path) as file:
            l_cs_value = np.loadtxt(file, delimiter=',')

        initial_r_cs_value_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
            "../fixtures/objective_calculation",
            "r_cs_value_after_iteration_1_1.csv"))
        with open(initial_r_cs_value_file_path) as file:
            r_cs_value = np.loadtxt(file, delimiter=',')

        beta_value = 0.0

        weights_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                         "../fixtures/weights_1.csv"))
        with open(weights_file_path) as file:
            weights = np.loadtxt(file, delimiter=',')

        expected_objective_values = np.array([1911.9056612619852,
            20.310140581069636, 207.5667132341283, 4.7643179724417583e-05])

        iterative_fitting = IterativeFitting(power_signals_d, rank_k=rank_k,
                                             auto_fix_time_shifts=False)

        actual_objective_values = iterative_fitting._calculate_objective(
            mu_l, mu_r, tau, l_cs_value, r_cs_value,
            beta_value, weights, sum_components=False)

        np.testing.assert_array_equal(actual_objective_values,
                                      expected_objective_values)

    def test_calculate_objective_iteration_2(self):

        input_power_signals_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                         "../fixtures/power_signals_d_1.csv"))
        with open(input_power_signals_file_path) as file:
            power_signals_d = np.loadtxt(file, delimiter=',')

        rank_k = 6

        mu_l = 5e2
        mu_r = 1e3
        tau = 0.9

        initial_l_cs_value_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
            "../fixtures/objective_calculation",
            "l_cs_value_after_iteration_2_1.csv"))
        with open(initial_l_cs_value_file_path) as file:
            l_cs_value = np.loadtxt(file, delimiter=',')

        initial_r_cs_value_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
            "../fixtures/objective_calculation",
            "r_cs_value_after_iteration_2_1.csv"))
        with open(initial_r_cs_value_file_path) as file:
            r_cs_value = np.loadtxt(file, delimiter=',')

        beta_value = 0.0

        weights_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                         "../fixtures/weights_1.csv"))
        with open(weights_file_path) as file:
            weights = np.loadtxt(file, delimiter=',')

        expected_objective_values = np.array([1613.9301083819053,
            18.429829588941793, 196.44983089760348, 2.347949251896394e-05])

        iterative_fitting = IterativeFitting(power_signals_d, rank_k=rank_k,
                                             auto_fix_time_shifts=False)

        actual_objective_values = iterative_fitting._calculate_objective(
            mu_l, mu_r, tau, l_cs_value, r_cs_value,
            beta_value, weights, sum_components=False)

        # np.testing.assert_array_equal(actual_objective_values,
        #                               expected_objective_values)
        np.testing.assert_almost_equal(actual_objective_values,
                                       expected_objective_values,
                                       decimal=13)

    def test_adjust_low_rank_matrices(self):

        # Data from Example_02 Jupyter notebook.
        # From 100th to 103th element of outer array,
        # first 4 elements of inner array.
        power_signals_d = np.array([[3.65099996e-01, 0.00000000e+00,
                                     0.00000000e+00, 2.59570003e+00],
                                    [6.21100008e-01, 0.00000000e+00,
                                     0.00000000e+00, 2.67740011e+00],
                                    [8.12500000e-01, 0.00000000e+00,
                                     0.00000000e+00, 2.72729993e+00],
                                    [9.00399983e-01, 0.00000000e+00,
                                     0.00000000e+00, 2.77419996e+00]])

        left_low_rank_matrix_u = np.array([[0.46881027, -0.77474963,
                                            0.39354624, 0.1584339],
                                           [0.49437073, -0.15174524,
                                            -0.6766346, -0.52415321],
                                           [-0.51153077, 0.32155093,
                                            -0.27710787, 0.74709605],
                                           [-0.5235941, 0.52282062,
                                            0.55722365, -0.37684163]])
        right_low_rank_matrix_v = np.array([[0.24562222, 0.0, 0.0, 0.96936563],
                                            [0.96936563, 0.0, 0.0, -0.24562222],
                                            [0.0, 1.0, 0.0, 0.0],
                                            [0.0, 0.0, 1.0, 0.0]])

        expected_left_low_rank_matrix_u = np.array([[-0.46881027, -0.77474963,
                                                     0.39354624, 0.1584339],
                                                    [-0.49437073, -0.15174524,
                                                     -0.6766346, -0.52415321],
                                                    [0.51153077, 0.32155093,
                                                     -0.27710787, 0.74709605],
                                                    [0.5235941, 0.52282062,
                                                     0.55722365, -0.37684163]])
        expected_right_low_rank_matrix_v = np.array([[-0.24562222, 0.0,
                                                      0.0, -0.96936563],
                                                     [0.96936563, 0.0,
                                                      0.0, -0.24562222],
                                                     [0.0, 1.0, 0.0, 0.0],
                                                     [0.0, 0.0, 1.0, 0.0]])

        iterative_fitting = IterativeFitting(power_signals_d,
                                             auto_fix_time_shifts=False)

        actual_left_low_rank_matrix_u, actual_right_low_rank_matrix_v = \
            iterative_fitting._adjust_low_rank_matrices(
                left_low_rank_matrix_u, right_low_rank_matrix_v)

        np.testing.assert_array_equal(actual_left_low_rank_matrix_u,
                                      expected_left_low_rank_matrix_u)
        np.testing.assert_array_equal(actual_right_low_rank_matrix_v,
                                      expected_right_low_rank_matrix_v)
