import unittest
import os
import numpy as np
import cvxpy as cvx
from statistical_clear_sky.algorithm.iterative_fitting import IterativeFitting
from statistical_clear_sky.solver_type import SolverType

class TestIterativeFitting(unittest.TestCase):

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
