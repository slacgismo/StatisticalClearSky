import unittest
from\
 statistical_clear_sky.iterative_clear_sky.initialization.linearization_helper\
 import LinearizationHelper
import numpy as np
import os
from statistical_clear_sky.solver_type import SolverType

class TestLinealizationHelper(unittest.TestCase):
    '''
    Unit test for obtaining initial data of Right Vectors component r0,
    which is used as a denomoniator of non-linear equation in order to make
    it linear.
    It convers the first part of the constructor of main.IterativeClearSky
    in the original code.
    '''

    def setUp(self):
        pass

    def test_obtain_component_r0(self):

        # input_power_signals_file_path = os.path.abspath(
        #     os.path.join(os.path.dirname(__file__),
        #                  "../fixtures/daily_signals_1.txt"))
        # power_signals_d = np.loadtxt(input_power_signals_file_path,
        #                            delimiter = ',')
        #
        # print("power_signals_d: %s" % (power_signals_d))

        # Data from Example_02 Jupyter notebook.
        # From 100th to 104th element of outer array,
        # first 8 elements of inner array.
        power_signals_d = np.array([[3.65099996e-01, 0.00000000e+00,
            0.00000000e+00, 2.59570003e+00],
            [6.21100008e-01, 0.00000000e+00, 0.00000000e+00, 2.67740011e+00],
            [8.12500000e-01, 0.00000000e+00, 0.00000000e+00, 2.72729993e+00],
            [9.00399983e-01, 0.00000000e+00, 0.00000000e+00, 2.77419996e+00]])
        rank_k = 4

        # Result based on entire D in Example_02 Jupyter notebook.
        # expected_result = np.array([29.0347036, 29.0185262, 29.00233706,
            # 28.9861128])
        expected_result = np.array([1.36527916, 2.70624333, 4.04720749,
            5.38817165])

        linearization_helper = LinearizationHelper(power_signals_d,
                                                   rank_k = rank_k)
        # left_vectors_u: Left singular vectors
        # sigma: singular values
        # v: Right singular vectors
        left_vectors_u, sigma, right_vectors_v = np.linalg.svd(power_signals_d)
        actual_result = linearization_helper.obtain_component_r0(left_vectors_u,
            sigma, right_vectors_v, solver_type = SolverType.ecos)

        # TODO: For debugging. Remove this:
        # print("actual_result: %s" % (actual_result))

        np.testing.assert_almost_equal(actual_result, expected_result,
            decimal = 2)
