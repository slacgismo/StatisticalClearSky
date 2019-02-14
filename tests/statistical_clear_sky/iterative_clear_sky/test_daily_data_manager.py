import unittest
from statistical_clear_sky.iterative_clear_sky.daily_data_manager import DailyDataManager
import numpy as np

class TestDailyDataManager(unittest.TestCase):
    '''
    Unit test for obtaining daily data.
    It convers the content of the constructor of main.IterativeClearSky in the original code.
    '''

    def setUp(self):
        pass

    def test_run_svd(self):

        # Data from Example_02 Jupyter notebook.
        # From 100th to 104th element of outer array,
        # first 8 elements of inner array.
        daily_signals = np.array([[3.65099996e-01, 0.00000000e+00,
            0.00000000e+00, 2.59570003e+00],
            [6.21100008e-01, 0.00000000e+00, 0.00000000e+00, 2.67740011e+00],
            [8.12500000e-01, 0.00000000e+00, 0.00000000e+00, 2.72729993e+00],
            [9.00399983e-01, 0.00000000e+00, 0.00000000e+00, 2.77419996e+00]])
        rank_k = 4

        expected_result = np.array([29.0347036, 29.0185262, 29.00233706,
            28.9861128])

        daily_data_manager = DailyDataManager(daily_signals, rank_k = rank_k)
        actual_result = daily_data_manager.run_svd()

        # [TODO]: For debugging. Remove this:
        print("actual_result: %s" % (actual_result))

        # np.testing.assert_almost_equal(actual_result, expected_result, 0.1)
