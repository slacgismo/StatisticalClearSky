import unittest
import numpy as np
from statistical_clear_sky.algorithm.initialization.weight_setting\
 import WeightSetting

class TestWeightSetting(unittest.TestCase):

    def setUp(self):
        self.weight_setting = WeightSetting()

    def test_obtain_weights(self):

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

        # Data from Example_02 Jupyter notebook.
        # From 100th to 103th element of array.
        #expected_weights = np.array([0.0, 0.97870261, 0.93385772, 0.0])
        # TODO: Get better test data, so that some of the values are > 0.6:
        expected_weights = np.array([0.0, 0.0, 0.0, 0.0])

        actual_weights = self.weight_setting.obtain_weights(power_signals_d)

        np.testing.assert_array_equal(actual_weights, expected_weights)
