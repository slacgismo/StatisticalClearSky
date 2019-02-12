import unittest
from statistical_clear_sky.iterative_clear_sky.daily_data_manager import DailyDataManager

class TestDailyDataManager(unittest.TestCase):
    '''
    Unit test for obtaining daily data.
    It convers the content of the constructor of main.IterativeClearSky in the original code.
    '''

    def setUp(self):
        self.daily_data_manager = DailyDataManager()

    def test_perfoem_optimization(self):

        # TODO: Use the value from Example_02 notebooK:
        expected_result = 1

        actual_result = self.daily_data_manager.perform_optimization()

        self.assertAlmostEqual(actual_result, expected_result, delta = 0.1)
