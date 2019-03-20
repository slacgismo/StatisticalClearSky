"""
This module defines a class for Weight Setting Algorithm.
"""
import numpy as np
import cvxpy as cvx
from statistical_clear_sky.solver_type import SolverType

class WeightSetting(object):
    """
    Delegate class.
    Weight Setting Algorithm:
    Two metrics are calculated and normalized to the interval [0, 1],
    and then the geometric mean is taken.
    Metric 1: daily smoothness
    Metric 2: seasonally weighted daily energy
    After calculating the geometric mean of these two values, weights below
    """

    def __init__(self, solver_type=SolverType.ecos):
        self._solver_type = solver_type

    def obtain_weights(self, power_signals_d):

        ########################################################
        # Beginning of extracted code from the constructor of
        # main.IterativeClearSky
        ########################################################
        # Take the norm of the second different of each day's signal.
        # This gives a rough estimate of the smoothness of
        # day in the data set
        smoothness_tc = np.linalg.norm(power_signals_d[:-2] - 2 *
            power_signals_d[1:-1] + power_signals_d[2:], ord=1, axis=0)
        # Shift this metric so the median is at zero
        smoothness_tc = np.percentile(smoothness_tc, 50) - smoothness_tc
        # Normalize such that the maximum value is equal to one
        smoothness_tc /= np.max(smoothness_tc)
        # Take the positive part function,
        # i.e. set the negative values to zero. This is the first metric
        smoothness_tc = np.clip(smoothness_tc, 0, None)
        # Calculate the daily energy
        daily_energy = np.sum(power_signals_d, axis=0)
        # Solve a convex minimization problem to roughly fit the local 90th
        # percentile of the data (quantile regression)
        x = cvx.Variable(len(smoothness_tc))
        objective = cvx.Minimize(
            cvx.sum(0.5 * cvx.abs(daily_energy - x) + (.9 - 0.5) *
            (daily_energy - x)) + 1e3 * cvx.norm(cvx.diff(x, k=2)))
        problem = cvx.Problem(objective)
        problem.solve(solver=self._solver_type.value)
        # x gives us the local top 90th percentile of daily energy,
        # i.e. the very sunny days. This gives us our
        # seasonal normalization.
        daily_energy = np.clip(np.divide(daily_energy, x.value), 0, 1)
        # theta sets the weighting on the geometric mean
        theta = 0.1
        weights = np.multiply(np.power(smoothness_tc, theta),
                              np.power(daily_energy, 1.-theta))
        # Finally, set values less than 0.6 to be equal to zero
        weights[weights < 0.6] = 0.
        ########################################################
        # End of extracted code from the constructor of
        # main.IterativeClearSky
        ########################################################

        return weights
