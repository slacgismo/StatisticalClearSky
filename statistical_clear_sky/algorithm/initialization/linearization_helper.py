"""
This module defines helper class for the purpose of linearization.
(Named as a helper instead of util since it doesn't directly do liniearization.)
"""

import numpy as np
import cvxpy as cvx
from statistical_clear_sky.solver_type import SolverType

class LinearizationHelper(object):
    """
    Delegate class to take care of obtaining a value used to make make a
    constraint to be linear, in order to make the optimization problem to
    be convex optimization problem.
    """

    def __init__(self, solver_type=SolverType.ecos):
        self._solver_type = solver_type

    def obtain_component_r0(self, power_signals_d, left_low_rank_matrix_u,
                            singular_values_sigma, right_low_rank_matrix_v,
                            rank_k=4):

        ########################################################
        # Beginning of extracted code from the constructor of
        # main.IterativeClearSky
        ########################################################
        right_vectors_r_cs = np.diag(singular_values_sigma[:rank_k]).dot(
            right_low_rank_matrix_v[:rank_k, :])
        component_r0 = right_vectors_r_cs[0]
        x = cvx.Variable(power_signals_d.shape[1])
        objective = cvx.Minimize(
            cvx.sum(0.5 * cvx.abs(component_r0 - x) + (.9 - 0.5) *
                    (component_r0 - x)) + 1e3 * cvx.norm(cvx.diff(x, k=2)))
        problem = cvx.Problem(objective)
        problem.solve(solver=self._solver_type.value)
        result_component_r0 = x.value
        ########################################################
        # End of extracted code from the constructor of
        # main.IterativeClearSky
        ########################################################

        return result_component_r0
