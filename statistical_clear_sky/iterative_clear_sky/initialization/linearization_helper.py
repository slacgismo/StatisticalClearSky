import numpy as np
import cvxpy as cvx
from statistical_clear_sky.solver_type import SolverType

class LinearizationHelper(object):

    def __init__(self, power_signals_d, rank_k = 4):
        self._power_signals_d = power_signals_d
        self._rank_k = rank_k

    def obtain_component_r0(self, left_vectors_u, sigma, right_vectors_v,
                            solver_type = SolverType.ecos):

        ########################################################
        # Beginning of extracted code from the constructor of
        # main.IterativeClearSky
        ########################################################
        if np.sum(left_vectors_u[:, 0]) < 0:
            left_vectors_u[:, 0] *= -1
            right_vectors_v[0] *= -1
        right_vectors_r_cs = np.diag(sigma[:self._rank_k]).dot(
            right_vectors_v[:self._rank_k, :])
        component_r0 = right_vectors_r_cs[0]
        x = cvx.Variable(self._power_signals_d.shape[1])
        objective = cvx.Minimize(
            cvx.sum(0.5 * cvx.abs(component_r0 - x) + (.9 - 0.5) *
                (component_r0 - x)) + 1e3 * cvx.norm(cvx.diff(x, k = 2)))
        problem = cvx.Problem(objective)
        problem.solve(solver = solver_type.value)
        result_component_r0 = x.value
        ########################################################
        # End of extracted code from the constructor of
        # main.IterativeClearSky
        ########################################################

        return result_component_r0
