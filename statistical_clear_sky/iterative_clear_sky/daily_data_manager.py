import numpy as np
import cvxpy as cvx

class DailyDataManager(object):

    def __init__(self, daily_signals, rank_k = 4):
        self._daily_signals = daily_signals
        self._rank_k = rank_k

    def run_svd(self):

        ########################################################
        # Beginning of extracted code from the constructor of
        # main.IterativeClearSky
        ########################################################
        # u: Left singular vectors
        # sigma: singular values
        # v: Right singular vectors
        u, sigma, v = np.linalg.svd(self._daily_signals)
        if np.sum(u[:, 0]) < 0:
            u[:, 0] *= -1
            v[0] *= -1
        l0 = u[:, :self._rank_k]
        r_cs = np.diag(sigma[:self._rank_k]).dot(v[:self._rank_k, :])
        r0 = r_cs[0]
        x = cvx.Variable(self._daily_signals.shape[1])
        objective = cvx.Minimize(
            cvx.sum(0.5 * cvx.abs(r0 - x) + (.9 - 0.5) * (r0 - x)) + 1e3 * cvx.norm(cvx.diff(x, k=2)))
        problem = cvx.Problem(objective)
        problem.solve(solver='MOSEK')
        result_r0 = x.value
        ########################################################
        # End of extracted code from the constructor of
        # main.IterativeClearSky
        ########################################################

        return result_r0
