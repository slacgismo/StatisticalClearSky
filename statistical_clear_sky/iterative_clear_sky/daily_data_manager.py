import numpy as np

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
        # u, sigma, v = np.linalg.svd(self._daily_signals)
        # if np.sum(u[:, 0]) < 0:
        #     u[:, 0] *= -1
        #     v[0] *= -1
        # l0 = u[:, :self._rank_k]
        # r_cs = np.diag(sigma[:self._rank_k]).dot(v[:self._rank_k, :])
        # r0 = self.r_cs[0]
        # x = cvx.Variable(self._daily_signals.shape[1])
        # objective = cvx.Minimize(
        #     cvx.sum(0.5 * cvx.abs(r0 - x) + (.9 - 0.5) * (r0 - x)) + 1e3 * cvx.norm(cvx.diff(x, k=2)))
        # problem = cvx.Problem(objective)
        # problem.solve(solver='MOSEK')
        # r0 = x.value
        ########################################################
        # End of extracted code from the constructor of
        # main.IterativeClearSky
        ########################################################

        # Temp:
        return np.array([29.0347036, 29.0185262, 29.00233706,
            28.9861128, 28.96981832, 28.95352385, 28.93740498, 28.92169378])
