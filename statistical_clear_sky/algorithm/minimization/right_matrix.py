"""
This module defines functionality unique to right matrix minimization.
"""
import cvxpy as cvx
import numpy as np
from statistical_clear_sky.solver_type import SolverType
from statistical_clear_sky.algorithm.minimization.abstract\
 import AbstractMinimization

class RightMatrixMinimization(AbstractMinimization):
    """
    Uses minimization method in parent class with fixed Left Matrix value,
    keeping Right matrix as a variable.
    """

    def __init__(self, power_signals_d, rank_k, weights, tau, mu_r,
                 is_degradation_calculated=True,
                 max_degradation=0., min_degradation=-0.25,
                 solver_type=SolverType.ecos):

        super().__init__(power_signals_d, rank_k, weights, tau,
                         solver_type=solver_type)
        self._mu_r = mu_r

        self._is_degradation_calculated = is_degradation_calculated
        self._max_degradation = max_degradation
        self._min_degradation = min_degradation

    def _define_parameters(self, l_cs_value, r_cs_value, beta_value):
        l_cs_param = l_cs_value
        r_cs_param = cvx.Variable(shape=(self._rank_k,
                                         self._power_signals_d.shape[1]))
        r_cs_param.value = r_cs_value
        beta_param = cvx.Variable()
        beta_param.value = beta_value
        return l_cs_param, r_cs_param, beta_param

    def _term_f2(self, l_cs_param, r_cs_param):
        r_tilde = self._obtain_r_tilde(r_cs_param)
        term_f2 = self._mu_r * cvx.norm(r_tilde[:, :-2] - 2
                   * r_tilde[:, 1:-1] + r_tilde[:, 2:], 'fro')
        return term_f2

    def _term_f3(self, l_cs_param, r_cs_param):
        r_tilde = self._obtain_r_tilde(r_cs_param)
        if self._power_signals_d.shape[1] > 365:
            term_f3 = self._mu_r * cvx.norm(r_tilde[1:, :-365]
                                      - r_tilde[1:, 365:], 'fro')
        else:
            term_f3 = self._mu_r * cvx.norm(r_tilde[:, :-365]
                                      - r_tilde[:, 365:], 'fro')
        return term_f3

    def _constraints(self, l_cs_param, r_cs_param, beta_param, component_r0):
        constraints = [
            l_cs_param * r_cs_param >= 0,
            r_cs_param[0] >= 0
        ]
        if self._power_signals_d.shape[1] > 365:
            r = r_cs_param[0, :].T
            if self._is_degradation_calculated:
                constraints.extend([
                    cvx.multiply(1./ component_r0[:-365],
                                 r[365:] - r[:-365]) == beta_param,
                    beta_param >= -.25
                ])
                if self._max_degradation is not None:
                    constraints.append(
                        beta_param <= self._max_degradation)
                if self._min_degradation is not None:
                    constraints.append(
                        beta_param >= self._min_degradation)
            else:
                constraints.append(cvx.multiply(1./ component_r0[:-365],
                                                r[365:] - r[:-365]) == 0)
        return constraints

    def _handle_exception(self, problem):
        if problem.status != 'optimal':
            raise ProblemStatusError('Minimize R status: ' + problem.status)

    def _result(self, l_cs_param, r_cs_param, beta_param):
        return l_cs_param, r_cs_param.value, beta_param.value

    def _obtain_r_tilde(self, r_cs_param):
        if r_cs_param.shape[1] < 365 + 2:
            n_tilde = 365 + 2 - r_cs_param.shape[1]
            r_tilde = cvx.hstack([r_cs_param,
                                  cvx.Variable(shape=(self._rank_k, n_tilde))])
        else:
            r_tilde = r_cs_param
        return r_tilde
