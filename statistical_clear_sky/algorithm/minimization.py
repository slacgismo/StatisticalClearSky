"""
This module defines minimization problem solution process.
Since there is common code for minimization of both L matrix and R matrix,
the common code is placed in the abstract base class, which is inherited by
the subclasses for L (left) matrix and R (right) matrix.
"""
from abc import abstractmethod
import cvxpy as cvx
import numpy as np
from statistical_clear_sky.solver_type import SolverType

class AbstractMinimization(object):
    """
    Abstract class for minimization that uses the same equation but
    the subclasses fix either L (left) matrix value or R (right) matrix
    value.
    """

    def __init__(self, power_signals_d, rank_k, weights, tau,
                 solver_type=SolverType.ecos):
        self._power_signals_d = power_signals_d
        self._rank_k = rank_k
        self._weights = weights
        self._tau = tau
        self._solver_type = solver_type

    def minimize(self, l_cs_value, r_cs_value, beta_value):
        l_cs_param, r_cs_param, beta_param = self._define_parameters(l_cs_value,
            r_cs_value, beta_value)
        objective = cvx.Minimize(self._term_f1(l_cs_param, r_cs_param)
                                 + self._term_f2(l_cs_param, r_cs_param)
                                 + self._term_f3(l_cs_param, r_cs_param))
        constraints = self._constraints(l_cs_param, r_cs_param, beta_param)
        problem = cvx.Problem(objective, constraints)
        problem.solve(solver=self._solver_type.value)
        self._handle_exception(problem)
        return self._result(l_cs_param, r_cs_param, beta_param)

    @abstractmethod
    def _define_parameters(self):
        pass

    def _term_f1(self, l_cs_param, r_cs_param):
        """
        This method defines the generic from of the first term of objective
        function.
        Subclass defines which of l_cs and r_cs value is fixed.
        """

        weights_w1 = np.diag(self._weights)
        return cvx.sum((0.5 * cvx.abs(self._power_signals_d
                        - l_cs_param * r_cs_param)
                      + (self._tau - 0.5) * (self._power_signals_d
                        - l_cs_param * r_cs_param))
                     * weights_w1)

    @abstractmethod
    def _term_f2(self, l_cs_param, r_cs_param):
        pass

    @abstractmethod
    def _term_f3(self, l_cs_param, r_cs_param):
        pass

    @abstractmethod
    def _constraints(self, l_cs_param, r_cs_param, beta_param):
        pass

    @abstractmethod
    def _handle_exception(self, problem):
        pass

    @abstractmethod
    def _result(self, l_cs_param, r_cs_param, beta_param):
        pass

class LeftMatrixMinimization(AbstractMinimization):
    """
    Uses minimization method in parent class with fixed Right matrix value,
    keeping Left matrix as a variable.
    """

    def __init__(self, power_signals_d, rank_k, weights, tau, mu_l,
                 solver_type=SolverType.ecos):

        super().__init__(power_signals_d, rank_k, weights, tau,
                         solver_type=solver_type)
        self._mu_l = mu_l

    def _define_parameters(self, l_cs_value, r_cs_value, beta_value):
        l_cs_param = cvx.Variable(shape=(self._power_signals_d.shape[0],
                                         self._rank_k))
        l_cs_param.value = l_cs_value
        r_cs_param = r_cs_value
        beta_param = cvx.Variable()
        beta_param.value = beta_value
        return l_cs_param, r_cs_param, beta_param

    def _term_f2(self, l_cs_param, r_cs_param):
        weights_w2 = np.eye(self._rank_k)
        term_f2 = self._mu_l * cvx.norm((l_cs_param[:-2, :] - 2
                * l_cs_param[1:-1, :] + l_cs_param[2:, :]) * weights_w2, 'fro')
        return term_f2

    def _term_f3(self, l_cs_param, r_cs_param):
        return 0

    def _constraints(self, l_cs_param, r_cs_param, beta_param):
        return [
            l_cs_param * r_cs_param >= 0,
            l_cs_param[np.average(self._power_signals_d, axis=1) <= 1e-5,
                       :] == 0,
            cvx.sum(l_cs_param[:, 1:], axis=0) == 0
        ]

    def _handle_exception(self, problem):
        if problem.status != 'optimal':
            raise ProblemStatusError('Minimize L status: ' + problem.status)

    def _result(self, l_cs_param, r_cs_param, beta_param):
        return l_cs_param.value, r_cs_param, beta_param.value

class RightMatrixMinimization(AbstractMinimization):
    """
    Uses minimization method in parent class with fixed Left Matrix value,
    keeping Right matrix as a variable.
    """

    def __init__(self, power_signals_d, rank_k, weights, tau, mu_r,
                 component_r0, is_degradation_calculated=True,
                 max_degradation=0., min_degradation=-0.25,
                 solver_type=SolverType.ecos):

        super().__init__(power_signals_d, rank_k, weights, tau,
                         solver_type=solver_type)
        self._mu_r = mu_r
        self._component_r0 = component_r0

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

    def _constraints(self, l_cs_param, r_cs_param, beta_param):
        constraints = [
            l_cs_param * r_cs_param >= 0,
            r_cs_param >= 0
        ]
        if self._power_signals_d.shape[1] > 365:
            r = r_cs_param[0, :].T
            if self._is_degradation_calculated:
                constraints.extend([
                    cvx.multiply(1./ self._component_r0[:-365],
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
                constraints.append(cvx.multiply(1./ self._component_r0[:-365],
                                                r[365:] - r[:-365]) == 0)
        return constraints

    def _handle_exception(self, problem):
        if problem.status != 'optimal':
            raise ProblemStatusError('Minimize R status: ' + problem.status)

    def _result(self, l_cs_param, r_cs_param, beta_param):
        return l_cs_param, r_cs_param.value, beta_param.value

    def _obtain_r_tilde(self, r_cs_param):
        if (not hasattr(self, '_r_tilde')) or (self._r_tilde is None):
            if r_cs_param.shape[1] < 365 + 2:
                n_tilde = 365 + 2 - r_cs_param.shape[1]
                self._r_tilde = cvx.hstack([r_cs_param,
                                  cvx.Variable(shape=(self._rank_k, n_tilde))])
            else:
                self._r_tilde = r_cs_param
        return self._r_tilde
