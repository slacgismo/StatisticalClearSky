"""
This module defines common functionality of minimization problem solution
 process.
Since there is common code for minimization of both L matrix and R matrix,
the common code is placed in the abstract base class.
"""
from abc import abstractmethod
import cvxpy as cvx
import numpy as np
from statistical_clear_sky.solver_type import SolverType

class AbstractMinimization():
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

    def minimize(self, l_cs_value, r_cs_value, beta_value, component_r0):
        l_cs_param, r_cs_param, beta_param = self._define_parameters(l_cs_value,
            r_cs_value, beta_value)
        objective = cvx.Minimize(self._term_f1(l_cs_param, r_cs_param)
                                 + self._term_f2(l_cs_param, r_cs_param)
                                 + self._term_f3(l_cs_param, r_cs_param))
        constraints = self._constraints(l_cs_param, r_cs_param, beta_param,
                                        component_r0)
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
    def _constraints(self, l_cs_param, r_cs_param, beta_param, component_r0):
        pass

    @abstractmethod
    def _handle_exception(self, problem):
        pass

    @abstractmethod
    def _result(self, l_cs_param, r_cs_param, beta_param):
        pass
