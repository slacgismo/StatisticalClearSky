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

    def __init__(self, power_signals_d, rank_k, weights, l_cs_variable,
                 r_cs_variable, beta_variable, tau,
                 solver_type=SolverType.ecos):
        self._power_signals_d = power_signals_d
        self._rank_k = rank_k
        self._weights = weights
        self._l_cs_variable = l_cs_variable
        self._r_cs_variable = r_cs_variable
        self._beta_variable = beta_variable
        self._tau = tau
        self._solver_type = solver_type

    def minimize(self):
        objective = cvx.Minimize(self._form1() + self._form2() + self._form3())
        problem = cvx.Problem(objective, self._constraints())
        problem.solve(self._solver_type.value)
        self._handle_exception(problem)
        return self._l_cs_variable, self._r_cs_variable, self._beta_variable

    def _form1(self):
        """
        This method defines the generic from of the first term of objective
        function.
        Subclass defines which of l_cs and r_cs value is fixed.
        """

        weights1 = np.diag(self._weights)
        return cvx.sum((0.5 * cvx.abs(self._power_signals_d
                        - self._l_cs_param * self._r_cs_param)
                      + (self._tau - 0.5) * (self._power_signals_d
                        - self._l_cs_param * self._r_cs_param))
                     * weights1)

    @abstractmethod
    def _form2(self):
        pass

    @abstractmethod
    def _form3(self):
        pass

    @abstractmethod
    def _constraints(self):
        pass

    @abstractmethod
    def _handle_exception(self, problem):
        pass

class LeftMatrixMinimization(AbstractMinimization):
    """
    Uses minimization method in parent class with fixed Right matrix value,
    keeping Left matrix as a variable.
    """

    def __init__(self, power_signals_d, rank_k, weights, l_cs_variable,
                 r_cs_variable, beta_variable, tau, mu_l):

        super().__init__(power_signals_d, rank_k, weights, l_cs_variable,
                       r_cs_variable, beta_variable, tau)
        self._l_cs_param = l_cs_variable
        self._r_cs_param = r_cs_variable.value
        self._mu_l = mu_l

    def _form2(self):
        weights2 = np.eye(self._rank_k)
        form2 = self._mu_l * cvx.norm((self._l_cs_param[:-2, :] - 2
                * self._l_cs_param[1:-1, :]
                + self._l_cs_param[2:, :]) * weights2, 'fro')
        return form2

    def _form3(self):
        return 0

    def _constraints(self):
        return [
            self._l_cs_param * self._r_cs_param >= 0,
            self._l_cs_param[np.average(self._power_signals_d, axis=1)
                <= 1e-5, :] == 0,
            cvx.sum(self._l_cs_param[:, 1:], axis=0) == 0
        ]

    def _handle_exception(self, problem):
        if problem.status != 'optimal':
            raise ProblemStatusError('Minimize L status: ' + problem.status)

class RightMatrixMinimization(AbstractMinimization):
    """
    Uses minimization method in parent class with fixed Left Matrix value,
    keeping Right matrix as a variable.
    """

    def __init__(self, power_signals_d, rank_k, weights, l_cs_variable,
                 r_cs_variable, beta_variable, tau, mu_r, component_r0,
                 is_degradation_calculated=True,
                 max_degradation=0., min_degradation=-0.25):

        super().__init__(power_signals_d, rank_k, weights, l_cs_variable,
                         r_cs_variable, tau)
        self._l_cs_param = l_cs_variable.value
        self._r_cs_param = r_cs_variable
        self._mu_r = mu_r
        self._component_r0 = component_r0

        self._r_tilde = self._get_r_tilde()

        self._is_degradation_calculated = is_degradation_calculated
        self._max_degradation = max_degradation
        self._min_degradation = min_degradation

    def _form2(self):
        form2 = self._mu_r * cvx.norm(self._r_tilde[:, :-2] - 2
                * self._r_tilde[:, 1:-1]
                + self._r_tilde[:, 2:], 'fro')
        return form2

    def _form3(self):
        if self._power_signals_d.shape[1] > 365:
            f3 = self._mu_r * cvx.norm(self._r_tilde[1:, :-365]
                                      - self._r_tilde[1:, 365:], 'fro')
        else:
            f3 = self._mu_r * cvx.norm(self._r_tilde[:, :-365]
                                      - self._r_tilde[:, 365:], 'fro')

    def _constraints(self):
        constraints = [
            self._l_cs_param * self._r_cs_param >= 0,
            self._r_cs_param >= 0
        ]
        if self._power_signals_d > 365:
            r = self._r_cs_param[0, :].T
            if self._is_degradation_calculated:
                constraints.extend([
                    cvx.multiply(1./ self._component_r0[:-365],
                                 r[365:] - r[:-365]) == self._beta_variable,
                    self._beta_variable >= -.25
                ])
                if self._max_degradation is not None:
                    constraints.append(
                        self._beta_variable <= self._max_degradation)
                if self._min_degradation is not None:
                    constraints.append(
                        self._beta_variable >= self._min_degradation)
            else:
                constraints.append(cvx.multiply(1./ self._component_r0[:-365],
                                                r[365:] - r[:-365]) == 0)
        return constraints

    def _handle_exception(self, problem):
        if problem.status != 'optimal':
            raise ProblemStatusError('Minimize R status: ' + problem.status)

    def _get_r_tilde(self):
        if self._r_cs_param.shape[1] < 365 + 2:
            n_tilde = 365 + 2 - self._r_cs_param.shape[1]
            r_tilde = cvx.hstack([self._r_cs_param,
                                  cvx.Variable(shape=(self._rank_k, n_tilde))])
        else:
            r_tilde = self._r_cs_param

        return r_tilde

