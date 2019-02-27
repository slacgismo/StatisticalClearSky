"""
This module defines "Statistical Clear Sky Fitting" algorithm.
"""

from time import time
import numpy as np
from numpy.linalg import norm
import cvxpy as cvx
from statistical_clear_sky.algorithm.initialization.linearization_helper\
 import LinearizationHelper
from statistical_clear_sky.algorithm.initialization.weight_setting\
 import WeightSetting
from statistical_clear_sky.solver_type import SolverType
from statistical_clear_sky.algorithm.exception import ProblemStatusError
from statistical_clear_sky.algorithm.serialization.state_data import StateData
from statistical_clear_sky.algorithm.serialization.serialization_mixin\
 import SerializationMixin

class IterativeClearSky(SerializationMixin):
    """
    Implementation of "Statistical Clear Sky Fitting" algorithm.
    """

    def __init__(self, power_signals_d, rank_k=4, solver_type=SolverType.ecos,
                 reserve_test_data=False, auto_fix_time_shifts=True):

        self._solver_type = solver_type

        self._fixed_time_stamps = False
        self._power_signals_d = power_signals_d
        self._rank_k = rank_k

        self._l_cs = cvx.Variable(shape=(power_signals_d.shape[0], rank_k))
        self._r_cs = cvx.Variable(shape=(rank_k, power_signals_d.shape[1]))
        self._beta = cvx.Variable()

        left_low_rank_matrix_u, singular_values_sigma, right_low_rank_matrix_v \
            = np.linalg.svd(power_signals_d)
        left_low_rank_matrix_u, right_low_rank_matrix_v = \
            self._adjust_low_rank_matrices(left_low_rank_matrix_u,
                                           right_low_rank_matrix_v)
        self._left_low_rank_matrix_u = left_low_rank_matrix_u
        self._singular_values_sigma = singular_values_sigma
        self._right_low_rank_matrix_v = right_low_rank_matrix_v

        self._matrix_l0 = self._left_low_rank_matrix_u[:, :rank_k]
        self._matrix_r0 = np.diag(self._singular_values_sigma[:rank_k]).dot(
            right_low_rank_matrix_v[:rank_k, :])
        self._l_cs.value = self._left_low_rank_matrix_u[:, :rank_k]
        self._r_cs.value = np.diag(self._singular_values_sigma[:rank_k]).dot(
            self._right_low_rank_matrix_v[:rank_k, :])

        self._residuals_median = None
        self._residuals_variance = None
        self._residual_l0_norm = None

        self._linearization_helper = LinearizationHelper(
            solver_type=self._solver_type)

        self._weight_setting = WeightSetting(solver_type=self._solver_type)

        self._set_testdays(power_signals_d, reserve_test_data)

        # Stores the current state of the object:
        self._state_data = StateData()
        self._store_initial_state_data()

    def minimize_objective(self, mu_l=1.0, mu_r=20.0, tau=0.8,
                           eps=1e-3, max_iter=100, calc_deg=True,
                           max_deg=None, min_deg=None,
                           verbose=True):

        self._obtain_component_r0()
        self._obtain_weights()

        self._minimization_state_data(mu_l, mu_r, tau)

        ti = time()
        try:
            obj_vals = self._calculate_objective(mu_l, mu_r, tau,
                                                 sum_components=False)
            if verbose:
                print('starting at {:.3f}'.format(np.sum(obj_vals)), obj_vals)
            improvement = np.inf
            old_obj = np.sum(obj_vals)
            it = 0
            f1_last = obj_vals[0]
            while improvement >= eps:
                if self._test_days is not None:
                    self._weights[self.test_days] = 0
                self.min_l()
                self.min_r(calc_deg=calc_deg, max_deg=max_deg, min_deg=min_deg)
                obj_vals = self._calculate_objective(mu_l, mu_r, tau,
                                                     sum_components=False)
                new_obj = np.sum(obj_vals)
                improvement = (old_obj - new_obj) * 1. / old_obj
                old_obj = new_obj
                it += 1
                if verbose:
                    print('iteration {}: {:.3f}'.format(it, new_obj), np.round(obj_vals, 3))
                if obj_vals[0] > f1_last:
                    self._state_data.f1_increase = True
                    if verbose:
                        print('Caution: residuals increased')
                if improvement < 0:
                    if verbose:
                        print('Caution: objective increased.')
                    self._state_data.obj_increase = True
                    improvement *= -1
                if it >= max_iter:
                    if verbose:
                        print('Reached iteration limit. Previous improvement: {:.2f}%'.format(improvement * 100))
                    improvement = 0.
        except cvx.SolverError:
            if verbose:
                print('solver failed!')
            self._state_data.is_solver_error = True
        except ProblemStatusError as e:
            if verbose:
                print(e)
            self._state_data.is_problem_status_error = True
        else:
            tf = time()
            if verbose:
                print('Minimization complete in {:.2f} minutes'.format((tf - ti) / 60.))
            # Residual analysis
            weights1 = np.diag(self._weights)
            wres = np.dot(self._l_cs.value.dot(
                self._r_cs.value) - self._power_signals_d, weights1)
            use_days = np.logical_not(np.isclose(np.sum(wres, axis=0), 0))
            scaled_wres = wres[:, use_days] / np.average(self._power_signals_d[:, use_days])
            final_metric = scaled_wres[self._power_signals_d[:, use_days] > 1e-3]
            self._residuals_median = np.median(final_metric)
            self._residuals_variance = np.power(np.std(final_metric), 2)
            self._residual_l0_norm = np.linalg.norm(
                self._metrix_l0[:, 0] - self._l_cs.value[:, 0]
            )

        self._store_final_state_data()

    def _calculate_objective(self, mu_l, mu_r, tau, sum_components=True):
        weights1 = np.diag(self._weights)
        form1 = (cvx.sum((0.5 * cvx.abs(
            self._power_signals_d - self._l_cs.value.dot(self._r_cs.value))
                              + (tau - 0.5) * (self._power_signals_d - self._l_cs.value.dot(self._r_cs.value))) * weights1)).value
        weights2 = np.eye(self._rank_k)
        form2 = mu_l * norm(((self._l_cs[:-2, :]).value -
                               2 * (self._l_cs[1:-1, :]).value +
                               (self._l_cs[2:, :]).value).dot(weights2), 'fro')
        form3 = mu_r * norm((self._r_cs[:, :-2]).value -
                              2 * (self._r_cs[:, 1:-1]).value +
                              (self._r_cs[:, 2:]).value, 'fro')
        if self._r_cs.shape[1] < 365 + 2:
            form4 = 0
        else:
            form4 = (mu_r * cvx.norm(self._r_cs[1:, :-365] - self._r_cs[1:, 365:], 'fro')).value
        components = [form1, form2, form3, form4]
        objective = sum(components)
        if sum_components:
            return objective
        else:
            return components

    def _adjust_low_rank_matrices(self, left_low_rank_matrix_u,
                                  right_low_rank_matrix_v):

        if np.sum(left_low_rank_matrix_u[:, 0]) < 0:
            left_low_rank_matrix_u[:, 0] *= -1
            right_low_rank_matrix_v[0] *= -1

        return left_low_rank_matrix_u, right_low_rank_matrix_v

    def _obtain_component_r0(self):
        self._component_r0 = self._linearization_helper.obtain_component_r0(
            self._power_signals_d, self._left_low_rank_matrix_u,
            self._singular_values_sigma, self._right_low_rank_matrix_v,
            rank_k=self._rank_k)

    def _obtain_weights(self):
        self._weights = self._weight_setting.obtain_weights(
            self._power_signals_d)

    def _set_testdays(self, power_signals_d, reserve_test_data):
        if reserve_test_data:
            m, n = power_signals_d.shape
            day_indices = np.arange(n)
            num = int(n * reserve_test_data)
            self._test_days = np.sort(np.random.choice(day_indices, num,
                                                       replace=False))
        else:
            self._test_days = None

    def _store_initial_state_data(self):
        self._state_data.power_signals_d = self._power_signals_d
        self._state_data.rank_k = self._rank_k
        self._state_data.matrix_l0 = self._matrix_l0
        self._state_data.matrix_r0 = self._matrix_r0
        self._state_data.l_value = self._l_cs.value
        self._state_data.r_value = self._r_cs.value

    def _minimization_state_data(self, mu_l, mu_r, tau):
        self._state_data.mu_l = mu_l
        self._state_data.mu_r = mu_r
        self._state_data.tau = tau

    def _store_final_state_data(self):
        self._state_data.l_value = self._l_cs.value
        self._state_data.r_value = self._r_cs.value
