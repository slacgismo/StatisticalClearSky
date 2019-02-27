"""
This module defines "Statistical Clear Sky Fitting" algorithm.
"""

from time import time
import numpy as np
import cvxpy as cvx
from statistical_clear_sky.algorithm.initialization.linearization_helper\
 import LinearizationHelper
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
        self._matrix_l0 = left_low_rank_matrix_u[:, :rank_k]
        self._matrix_r0 = np.diag(singular_values_sigma[:rank_k]).dot(
            right_low_rank_matrix_v[:rank_k, :])
        self._l_cs.value = left_low_rank_matrix_u[:, :rank_k]
        self._r_cs.value = np.diag(singular_values_sigma[:rank_k]).dot(
            right_low_rank_matrix_v[:rank_k, :])

        self._mu_l = 1.
        self._mu_r = 20.
        self._tau = 0.8

        self._residuals_median = None
        self._residuals_variance = None
        self._residual_l0_norm = None

        self._linearization_helper = LinearizationHelper(
            solver_type=SolverType.ecos)

        # Stores the current state of the object:
        self._state_data = StateData()
        self._store_initial_state_data()

    def minimize_objective(self, eps=1e-3, max_iter=100, calc_deg=True,
                           max_deg=None, min_deg=None,
                           mu_l=None, mu_r=None, tau=None, verbose=True):
        if mu_l is not None:
            self._mu_l = mu_l
        if mu_r is not None:
            self._mu_r = mu_r
        if tau is not None:
            self._tau = tau
        ti = time()
        try:
            obj_vals = self.calc_objective(False)
            if verbose:
                print('starting at {:.3f}'.format(np.sum(obj_vals)), obj_vals)
            improvement = np.inf
            old_obj = np.sum(obj_vals)
            it = 0
            f1_last = obj_vals[0]
            while improvement >= eps:
                if self.test_days is not None:
                    self._weights[self.test_days] = 0
                self.min_L()
                self.min_R(calc_deg=calc_deg, max_deg=max_deg, min_deg=min_deg)
                obj_vals = self.calc_objective(sum_components=False)
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
            W1 = np.diag(self._weights)
            wres = np.dot(self._l_cs.value.dot(self._r_cs.value) - self.D, W1)
            use_days = np.logical_not(np.isclose(np.sum(wres, axis=0), 0))
            scaled_wres = wres[:, use_days] / np.average(self.D[:, use_days])
            final_metric = scaled_wres[self.D[:, use_days] > 1e-3]
            self._residuals_median = np.median(final_metric)
            self._residuals_variance = np.power(np.std(final_metric), 2)
            self._residual_l0_norm = np.linalg.norm(
                self.L0[:, 0] - self._l_cs.value[:, 0]
            )

        self._final_state_data()

    def _adjust_low_rank_matrices(self, left_low_rank_matrix_u,
                                  right_low_rank_matrix_v):

        if np.sum(left_low_rank_matrix_u[:, 0]) < 0:
            left_low_rank_matrix_u[:, 0] *= -1
            right_low_rank_matrix_v[0] *= -1

        return left_low_rank_matrix_u, right_low_rank_matrix_v

    def _store_initial_state_data(self):
        self._state_data.power_signals_d = self._power_signals_d
        self._state_data.rank_k = self._rank_k
        self._state_data.matrix_l0 = self._matrix_l0
        self._state_data.matrix_r0 = self._matrix_r0
        self._state_data.l_value = self._l_cs.value
        self._state_data.r_value = self._r_cs.value

    def _final_state_data(self):
        self._state_data.l_value = self._l_cs.value
        self._state_data.r_value = self._r_cs.value
