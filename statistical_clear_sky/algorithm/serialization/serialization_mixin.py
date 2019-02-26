"""
This module defines Mixin for serialization.
"""
from statistical_clear_sky.algorithm.serialization.state_data import StateData

class SerializationMixin(object):

    def __init__(self):
        self._state_data = StateData()

    @staticmethod
    def save_instance(self, filepath):
        save_dict = dict(
            power_signals_d = self._state_data.power_signals_d.tolist(),
            rank_k = self._state_data.rank_k,
            matrix_l0 = self._state_data.matrix_l0.tolist(),
            matrix_r0 = self._state_data.matrix_r0.tolist(),
            l_value = self._state_data.l_value.tolist(),
            r_value = self._state_data.r_value.tolist(),
            beta_value = float(self._state_data.beta_value),
            component_r0 = self._state_data.component_r0.tolist(),
            mu_l = self._state_data.mu_l,
            mu_r = self._state_data.mu_r,
            tau = self._state_data._tau,
            is_solver_error = self._state_data.is_solver_error,
            is_problem_status_error = self._state_data.is_problem_status_error,
            f1_increase = self._state_data.f1_increase,
            obj_increase = self._state_data.obj_increase,
            residuals_median = self._state_data.residuals_median,
            residuals_variance = self._state_data.residuals_variance,
            residual_l0_norm = self._state_data.residual_l0_norm,
            weights = self._state_data.weights.tolist()
        )
        json.dump(save_dict, open(filepath, 'w'))

    @staticmethod
    def load_instance(self, filepath):
        load_dict = json.load(open(filepath))
        self.__init__(np.array(load_dict['power_signals_d'],
                      rank_k=load_dict['rank_k']))

        self._matrix_l0 = np.array(load_dict['matrix_l0'])
        self._matrix_r0 = np.array(load_dict['matrix_r0'])

        self._l_cs.value = np.array(load_dict['l_value'])
        self._r_cs.value = np.array(load_dict['r_value'])
        self._beta.value = load_dict['beta_value']

        self._component_r0 = np.array(load_dict['component_r0'])

        self._mu_l = load_dict['mu_l']
        self._mu_r = load_dict['mu_r']
        self._tau = load_dict['tau']
        self._is_solver_error = load_dict['is_solver_error']
        self._is_problem_status_error = load_dict['is_problem_status_error']
        self._f1_increase = load_dict['f1_increase']
        self._obj_increase = load_dict['obj_increase']
        self._residuals_median = load_dict['residuals_median']
        self._residuals_variance = load_dict['residuals_variance']
        self._residual_l0_norm = load_dict['residual_l0_norm']

        self._weights = np.array(load_dict['weights'])
