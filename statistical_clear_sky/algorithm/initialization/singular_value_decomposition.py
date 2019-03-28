"""
This module defines the class for Singular Value Decomposition related
 operations.
"""
import numpy as np

class SingularValueDecomposition:
    """
    Class to perform various calculations based on Sigular Value Decomposition.
    """

    def decompose(self, power_signals_d, rank_k=4):
        """
        Arguments
        ---------
        power_signals_d : numpy array
            Representing a matrix with row for dates and column for time of day,
            containing input power signals.

        Keyword arguments
        -----------------
        rank_k : integer
            Rank of the resulting low rank matrices.
        """

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

    def _adjust_low_rank_matrices(self, left_low_rank_matrix_u,
                                  right_low_rank_matrix_v):

        if np.sum(left_low_rank_matrix_u[:, 0]) < 0:
            left_low_rank_matrix_u[:, 0] *= -1
            right_low_rank_matrix_v[0] *= -1

        return left_low_rank_matrix_u, right_low_rank_matrix_v

    @property
    def left_low_rank_matrix_u(self):
        return self._left_low_rank_matrix_u

    @property
    def singular_values_sigma(self):
        return self._singular_values_sigma

    @property
    def right_low_rank_matrix_v(self):
        return self._right_low_rank_matrix_v

    @property
    def matrix_l0(self):
        return self._matrix_l0

    @property
    def matrix_r0(self):
        return self._matrix_r0
