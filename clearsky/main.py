# -*- coding: utf-8 -*-
"""
This module contains the algorithm to statistically fit a clear sky signal.
"""

from clearsky.utilities import envelope_fit, masked_smooth_fit_periodic
import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt

# Python 2.x, 3.x compatibility
try:
    xrange
except NameError:
    xrange = range


class StatisticalClearSky(object):
    def __init__(self, data):
        self.data = data
        self.U = None
        self.D = None
        self.P = None
        self.DP_clearsky = None
        self.cleardays = None
    def get_eigenvectors(self):
        data_matrix = self.data.as_matrix().reshape(-1, 288).T
        U, D, P = svd(data_matrix)
        self.U = U
        self.D = D
        self.P = P
        self.data = data_matrix

    def reconstruct_day(self, day=20, n=100, plot=True):
        if self.U is None:
            self.get_eigenvectors()
        if plot:
            plt.plot(self.data[:, day], linewidth=1)
            plt.plot(self.U[:, :n].dot(np.diag(self.D[:n])).dot(self.P[:n, day]), linewidth=1)
        else:
            return self.data[:, day], self.U[:, :n].dot(np.diag(self.D[:n])).dot(self.P[:n, day])

    def make_clearsky_model(self, n=5, mu1=3.5, eta=1.5, mu2=3, deg_terms=False, plot=False, return_fits=False, env_fit=0):
        if self.U is None:
            self.get_eigenvectors()
        daily_scale_factors = ((np.diag(self.D).dot(self.P[:288])))
        signals = [None] * n
        fits = np.empty((n, self.P.shape[1]))
        ind = env_fit
        signal = daily_scale_factors[ind, :]
        fit = envelope_fit(signal, mu=10 ** mu1, eta=10 ** eta, kind='lower', period=365, linear_term=deg_terms)
        mask = np.abs(signal - fit) < 1.5
        signals[ind] = signal
        fits[ind, :] = fit
        for ind in xrange(n):
            if ind != env_fit:
                signal = daily_scale_factors[ind, :]
                mu_i = mu2
                fit = masked_smooth_fit_periodic(signal, mask, 365, mu=10**mu_i, linear_term=deg_terms)
                signals[ind] = signal
                fits[ind, :] = fit
        self.DP_clearsky = fits
        self.cleardays = np.arange(self.P.shape[1])[mask]
        if plot:
            pass
        if return_fits:
            pass
        return

    def plot_fits(self, selection=None):
        '''

        :param selection:   if None     --> plot first daily signal
                            if int      --> plot top n daily signals
                            if iterable --> plot daily signals at indices in interable
        :return:
        '''
        if selection is None:
            iterable = [0]
            num_fits = 1
        else:
            try:
                iterable = xrange(int(selection))
                num_fits = int(selection)
            except TypeError:
                iterable = iter(selection)
                num_fits = len(selection)
        fig, axes = plt.subplots(nrows=num_fits, figsize=(8, num_fits * 3), sharex=True)
        if num_fits == 1:
            axes = [axes]
        signals = np.diag(self.D).dot(self.P[:288])
        fits = self.DP_clearsky
        for ix1, ix2 in enumerate(iterable):
            axes[ix1].plot(signals[ix2], linewidth=1)
            axes[ix1].plot(fits[ix2], linewidth=1, color='r')
            axes[ix1].set_title('Daily scale factors for singular vector {}'.format(xrange(fits.shape[0])[ix2] + 1))
            axes[ix1].scatter(self.cleardays, signals[ix2][self.cleardays], marker='.', color='orange',
                              alpha=0.7, s=100)
        axes[ix1].set_xlabel('Day Number')
        axes[ix1].set_xlim(0, signals.shape[1])
        plt.tight_layout()
        if num_fits == 1:
            axes = axes[0]
        return fig, axes

    def estimate_clearsky(self, day_slice, n=None):
        '''
        Make a clearsky estimate based on provided data for a given set of days
        :param day_slice: A numpy.slice object indicating the days to be used in the estimation (see: numpy.s_)
        :return: A matrix of daily clear sky estimates. The columns are individual days and the rows 5-minute
        time periods
        '''
        if self.DP_clearsky is None:
            self.make_clearsky_model()
        if n is None:
            n = self.DP_clearsky.shape[0]
        clearsky = self.U[:, :n].dot(self.DP_clearsky[:n, day_slice])
        return clearsky.clip(min=0)