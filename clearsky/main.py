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

    def make_clearsky_model(self, n=5, mu1=3.5, eta=1.5, mu2=3, plot=False, return_fits=False, env_fit=0):
        if self.U is None:
            self.get_eigenvectors()
        daily_scale_factors = ((np.diag(self.D).dot(self.P[:288])))
        signals = [None] * n
        fits = np.empty((n, self.P.shape[1]))
        ind = env_fit
        signal = daily_scale_factors[ind, :]
        fit = envelope_fit(signal, mu=10 ** mu1, eta=10 ** eta, kind='lower', period=365)
        mask = np.abs(signal - fit) < 1.5
        signals[ind] = signal
        fits[ind, :] = fit
        for ind in xrange(n):
            if ind != env_fit:
                signal = daily_scale_factors[ind, :]
                mu_i = mu2
                fit = masked_smooth_fit_periodic(signal, mask, 365, mu=10**mu_i)
                signals[ind] = signal
                fits[ind, :] = fit
        self.DP_clearsky = fits[:, :365]
        self.cleardays = np.arange(self.P.shape[1])[mask]
        if plot:
            fig, axes = plt.subplots(nrows=n, figsize=(8,n*3), sharex=True)
            try:
                for ind in xrange(n):
                    axes[ind].plot(signals[ind], linewidth=1)
                    axes[ind].plot(fits[ind], linewidth=1)
                    axes[ind].set_title('Daily scale factors for singular vector {}'.format(ind+1))
                    axes[ind].scatter(np.nonzero(mask), signals[ind][mask], marker='.', color='yellow', alpha=0.7)
                axes[ind].set_xlabel('Day Number')
                plt.tight_layout()
                return fig, axes
            except TypeError:
                axes.plot(signals[0], linewidth=1)
                axes.plot(fits[0], linewidth=1)
                axes.set_xlabel('Day Number')
                axes.set_title('Daily scale factors for singular vector 1')
                axes.scatter(np.nonzero(mask), signals[ind][mask], marker='.', color='yellow', alpha=0.7)
                return fig, axes
        if return_fits:
            return signals, fits

    def estimate_clearsky(self, day_slice):
        '''
        Make a clearsky estimate based on provided data for a given set of days
        :param day_slice: A numpy.slice object indicating the days to be used in the estimation (see: numpy.s_)
        :return: A matrix of daily clear sky estimates. The columns are individual days and the rows 5-minute
        time periods
        '''
        if self.DP_clearsky is None:
            self.make_clearsky_model()
        n = self.DP_clearsky.shape[0]
        clearsky = self.U[:, :n].dot(self.DP_clearsky[:, day_slice])
        return clearsky.clip(min=0)