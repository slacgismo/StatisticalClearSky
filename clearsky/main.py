# -*- coding: utf-8 -*-
"""
This module contains the algorithm to statistically fit a clear sky signal.
"""

from clearsky.utilities import envelope_fit, envelope_fit_with_deg, masked_smooth_fit_periodic
import numpy as np
from numpy.linalg import svd, matrix_rank, eigh, inv, norm
import matplotlib.pyplot as plt
from datetime import date, datetime
from time import time
import cvxpy as cvx

# Python 2.x, 3.x compatibility
try:
    xrange
except NameError:
    xrange = range


class IterativeClearSky(object):
    def __init__(self, D, k=8, deg=True, adaptive_weighting=True):
        self.D = D
        self.k = k
        self.L_cs = cvx.Variable(D.shape[0], k)
        self.R_cs = cvx.Variable(k, D.shape[1])
        self.C = cvx.Variable(*D.shape)
        self.d = cvx.Variable(D.shape[1])
        U, Sigma, V = np.linalg.svd(D)
        if np.sum(U[:, 0]) < 0:
            U[:, 0] *= -1
            V[0] *= -1
        self.L_cs.value = U[:, :k]
        self.R_cs.value = np.diag(Sigma[:k]).dot(V[:k, :])
        self.C.value = np.ones_like(D)
        self.d.value = np.ones(D.shape[1])
        self.mu_L = 1.
        self.mu_R = 20.
        self.mu_C = 0.05
        self.tau = 0.8
        self.theta = 0.1
        self.weights = np.ones(D.shape[1])
        tc = np.linalg.norm(D[:-2] - 2 * D[1:-1] + D[2:], ord=1, axis=0)
        tc = np.percentile(tc, 50) - tc
        tc /= np.max(tc)
        tc = np.clip(tc, 0, None)
        self.daily_tc_weights = tc
        self.deg = deg
        self.adaptive_weighting = adaptive_weighting

    def calc_objective(self, sum_components=True):
        W1 = np.diag(self.weights)
        f1 = norm(((self.D -
                  cvx.mul_elemwise(cvx.pos(self.L_cs.value * self.R_cs.value), self.C.value) *
                   cvx.diag(self.d.value.A1)) * W1).value, 'fro')
        W2 = np.eye(self.k)
        W2[0, 0] = 10
        f2 = self.mu_L * norm(((self.L_cs[:-2, :]).value -
                               2 * (self.L_cs[1:-1, :]).value +
                               (self.L_cs[2:, :]).value) * W2, 'fro')
        f3 = self.mu_R * norm((self.R_cs[:, :-2]).value -
                              2 * (self.R_cs[:, 1:-1]).value +
                              (self.R_cs[:, 2:]).value, 'fro')
        if self.D.shape[1] > 365:
            f4 = self.mu_R * norm((self.R_cs[1:, :-365]).value - (self.R_cs[1:, 365:]).value, 'fro')
        else:
            f4 =0
        f5 = self.mu_C * (
            cvx.sum_entries((0.5 * cvx.abs((self.C).value - 1) + (self.tau - 0.5) * ((self.C).value - 1)) * W1)).value
        #f6 = self.mu_C * 10 * cvx.abs(cvx.sum_entries(cvx.sum_entries(self.C, axis=0).T[:-1]
        #                                              - cvx.sum_entries(self.C, axis=0).T[1:])).value
        f6 = 0 # (0.1 * self.mu_C * cvx.norm(self.C.value, 2)).value
        if self.D.shape[1] > 365:
            f7 = self.mu_C * 10 * cvx.abs(cvx.sum_entries(cvx.sum_entries(self.C, axis=0).T[:-365]
                                                          - cvx.sum_entries(self.C, axis=0).T[365:])).value
        else:
            f7 = 0
        components = [f1, f2, f3, f4, f5, f6, f7]
        objective = sum(components)
        if sum_components:
            return objective
        else:
            return components

    def minimize_objective(self, eps=1e-3, max_iter=100):
        ti = time()
        print 'starting at {:.3f}'.format(self.calc_objective()), self.calc_objective(False)
        improvement = np.inf
        old_obj = self.calc_objective()
        it = 0
        while improvement >= eps:
            if self.adaptive_weighting:
                wf  = np.linalg.norm(self.D - self.L_cs.value.dot(self.R_cs.value), axis=0)
                wf = np.percentile(wf, 90) - wf
                wf /= np.max(wf)
                wf = np.clip(wf, 0, None)
                tc = self.daily_tc_weights
                self.weights = np.multiply(np.power(wf, self.theta), np.power(tc, 1 - self.theta))
            else:
                self.weights = np.ones(self.D.shape[1])
            self.min_L()
            print self.calc_objective(False)[0]
            self.min_R()
            print self.calc_objective(False)[0]
            self.min_C()
            print self.calc_objective(False)[0]
            if self.deg:
                self.min_d()
            new_obj = self.calc_objective()
            improvement = (old_obj - new_obj) * 1. / old_obj
            old_obj = new_obj
            it += 1
            print 'iteration {}: {:.3f}'.format(it, new_obj), np.round(self.calc_objective(False), 3)
            if improvement < 0:
                print 'Objective increased.'
                improvement *= -1
            if it >= max_iter:
                print 'Reached iteration limit. Previous improvement: {:.2f}%'.format(improvement * 100)
                improvement = 0.
        tf = time()
        print 'Minimization complete in {:.2f} minutes'.format((tf - ti) / 60.)

    def min_L(self):
        W1 = np.diag(self.weights)
        f1 = cvx.norm((self.D
                      - cvx.mul_elemwise(self.C.value,
                                         self.L_cs * self.R_cs.value)
                      * cvx.diag(self.d.value)) * W1, 'fro')
        W2 = np.eye(self.k)
        W2[0, 0] = 10
        f2 = self.mu_L * cvx.norm((self.L_cs[:-2, :] - 2 * self.L_cs[1:-1, :] + self.L_cs[2:, :]) * W2, 'fro')
        objective = cvx.Minimize(f1 + f2)
        foo = cvx.sum_entries(self.L_cs * self.R_cs.value, axis=0).T
        constraints = [
            self.L_cs * self.R_cs.value >= 0
            #cvx.norm(foo[:-2] + 2 * foo[1:-1] - foo[2:]) <= 1e4
        ]
        problem = cvx.Problem(objective, constraints)
        problem.solve(solver='MOSEK')

    def min_R(self):
        W1 = np.diag(self.weights)
        f1 = cvx.norm((self.D
                      - cvx.mul_elemwise(self.C.value,
                                         self.L_cs.value * self.R_cs)
                      * cvx.diag(self.d.value)) * W1, 'fro')
        f2 = self.mu_R * cvx.norm(self.R_cs[:, :-2] - 2 * self.R_cs[:, 1:-1] + self.R_cs[:, 2:], 'fro')
        if self.D.shape[1] > 365:
            f3 = self.mu_R * cvx.norm(self.R_cs[1:, :-365] - self.R_cs[1:, 365:], 'fro')
        else:
            f3 = 0
        objective = cvx.Minimize(f1 + f2 + f3)
        foo = cvx.sum_entries(self.L_cs.value * self.R_cs, axis=0).T
        constraints = [
            self.L_cs.value * self.R_cs >= 0
            #cvx.norm(foo[:-2] + 2 * foo[1:-1] - foo[2:]) <= 1e4
        ]
        if self.D.shape[1] > 365:
            beta = cvx.Variable()
            constraints.append(self.R_cs[0, :-365] - self.R_cs[0, 365:] == beta)
        problem = cvx.Problem(objective, constraints)
        problem.solve(solver='MOSEK')

    def min_C(self):
        W1 = np.diag(self.weights)
        f1 = cvx.norm((self.D
                      - cvx.mul_elemwise(cvx.pos(self.L_cs.value * self.R_cs.value),
                                         self.C)
                      * cvx.diag(self.d.value)) * W1, 'fro')
        f2 = self.mu_C * cvx.sum_entries((0.5 * cvx.abs(self.C - 1) + (self.tau - 0.5) * (self.C - 1)) * W1)
        #f3 = self.mu_C * 10 * cvx.abs(cvx.sum_entries(cvx.sum_entries(self.C, axis=0).T[:-1]
        #                                              - cvx.sum_entries(self.C, axis=0).T[1:]))
        f3 = 0 # 0.1 * self.mu_C * cvx.norm(self.C, 2)
        if self.D.shape[1] > 365:
            f4 = self.mu_C * 10 * cvx.abs(cvx.sum_entries(cvx.sum_entries(self.C, axis=0).T[:-365]
                                                          - cvx.sum_entries(self.C, axis=0).T[365:]))
        else:
            f4 = 0
        objective = cvx.Minimize(f1 + f2 + f3 + f4)
        # constraints = [
        #    cvx.sum_entries(cvx.sum_entries(self.C, axis=0).T[:-1] - cvx.sum_entries(self.C, axis=0).T[1:]) <= 0,
        #    cvx.sum_entries(cvx.sum_entries(self.C, axis=0).T[:-365] - cvx.sum_entries(self.C, axis=0).T[365:]) <= 0
        # ]
        problem = cvx.Problem(objective)
        problem.solve(solver='MOSEK')

    def min_d(self):
        f1 = cvx.norm(self.D
                      - cvx.mul_elemwise(cvx.pos(self.L_cs.value * self.R_cs.value),
                                         self.C.value)
                      * cvx.diag(self.d), 'fro')
        objective = cvx.Minimize(f1)
        constraints = [
            self.d[0] == 1.,
            self.d[:-1] >= self.d[1:],
            self.d[:-2] - 2 * self.d[1:-1] + self.d[2:] == 0
        ]
        problem = cvx.Problem(objective, constraints)
        problem.solve(solver='MOSEK')


class StatisticalClearSky(object):
    def __init__(self, data, start_date=None):
        self.data = data
        self.U = None
        self.D = None
        self.P = None
        self.R = None
        self.U_tilde = None
        self.R_clearsky = None
        self.cleardays = None
        self.deg_rate = None
        self.selected_vectors = None
        if start_date is not None:
            if isinstance(start_date, date):
                self.start_date = start_date
            else:
                try:
                    self.start_date = datetime.strptime(start_date, '%m/%d/%y').date()
                except ValueError:
                    print "Please input date string in '%m/%d/%y' format"
        else:
            try:
                self.start_date = data.resample('D').max().index[0].date()
            except AttributeError:
                self.start_date = None

    def get_eigenvectors(self, data_as_matrix=False, nrows=288, smooth=False, mu_val=15):
        if not data_as_matrix:
            data_matrix = self.data.as_matrix().reshape(-1, nrows).T
        else:
            data_matrix = self.data
        data_rank = matrix_rank(data_matrix)
        U, D, P = svd(data_matrix)
        if np.sum(U[:, 0]) < 0:
            U[:, 0] *= -1
            P[0] *= -1
        if smooth:
            U_smooth = []
            done = False
            i = 0
            skipped = 0
            skipped_consecutive = 0
            self.selected_vectors = []
            while not done:
                if i == 0:
                    vec = U[:, i]
                    vec_smooth = cvx.Variable(len(vec))
                    mu = cvx.Parameter(value=mu_val, sign='positive')
                    objective = cvx.Minimize(
                    cvx.norm(vec - vec_smooth) + mu * cvx.norm(vec_smooth[:-1] - vec_smooth[1:]))
                    problem = cvx.Problem(objective)
                    try:
                        problem.solve(solver='MOSEK')
                    except Exception as e:
                        print(e)
                        print('Trying ECOS solver')
                        problem.solve(solver='ECOS')
                    U_smooth.append(vec_smooth.value.A1 / np.linalg.norm(vec_smooth.value.A1))
                    self.selected_vectors.append(i)
                    i += 1
                    continue
                vec = U[:, i]
                vec_smooth = cvx.Variable(len(vec))
                mu = cvx.Parameter(value=mu_val, sign='positive')
                objective = cvx.Minimize(
                    cvx.norm(vec - vec_smooth) + mu * cvx.norm(vec_smooth[:-2] - 2 * vec_smooth[1:-1] + vec_smooth[2:])
                )
                constraints = []
                for j in xrange(i - skipped):
                    constraints.append(U_smooth[j] * vec_smooth == 0)
                problem = cvx.Problem(objective, constraints)
                try:
                    problem.solve(solver='MOSEK')
                except Exception as e:
                    print(e)
                    print('Trying ECOS solver')
                    problem.solve(solver='ECOS')
                err = np.linalg.norm(vec - vec_smooth.value.A1)
                if err < 0.8:
                    U_smooth.append(vec_smooth.value.A1 / np.linalg.norm(vec_smooth.value.A1))
                    self.selected_vectors.append(i)
                    skipped_consecutive = 0
                    i += 1
                else:
                    skipped += 1
                    skipped_consecutive += 1
                    if skipped_consecutive >= U.shape[1] / 10:
                        done = True
                        U_smooth = np.r_[U_smooth].T
                        print '{} vectors selected'.format(U_smooth.shape[1])
                        print '{} vectors skipped'.format(skipped - skipped_consecutive)
                    else:
                        i += 1
            # M = data_matrix.dot(data_matrix.T)
            # lambd = np.diag(U_smooth.T.dot(M).dot(U_smooth))
            # M1 = M - U_smooth.dot(np.diag(lambd)).dot(U_smooth.T)
            # L, Q = eigh(M1)
            # U_tilde = np.concatenate([U_smooth, Q[:, ::-1]], axis=1)[:, :data_rank]
            U_tilde = U_smooth
            R_tilde = inv(U_tilde.T.dot(U_tilde)).dot(U_tilde.T).dot(data_matrix)
            self.U_tilde = U_tilde
            self.R = R_tilde
        else:
            self.R = np.diag(D[:data_rank]).dot(P[:data_rank])
            self.U_tilde = None
        self.U = U
        self.D = D
        self.P = P
        self.data = data_matrix

    def reconstruct_day(self, day=20, n=100, plot=True, figsize=None):
        if self.U is None:
            self.get_eigenvectors()
        if plot:
            if figsize is not None:
                plt.figure(figsize=figsize)
            plt.plot(self.data[:, day], linewidth=1)
            plt.plot(self.U[:, :n].dot(np.diag(self.D[:n])).dot(self.P[:n, day]), linewidth=1)
        else:
            return self.data[:, day], self.U[:, :n].dot(np.diag(self.D[:n])).dot(self.P[:n, day])

    def make_clearsky_model(self, n=None, mu1=3.5, eta=1.5, mu2=3, deg=False, plot=False, return_fits=False,
                            mu_schedule=False, window=0.025):
        env_fit = 0
        if self.U is None:
            self.get_eigenvectors()
        daily_scale_factors = self.R
        if n is None:
            n = daily_scale_factors.shape[0]
        padded = False
        if daily_scale_factors.shape[1] == 365:
            padded = True
            daily_scale_factors = np.c_[daily_scale_factors, daily_scale_factors[:, :2]]
        fits = np.empty((n, daily_scale_factors.shape[1]))
        ind = env_fit
        signal = daily_scale_factors[ind, :]
        if not deg:
            fit = envelope_fit(signal, mu=10 ** mu1, eta=10 ** eta, period=365)
            self.deg_rate = None
        else:
            fit, beta = envelope_fit_with_deg(signal, period=365, mu=10 ** mu1, eta=10 ** eta)
            self.deg_rate = 1 - 1. / beta
        mask = np.abs(signal - fit) / signal < window
        fits[ind, :] = fit
        mu_i = mu2
        for ind in xrange(n):
            if ind != env_fit:
                signal = daily_scale_factors[ind, :]
                fit = masked_smooth_fit_periodic(signal, mask, 365, mu=10**mu_i, linear_term=False)
                # if deg_terms:
                #     fit, beta = fit
                #     self.betas.append(beta)
                fits[ind, :] = fit
                if mu_schedule:
                    mu_i *= .995 # observed that later fits require a bit more flexibility
        if padded:
            fits = fits[:, :-2]
            mask = mask[:-2]
        self.R_clearsky = fits
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
        signals = self.R
        fits = self.R_clearsky
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
        if self.R_clearsky is None:
            self.make_clearsky_model()
        if n is None:
            n = self.R_clearsky.shape[0]
        else:
            n = min(n, self.R_clearsky.shape[0])
        if self.U_tilde is not None:
            U = self.U_tilde
        else:
            U = self.U
        clearsky = U[:, :n].dot(self.R_clearsky[:n, day_slice])
        return clearsky.clip(min=0)

    def predict_day(self, dt, n=None):
        '''
        Predict a future clear sky day, out of the time window of the observed data. This only gives a different
        result than estimate_clearsky if a deg rate is estimated. However, there is an additional difference:
        estimate_clearsky uses a "day index" to select the days to estimate, which references the column index of the
        original data. This method references an actual date, either as a python datetime.date object or a string in
        '%m/%d/%y' format. To use this method, the 'self.start_date' attribute must be set.
        :param dt: The date
        :param n:
        :return:
        '''
        if self.R_clearsky is None:
            self.make_clearsky_model()
        if n is None:
            n = self.R_clearsky.shape[0]
        else:
            n = min(n, self.R_clearsky.shape[0])
        if self.U_tilde is not None:
            U = self.U_tilde
        else:
            U = self.U
        if not isinstance(dt, date):
            try:
                dt = datetime.strptime(dt, '%m/%d/%y').date()
            except ValueError:
                print "Please input date string in '%m/%d/%y' format"
        day_ix = (dt - self.start_date).days % 365
        if self.deg_rate is not None:
            year_ix = (dt - self.start_date).days / 365
            r = self.deg_rate
        else:
            r = 0
            year_ix = 0
        scale = np.ones(n)
        scale[0] = (1. - r) ** year_ix
        clearsky = U[:, :n].dot(self.R_clearsky[:n, day_ix] * scale)
        return clearsky.clip(min=0)