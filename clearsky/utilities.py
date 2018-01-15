# -*- coding: utf-8 -*-
"""
This module contains utility functions and classes.
"""
import cvxpy as cvx

# Python 2.x, 3.x compatibility
try:
    xrange
except NameError:
    xrange = range

def envelope_fit(signal, mu, eta, kind='upper', period=None):
    '''
    Perform an envelope fit of a signal. See: https://en.wikipedia.org/wiki/Envelope_(waves)
    :param signal: The signal to be fit
    :param mu: A parameter to control the overall stiffness of the fit
    :param eta: A parameter to control the permeability of the envelope. A large value result in
    no data points outside the fitted envelope
    :param kind: 'upper' or 'lower'
    :return: An envelope signal of the same length as the input
    '''
    if kind == 'lower':
        signal *= -1
    n_samples = len(signal)
    envelope = cvx.Variable(len(signal))
    mu = cvx.Parameter(sign='positive', value=mu)
    eta = cvx.Parameter(sign='positive', value=eta)
    cost = (cvx.sum_entries(cvx.huber(envelope - signal)) +
            mu * cvx.norm2(envelope[2:] - 2 * envelope[1:-1] + envelope[:-2]) +
            eta * cvx.norm1(cvx.max_elemwise(signal - envelope, 0)))
    objective = cvx.Minimize(cost)
    if period is not None:
        constraints = [
            envelope[:n_samples - period] == envelope[period:]
        ]
    problem = cvx.Problem(objective, constraints)
    try:
        problem.solve(solver='MOSEK')
    except Exception as e:
        print(e)
        print('Trying ECOS solver')
        problem.solve(solver='ECOS')
    if kind == 'upper':
        return envelope.value.A1
    elif kind == 'lower':
        signal *= -1
        return -envelope.value.A1

def masked_smooth_fit_periodic(signal, mask, period, mu):
    n_samples = len(signal)
    fit = cvx.Variable(n_samples)
    mu = cvx.Parameter(sign='positive', value=mu)
    cost = (cvx.sum_entries(cvx.huber(fit[mask] - signal[mask]))
            + mu * cvx.norm2(fit[2:] - 2 * fit[1:-1] + fit[:-2]))
    objective = cvx.Minimize(cost)
    constraints = [fit[:len(signal) - period] == fit[period:]]
    problem = cvx.Problem(objective, constraints)
    try:
        problem.solve(solver='MOSEK')
    except Exception as e:
        print(e)
        print('Trying ECOS solver')
        problem.solve(solver='ECOS')
    return fit.value.A1