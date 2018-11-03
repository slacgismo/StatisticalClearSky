# -*- coding: utf-8 -*-
"""
This module contains utility functions and classes.
"""
import cvxpy as cvx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Python 2.x, 3.x compatibility
try:
    xrange
except NameError:
    xrange = range

class ProblemStatusError(Exception):
    """Error thrown when SCSF algorithm experiences something other than an 'optimal' problem status during one of
    the solve steps."""
    pass


def envelope_fit(signal, mu, eta, linear_term=False, kind='upper', period=None):
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
    if linear_term:
        beta = cvx.Variable()
    else:
        beta = 0.
    envelope = cvx.Variable(len(signal))
    mu = cvx.Parameter(sign='positive', value=mu)
    eta = cvx.Parameter(sign='positive', value=eta)
    cost = (cvx.sum_entries(cvx.huber(envelope - signal)) +
            mu * cvx.norm2(envelope[2:] - 2 * envelope[1:-1] + envelope[:-2]) +
            eta * cvx.norm1(cvx.max_elemwise(signal - envelope, 0)))
    objective = cvx.Minimize(cost)
    if period is not None:
        constraints = [
            envelope[:n_samples - period] == envelope[period:] + beta
        ]
        problem = cvx.Problem(objective, constraints)
    else:
        problem = cvx.Problem(objective)
    try:
        problem.solve(solver='MOSEK')
    except Exception as e:
        print(e)
        print('Trying ECOS solver')
        problem.solve(solver='ECOS')
    if not linear_term:
        if kind == 'upper':
            return envelope.value.A1
        elif kind == 'lower':
            signal *= -1
            return -envelope.value.A1
    else:
        if kind == 'upper':
            return envelope.value.A1, beta.value
        elif kind == 'lower':
            signal *= -1
            return -envelope.value.A1, -beta.value

def envelope_fit_with_deg(signal, period, mu, eta, kind='upper', eps=1e-4):
    '''
    Perform an approximately periodic envelope fit of a signal.
    See: https://en.wikipedia.org/wiki/Envelope_(waves)
    :param signal: The signal to be fit
    :param mu: A parameter to control the overall stiffness of the fit
    :param eta: A parameter to control the permeability of the envelope. A large value result in
    no data points outside the fitted envelope
    :param kind: 'upper' or 'lower'
    :return: An envelope signal of the same length as the input
    '''
    offset_eps = False
    if kind == 'lower':
        signal *= -1
    if np.min(signal) < 0:
        print('function only works for signals with all values >= 0')
    elif np.min(signal) == 0:
        offset_eps = True
        signal += eps
    n_samples = len(signal)
    beta = cvx.Variable()
    envelope = cvx.Variable(len(signal))
    mu = cvx.Parameter(sign='positive', value=mu)
    eta = cvx.Parameter(sign='positive', value=eta)
    cost = (cvx.sum_entries(cvx.huber(envelope - np.log(signal))) +
            mu * cvx.norm2(envelope[2:] - 2 * envelope[1:-1] + envelope[:-2]) +
            eta * cvx.norm1(cvx.max_elemwise(np.log(signal) - envelope, 0)))
    objective = cvx.Minimize(cost)
    constraints = [
        envelope[:n_samples - period] == envelope[period:] + beta
    ]
    problem = cvx.Problem(objective, constraints)
    try:
        problem.solve(solver='MOSEK')
    except Exception as e:
        print(e)
        print('Trying ECOS solver')
        problem.solve(solver='ECOS')
    if offset_eps:
        signal -= eps
    if kind == 'upper':
        return np.exp(envelope.value.A1), np.exp(beta.value)
    elif kind == 'lower':
        signal *= -1
        return -np.exp(envelope.value.A1), np.exp(beta.value)

def masked_smooth_fit_periodic(signal, mask, period, mu, linear_term=False):
    n_samples = len(signal)
    if linear_term:
        beta = cvx.Variable()
    else:
        beta = 0.
    fit = cvx.Variable(n_samples)
    mu = cvx.Parameter(sign='positive', value=mu)
    cost = (cvx.sum_entries(cvx.huber(fit[mask] - signal[mask]))
            + mu * cvx.norm2(fit[2:] - 2 * fit[1:-1] + fit[:-2]))
    objective = cvx.Minimize(cost)
    constraints = [fit[:len(signal) - period] == fit[period:] + beta]
    problem = cvx.Problem(objective, constraints)
    try:
        problem.solve(solver='MOSEK')
    except Exception as e:
        print(e)
        print('Trying ECOS solver')
        problem.solve(solver='ECOS')
    if not linear_term:
        return fit.value.A1
    else:
        return fit.value.A1, beta.value

def make_time_series(df, return_keys=True, localize_time=-8, filter_length=200):
    '''
    Accepts a Pandas data frame extracted from the Cassandra database. Returns a data frame with a single timestamp
    index and the data from different systems split into columns.
    :param df: A Pandas data from generated from a CQL query to the VADER Cassandra database
    :param return_keys: If true, return the mapping from data column names to site and system ID
    :param localize_time: If non-zero, localize the time stamps. Default is PST or UTC-8
    :param filter_length: The number of non-null data values a single system must have to be included in the output
    :return: A time-series data frame
    '''
    df.sort_values('ts', inplace=True)
    start = df.iloc[0]['ts']
    end = df.iloc[-1]['ts']
    time_index = pd.date_range(start=start, end=end, freq='5min')
    output = pd.DataFrame(index=time_index)
    site_keys = []
    site_keys_a = site_keys.append
    grouped = df.groupby(['site', 'sensor'])
    keys = grouped.groups.keys()
    counter = 1
    for key in keys:
        df_view = df.loc[grouped.groups[key]]
        ############## data cleaning ####################################
        df_view = df_view[pd.notnull(df_view['meas_val_f'])]            # Drop records with nulls
        df_view.set_index('ts', inplace=True)                           # Make the timestamp column the index
        df_view.sort_index(inplace=True)                                # Sort on time
        df_view = df_view[~df_view.index.duplicated(keep='first')]      # Drop duplicate times
        df_view.reindex(index=time_index).interpolate()                 # Match the master index, interp missing
        #################################################################
        meas_name = str(df_view['meas_name'][0])
        col_name = meas_name + '_{:02}'.format(counter)
        output[col_name] = df_view['meas_val_f']
        if output[col_name].count() > filter_length:  # final filter on low data count relative to time index
            site_keys_a((key, col_name))
            counter += 1
        else:
            del output[col_name]
    if localize_time:
        output.index = output.index + pd.Timedelta(hours=localize_time)  # Localize time

    if return_keys:
        return output, site_keys
    else:
        return output


def lowpass_2d(D, r=25):
    FS = np.fft.fft2(D)
    fltr = np.zeros_like(D, dtype=np.float)
    m, n = D.shape
    c = (m // 2, n // 2)
    if m % 2 == 0:
        di = 0
    else:
        di = 1
    if n % 2 == 0:
        dj = 0
    else:
        dj = 1
    y, x = np.ogrid[-c[0]:c[0] + di, -c[1]:c[1] + dj]
    mask = x ** 2 + y ** 2 <= r ** 2
    fltr[mask] = 1
    FS_filtered = np.fft.fftshift(np.multiply(np.fft.fftshift(FS), fltr))
    D_filtered = np.abs(np.fft.ifft2(FS_filtered))
    return D_filtered


def fix_time_shifts(D, w=30, tol=5e-2, verbose=False):
    # For each day, weight each time step by the power produced in that period, and take the average. This gives us
    # the "center" time point for each day, in terms of hours between 0 and 23 inclusive
    old_err_state = np.seterr(divide='ignore')
    s1 = np.divide(np.dot(np.linspace(0, 24, D.shape[0]), D), np.sum(D, axis=0))
    # Apply a sliding-window average filter
    s2 = np.convolve(s1, np.ones(w), mode='valid') / w
    # Apply 1D edge finding algorithm
    ixs = edge_find_1D(s2, tol=tol)
    if len(ixs) == 0:
        if verbose:
            print('No time shifts found')
        return D
    ixs = list(map(lambda x: int(x + w / 2), ixs))
    num = len(ixs)
    if verbose:
        print('{} segments found'.format(num + 1))
        print('index locations: ', ixs)
    ixs = np.r_[[None], ixs, [None]]
    A = []
    for i in range(len(ixs) - 1):
        avg = np.average(np.ma.masked_invalid(s1[ixs[i]:ixs[i+1]]))
        A.append(np.round(avg * D.shape[0] / 24))
    A = np.array(A)
    rolls = A[0] - A[1:]
    Dout = np.copy(D)
    for ind, roll in enumerate(rolls):
        D_rolled = np.roll(D, int(roll), axis=0)
        Dout[:, ixs[ind + 1]:] = D_rolled[:, ixs[ind + 1]:]
    return Dout



def edge_find_1D(s1, tol=5e-2, ixs=None, ix0=0, w=30, mu=3, debug=False):
    # Returns the indices of edges in a 1-D array. This algorithm recursively segments the input array until all edges
    # have been found.
    if ixs is None:
        ixs = []
    x = cvx.Variable(len(s1))
    mu = cvx.Constant(mu)
    obj = cvx.Minimize(cvx.norm(s1[np.isfinite(s1)] - x[np.isfinite(s1)]) + mu * cvx.norm1(x[:-1] - x[1:]))
    prob = cvx.Problem(obj)
    prob.solve(solver='MOSEK')
    if debug:
        plt.plot(x.value)
        plt.show()
    s2 = np.abs(x.value[:-1] - x.value[1:])
    if debug:
        print(s2.max() - s2.min())
    if s2.max() - s2.min() < tol:
        # There are no step shifts in this data segment
        return ixs
    else:
        # There is a step shift in this data segment
        ix = np.argsort(-s2)[0]
        vr_best = -np.inf
        for j in xrange(ix - w, ix + w):
            jx = max(0, j)
            jx = min(jx, len(s1))
            sa = s1[:jx][np.isfinite(s1)[:jx]]
            sb = s1[jx:][np.isfinite(s1)[jx:]]
            vr = (np.std(s1[np.isfinite(s1)]) ** 2
                  - (len(sa) / len(s1[np.isfinite(s1)])) * np.std(sa)
                  - (len(sb) / len(s1[np.isfinite(s1)])) * np.std(sb))
            if vr > vr_best:
                vr_best = vr
                ix_best = jx
        ixs.append(ix_best + ix0)
        ixs1 = edge_find_1D(s1[:ix_best], tol=tol, ixs=ixs, ix0=ix0)
        ixs2 = edge_find_1D(s1[ix_best:], tol=tol, ixs=ixs1, ix0=ix0+ix_best)
        ixs2.sort()
        return ixs2