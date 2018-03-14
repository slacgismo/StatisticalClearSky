# -*- coding: utf-8 -*-
"""
This module contains utility functions and classes.
"""
import cvxpy as cvx
import numpy as np
import pandas as pd

# Python 2.x, 3.x compatibility
try:
    xrange
except NameError:
    xrange = range

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
        print 'function only works for signals with all values >= 0'
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
