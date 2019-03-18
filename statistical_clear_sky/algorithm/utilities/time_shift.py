"""
This module defines time shift related functions that are already used in
example codes.
They are kept as functions for backward compatibility.
"""

import numpy as np
import cvxpy as cvx
from statistical_clear_sky.solver_type import SolverType

def fix_time_shifts(power_signals_d, w=30, tol=5e-2,
                    solver_type=SolverType.ecos, verbose=False):
    # For each day, weight each time step by the power produced in that period, and take the average. This gives us
    # the "center" time point for each day, in terms of hours between 0 and 23 inclusive
    div1 = np.dot(np.linspace(0, 24, power_signals_d.shape[0]), power_signals_d)
    div2 = np.sum(power_signals_d, axis=0)
    s1 = np.empty_like(div1)
    s1[:] = np.nan
    msk = div2 != 0
    s1[msk] = np.divide(div1[msk], div2[msk])
    # Apply a sliding-window average filter
    s2 = np.convolve(s1, np.ones(w), mode='valid') / w
    # Apply 1D edge finding algorithm
    ixs = edge_find_1d(s2, tol=tol)
    if len(ixs) == 0:
        if verbose:
            print('No time shifts found')
        return power_signals_d
    ixs = list(map(lambda x: int(x + w / 2), ixs))
    num = len(ixs)
    if verbose:
        print('{} segments found'.format(num + 1))
        print('index locations: ', ixs)
    ixs = np.r_[[None], ixs, [None]]
    A = []
    for i in range(len(ixs) - 1):
        avg = np.average(np.ma.masked_invalid(s1[ixs[i]:ixs[i+1]]))
        A.append(np.round(avg * power_signals_d.shape[0] / 24))
    A = np.array(A)
    rolls = A[0] - A[1:]
    power_signals_d_out = np.copy(power_signals_d)
    for ind, roll in enumerate(rolls):
        power_signals_d_rolled = np.roll(power_signals_d, int(roll), axis=0)
        power_signals_d_out[:, ixs[ind + 1]:] = power_signals_d_rolled[:, ixs[ind + 1]:]
    return power_signals_d_out

def edge_find_1d(s1, tol=5e-2, ixs=None, ix0=0, w=30, mu=3,
                 solver_type=SolverType.ecos, debug=False):
    # Returns the indices of edges in a 1-D array. This algorithm recursively segments the input array until all edges
    # have been found.
    if ixs is None:
        ixs = []
    x = cvx.Variable(len(s1))
    mu = cvx.Constant(mu)
    obj = cvx.Minimize(cvx.norm(s1[np.isfinite(s1)] - x[np.isfinite(s1)]) + mu * cvx.norm1(x[:-1] - x[1:]))
    prob = cvx.Problem(obj)
    prob.solve(solver=solver_type.value)
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
        for j in range(ix - w, ix + w):
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
        ixs1 = edge_find_1d(s1[:ix_best], tol=tol, ixs=ixs, ix0=ix0)
        ixs2 = edge_find_1d(s1[ix_best:], tol=tol, ixs=ixs1, ix0=ix0+ix_best)
        ixs2.sort()
        return ixs2
