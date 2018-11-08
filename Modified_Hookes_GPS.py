import numpy as np
''' 
This is a scripy that implemented Modified Hookes and Jeeves Generalized Pattern Search Algorithm.
For more information, please refer "Introduction to Optimziation and Data Fitting" by K. Madsen, 2008. Section 4.3.
'''


def explore(x, delta, func, lb=[], ub=[]):
    '''
    Implement the Algorithm 4.2: Explore around the current best point by evaluating points with the unit length distance
    in each coordiante.
    :param x:       The current best point. Form: n by 1 vector.
    :param delta:   The step length of current move.
    :param func:    The objective function that aims at minimizing.
    :param lb:      The lower bound of parameters.
    :param ub:      The upper bound of parameters.
    :return: A new point for exploration.
    '''
    n = x.shape[0]
    if not lb:  # lb == []
        lb = np.zeros((n, 1))
    if not ub:
        ub = np.ones((n, 1))
    x_bar = np.copy(x)
    for i in range(n):
        new_x = step_length_constraints_check(delta, x_bar, i, lb, ub)
        psi_value = np.zeros(3)
        for j in range(new_x.shape[1]):
            psi_value[j] = func(new_x[:, j].reshape(-1, 1))
        index = np.argmin(psi_value)
        x_bar += (index - 1) * np.abs((new_x[:, index].reshape(-1, 1) - x_bar))
    return x_bar


def step_length_constraints_check(delta, x_hat, i, lb, ub):
    '''
    Boudn the coordinate seach inside the box domain.
    :param delta: Step length.
    :param x_hat: current best point
    :param i: Coordinate
    :return: Bounded step points in coordinate i.
    '''
    n = x_hat.shape[0]
    step = delta * np.identity(n)[:, i].reshape(-1, 1)
    new_x = np.hstack((x_hat - step, x_hat, x_hat + step))
    for i in range(new_x.shape[0]):
        for j in range(new_x.shape[1]):
            if new_x[i, j] > ub[i, 0]:
                new_x[i, j] = ub[i, 0]
            elif new_x[i, j] < lb[i, 0]:
                new_x[i, j] = lb[i, 0]
    return new_x


def bound_base_point(z, lb, ub):
    '''
    Bound the base point in the box domain.
    :param z:
    :return: Bounded base point.
    '''
    n = z.shape[0]
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if z[i, j] > ub[i, 0]:
                z[i, j] = ub[i, 0]
            elif z[i, j] < lb[i, 0]:
                z[i, j] = lb[i, 0]
    return z


def check_stop(delta, delta_threshold, iter_max, iter_num):
    '''
    Check whether or not the stop criteria is satisfied.
    :param delta: step length
    :param delta_threshold: target step length
    :param iter_max: Maximum number of iterations
    :param iter_num: Current iteration number
    :return: Stop or not.
    '''
    if delta <= delta_threshold or iter_num >= iter_max:
        stop = 1
    else:
        stop = 0
    return stop


def move(x, delta, func, iter_max, delta_refine, lb, ub):
    '''
    The main script.
    :param x: Current best point or initial point.
    :param delta: Initial step length.
    :param func: Objective function.
    :param iter_max: Maximum number of function evaluations
    :param delta_refine: Times of refinement on delta.
    :return: Minimizer.
    '''
    assert x.ndim == 2, 'The input x should be n by 1 vector!'
    assert isinstance(x, np.ndarray), 'The inputx is not a np.ndarray!'
    delta_threshold = (1/2) ** delta_refine * delta
    x_hat = explore(x, delta, func, lb, ub)
    stop = 0
    iter_num = 0
    while not stop:
        iter_num += 1
        if func(x_hat) < func(x):
            z = np.copy(x_hat + (x_hat - x))
            z = bound_base_point(z, lb, ub)
            x = np.copy(x_hat)
        else:
            z = np.copy(x)
            delta /= 2
        x_hat = explore(z, delta, func, lb, ub)
        stop = check_stop(delta, delta_threshold, iter_max, iter_num)
        # print('xhat = ', x_hat.T)
        # print('delta = ', delta)
    return x_hat, iter_num, delta


# # Test case:
# def test_fun_schwefel(n):
#     lb = np.zeros(n)
#     ub = np.ones(n)
#     fun = lambda x: - sum(np.multiply(500 * x, np.sin(np.sqrt(abs(500 * x))))) / 250
#     y0 = -1.6759316 * n  # targert value for objective function
#     xmin = 0.8419 * np.ones((n, 1))
#     fname = 'Schewfel'
#     return fname, xmin, y0, fun, lb, ub
#
# n = 2
# m = 10
# Nm = 8
# x = np.random.rand(n, m)
# x = np.round(x * Nm) / Nm
# y = np.zeros(m)
# fname, xmin, y0, fun, lb, ub = test_fun_schwefel(n)
#
# for i in range(m):
#     y[i] = fun(x[:, i].reshape(-1, 1))
#
# best = x[:, np.argmin(y)].reshape(-1, 1)
# delta = 0.2
# iter_max = 100
# delta_refine = 8
#
#
# x_minima, iter_num, delta = move(best, delta, fun, iter_max, delta_refine)
# This method will converge to minimum if we have a good point near by.

