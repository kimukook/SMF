import numpy as np
import matplotlib.pyplot as plt


def test_fun_schwefel(n):
    lb = np.zeros(n)
    ub = np.ones(n)
    fun = lambda x: - sum(np.multiply(500 * x, np.sin(np.sqrt(abs(500 * x))))) / 250
    y0 = -1.6759316 * n  # targert value for objective function
    xmin = 0.8419 * np.ones((n, 1))
    fname = 'Schewfel'
    return fname, xmin, y0, fun, lb, ub

n = 2
m = 10
Nm = 8
x = np.random.rand(n, m)
x = np.round(x * Nm) / Nm
y = np.zeros(m)
fname, xmin, y0, fun, lb, ub = test_fun_schwefel(n)

for i in range(m):
    y[i] = fun(x[:, i].reshape(-1, 1))


def normalize_input(x):
    '''

    :param x: 2-dimensional array
    :return: normalized array x by (x - mean x ) / std(x)
    '''
    return


def main_dace(y, x, regr, corr, theta0):
    '''

    :param y: The function values y_i at the sites x_i
    :param x: The set of evaluated sites.
    :param regr: The regression model, consists of a basis of functions.
    :param corr: The correlation functions.
    :param theta0: The initial guess for parameters of theta.
    :return:
    '''
    assert isinstance(y, np.ndarray), 'The input of function evaluations matrix y is not an array!'
    assert isinstance(x, np.ndarray), 'The input of sites matrix x is not an array!'
    assert y.ndim == 1, 'The input of function evaluations matrix y should be 1 dimension array!'
    assert x.ndim == 2, 'The input of sites matrix x should be 2 dimension array!'

    # Normalize data, x & y --> X & Y
    Y = (y - np.mean(y)) / (np.std(y))
    X = (x - np.mean(x, axis=1).reshape(-1, 1)) / np.std(x, axis=1).reshape(-1, 1)
    F = evaluate_regression(regr, X)
    psi = minimize_psi(theta0, corr, Y, X, F)
    return psi


def evaluate_regression(regr, X):
    n, m = X.shape
    F = np.zeros((m, m))
    for i in range(m):
        for j in range(m):


    return F


def minimize_psi(theta0, corr, Y, X, F):

    return psi


