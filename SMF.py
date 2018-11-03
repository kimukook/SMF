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


def




n = 2
m = 10
Nm = 8
x = np.random.rand((n, m))
y = np.zeros(m)
fname, xmin, y0, fun, lb, ub = test_fun_schwefel(n)

for i in range(m):
    y[i] = fun(x[i, :].reshape(-1, 1))
