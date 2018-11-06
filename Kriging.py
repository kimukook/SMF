import  numpy               as np
import  matplotlib.pyplot   as plt
from    functools           import partial
from    itertools           import combinations
import  Modified_Hookes_GPS as GPS

def eval_regression_basis(keyword):
    '''

    :param keyword: should be linear or quadratic
    :return: a basis function
    '''
    assert isinstance(keyword, np.str), 'The input name "keyword" should be a string, linear or quadratic'
    if keyword =='linear':
        def linear_basis(x):
            '''
            Return a basis regression, consists of n+1 functions. 1 constant function plus each coordinates.
            :param x: The sites x.
            :return: n+1 basis functions.
            '''
            assert isinstance(x, np.ndarray), 'The input of sites matrix x is not an array!'
            n, m = x.shape
            F = np.hstack((np.ones((m, 1)), x.T))
            return F
        return linear_basis

    elif keyword == 'quadratic':
        def quadratic_basis(x):
            '''
            Return a basis regression, consists of (n+1)(n+2)/2. 1 constant function, n linear basis functions,
            n square quadratic, (n+1)n/2 cross-term basis functions.
            :param x: The sites x
            :return: (n+1)(n+2)/2 basis functions.
            '''
            assert isinstance(x, np.ndarray), 'The input of sites matrix x is not an array!'
            n, m = x.shape

            # linear part
            F = np.hstack((np.ones((m, 1)), x.T))

            # square terms part
            F = np.hstack(( F, x.T**2 ))

            # Cross-terms part
            nlist = np.arange(1, n + 1)
            # the order of functions in F can be shuffled.
            comb = np.array([list(c) for c in combinations(nlist, 2)])  # we use 2 because its quadratic.
            comb_matrix = x[comb]
            cross_term = comb_matrix[:, 0, :] * comb_matrix[:, 1, :]
            F = np.hstack((F, cross_term.T))
            return F
        return quadratic_basis

    else:
        print('Keyword should be "linear" or "quadratic" temporarily ')


def correlation_basis(keyword):
    '''
    Return the correlation function of two sites.
    :param keyword: Determine the type of correlation.
    :return:  A function that evaluates the correlation between two sites.
    '''
    if keyword == 'correxp':
        def correxp_scalar(theta, x, y):
            '''
            Evaluate the exponential correlation between two sites.
            :param theta: Weight parameters for different dimensions, should be a 1 by n vector for now.
            :param x: One site, has the form as n by , vector.
            :param y: Another site, has the form as n by , vector.
            :return: An exponential correlation function
            '''
            diff = np.abs(x-y)
            return np.exp(-theta.dot(diff))


        def correxp_vector(theta, x, Y):
            '''
            Evaluate the exponential correlation between site x and other sites Y.
            :param theta: Weight parameters for different dimensions, should be a 1 by n vector for now.
            :param x: One site, has the form as n by , vector.
            :param y: Another site, has the form as n by m vector.
            :return: An exponential correlation function
            '''
            theta = theta.reshape(-1, 1).T  # transform theta to be a 2D, n by 1 vector.
            x = x.reshape(-1, 1)  # transform x into a 2D, n by 1 vector
            diff = np.abs(Y - x)
            return np.exp(-theta.dot(diff))
        return correxp_scalar, correxp_vector

    elif keyword == 'Gaussian':
        def corrgauss_scalar(theta, x, y):
            '''
            Evaluate the exponential correlation between two sites.
            :param theta: Weight parameters for different dimensions, should be a 1 by n vector for now.
            :param x: One site, has the form as n by , vector.
            :param y: Another site, has the form as n by , vector.
            :return: An gaussian correlation function
            '''
            return np.exp(-theta.dot((x-y)**2))


        def corrgauss_vector(theta, x, Y):
            '''
            Evaluate the exponential correlation between site x and other sites Y.
            :param theta: Weight parameters for different dimensions, should be a 1 by n vector for now.
            :param x: One site, has the form as n by , vector.
            :param y: Another site, has the form as n by m vector.
            :return: An exponential correlation function
            '''
            theta = theta.reshape(-1, 1).T  # transform theta to be a 2D, n by 1 vector.
            x = x.reshape(-1, 1)  # transform x into a 2D, n by 1 vector
            diff = np.abs(Y - x)
            return np.exp(-theta.dot(diff**2))
        return corrgauss_scalar, corrgauss_vector

    else:
        print('Wrong type of keyword!')


def corr_eval(corr_vector, theta, x):
    assert isinstance(x, np.ndarray), 'The input of sites matrix x is not an array!'
    assert x.ndim == 2, 'The input of sites matrix x should be 2 dimension array!'
    n, m = x.shape
    Phi = np.zeros((m, m))
    for i in range(m):
        Phi[i, :] = corr_vector(theta, x[:, i], x)
    return Phi


def psi_eval(corr_vector, y, x, F, theta):
    n, m = x.shape
    Phi = corr_eval(corr_vector, theta, x)  # TODO need add a mu*I on the digonal.
    C = np.linalg.cholesky(Phi)
    C_inv = np.linalg.inv(C)  # TODO This could be optimized by forward substitution.
    tilde_F = np.dot(C_inv, F)
    tilde_Y = np.dot(C_inv, y)
    Q, R = np.linalg.qr(tilde_F)
    beta = np.dot(np.dot(np.linalg.inv(R), Q.T), tilde_Y)
    residual = tilde_Y - np.dot(tilde_F, beta)
    sigma2 = 1/m * np.dot(residual.T, residual)
    psi = np.linalg.det(Phi) ** (1/m) * sigma2
    return psi


def main_dace(y, x, regr_keyword, corr_keyword, theta0):
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

    regr = eval_regression_basis(regr_keyword)
    F = regr(x)
    corr_scalar, corr_vector = correlation_basis(corr_keyword)
    func_eval = partial(psi_eval, corr_vector, y, x, F)
    delta, delta_refine = 0.2, 8
    iter_max = 100
    theta, iter_num, delta = GPS.move(theta0, delta, func_eval, iter_max, delta_refine)
    # TODO determine those outputs.
    # TODO theta must be n by 1 vector, restricted by GPS.move algorithm.
    return theta, C, R, tilde_F, beta, gamma


# Test case
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
