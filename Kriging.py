import  numpy               as np
import  matplotlib.pyplot   as plt
from    functools           import partial
from    itertools           import combinations
import  Modified_Hookes_GPS as GPS


class Kriging:
    def __init__(self, corr_keyword, regr_keyword, y, x, theta0):
        '''
        :param y: The function values y_i at the sites x_i
        :param x: The set of evaluated sites.
        :param regr_keyword: The regression model, consists of a basis of functions.
        :param corr_keyword: The correlation functions.
        :param theta0: The initial guess for parameters of theta.
        '''
        assert isinstance(regr_keyword, np.str), \
            'The input name "keyword" for regression should be a string, linear or quadratic.'
        assert isinstance(corr_keyword, np.str), \
            'The input name "keyword" for correlation should be a string, correxp or corrgauss.'
        assert isinstance(x, np.ndarray), 'The input of sites matrix x is not an array!'
        assert x.ndim == 2, 'The input of sites matrix x should be 2 dimension array!'
        assert isinstance(y, np.ndarray), 'The input of function values y is not an array!'
        assert y.ndim == 1, 'The input y should be 1D (n by ,) vector!'

        # Normalize data, x & y --> X & Y
        Y = (y - np.mean(y)) / (np.std(y))
        X = (x - np.mean(x, axis=1).reshape(-1, 1)) / np.std(x, axis=1).reshape(-1, 1)

        self.corr_keyword = corr_keyword
        self.regr_keyword = regr_keyword
        self.x = np.copy(X)
        self.y = np.copy(Y)
        self.theta = np.copy(theta0)

        self.regr_func_eval = Kriging.eval_regression_basis(self)
        self.corr_func_scalar_eval, self.corr_func_vector_eval = Kriging.correlation_basis(self)

        self.C_inv = []
        self.R = []
        self.beta = []
        self.gamma = []
        self.tilde_F = []

    def eval_regression_basis(self):
        '''
        Evaluate the regression basis, using the keyword self.regr_keyword, which could be linear or quadratic.
        :param keyword: should be linear or quadratic
        :return: a basis function
        '''

        if self.regr_keyword == 'linear':
            def linear_basis(x):
                '''
                Return a basis regression, consists of n+1 functions. 1 constant function plus each coordinates.
                :param x: The sites x.
                :return: n+1 basis functions.
                '''
                assert isinstance(x, np.ndarray), 'The input of sites matrix x is not an array!'
                assert x.ndim == 2, 'The input x should be columnwise vector, 2D!'
                n, m = x.shape
                F = np.hstack((np.ones((m, 1)), x.T))
                return F
            return linear_basis

        elif self.regr_keyword == 'quadratic':
            def quadratic_basis(x):
                '''
                Return a basis regression, consists of (n+1)(n+2)/2. 1 constant function, n linear basis functions,
                n square quadratic, (n+1)n/2 cross-term basis functions.
                :param x: The sites x
                :return: (n+1)(n+2)/2 basis functions.
                '''
                assert isinstance(x, np.ndarray), 'The input of sites matrix x is not an array!'
                assert x.ndim == 2, 'The input x should be column-wise vector, 2D!'
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

    def correlation_basis(self):
        '''
        Return the correlation function of two sites.
        :param keyword: Determine the type of correlation.
        :return:  A function that evaluates the correlation between two sites.
        '''
        if self.corr_keyword == 'correxp':

            def correxp_scalar(theta, x, y):
                '''
                Evaluate the exponential correlation between two sites.
                Important: notice that theta should be in exp form
                :param theta: Weight parameters for different dimensions, should be a 1 by n vector for now.
                :param x: One site, has the form as n by , vector.
                :param y: Another site, has the form as n by , vector.
                :return: An exponential correlation function
                '''
                exp_theta = np.power(10, theta)
                diff = np.abs(x-y)
                return np.exp(-exp_theta.dot(diff))

            def correxp_vector(theta, x, Y):
                '''
                Evaluate the exponential correlation between site x and other sites Y.
                Important: notice that theta should be in exp form
                :param theta: Weight parameters for different dimensions, should be a 1 by n vector for now.
                :param x: One site, has the form as n by , vector.
                :param y: Another site, has the form as n by m vector.
                :return: An exponential correlation function
                '''
                theta = theta.reshape(-1, 1).T  # transform theta to be a 2D, n by 1 vector.
                exp_theta = np.power(10, theta)
                x = x.reshape(-1, 1)  # transform x into a 2D, n by 1 vector
                diff = np.abs(Y - x)
                return np.exp(-exp_theta.dot(diff))
            return correxp_scalar, correxp_vector

        elif self.corr_keyword == 'corrgauss':

            def corrgauss_scalar(theta, x, y):
                '''
                Evaluate the exponential correlation between two sites.
                Important: notice that theta should be in exp form
                :param theta: Weight parameters for different dimensions, should be a 1 by n vector for now.
                :param x: One site, has the form as n by , vector.
                :param y: Another site, has the form as n by , vector.
                :return: An gaussian correlation function
                '''
                exp_theta = np.power(10, theta)
                return np.exp(-exp_theta.dot((x-y)**2))

            def corrgauss_vector(theta, x, Y):
                '''
                Evaluate the exponential correlation between site x and other sites Y.
                :param theta: Weight parameters for different dimensions, should be a 1 by n vector for now.
                Important: notice that theta should be in exp form
                :param x: One site, has the form as n by , vector.
                :param y: Another site, has the form as n by m vector.
                :return: An exponential correlation function
                '''
                theta = theta.reshape(-1, 1).T  # transform theta to be a 2D, n by 1 vector.
                exp_theta = np.power(10, theta)
                x = x.reshape(-1, 1)  # transform x into a 2D, n by 1 vector
                diff = np.abs(Y - x)
                return np.exp(-exp_theta.dot(diff**2))
            return corrgauss_scalar, corrgauss_vector

        else:
            print('Wrong type of keyword!')

    def corr_eval(self, theta):
        '''
        Evaluate the correlation function.
        :param theta: The regression parameters.
        :return:
        '''
        x = self.x
        n, m = x.shape
        Phi = np.zeros((m, m))
        for i in range(m):
            Phi[i, :] = self.corr_func_vector_eval(theta, x[:, i], x)
        return Phi

    def psi_eval(self, F, theta):
        n, m = self.x.shape
        Phi = Kriging.corr_eval(self, theta)
        # Regularization, Phi = Phi + mu * I, mu = (10+m)*epsilon
        mu = (10 + m) * 1e-3
        Phi = np.copy(Phi + mu * np.identity(m))
        C = np.linalg.cholesky(Phi)
        # TODO inv optimize, forward substitution.
        C_inv = np.linalg.inv(C)
        tilde_F = np.dot(C_inv, F)
        tilde_Y = np.dot(C_inv, self.y.reshape(-1, 1))
        Q, R = np.linalg.qr(tilde_F)
        # TODO inv optimize, forward substitution.
        beta = np.dot(np.dot(np.linalg.inv(R), Q.T), tilde_Y)
        residual = tilde_Y - np.dot(tilde_F, beta)
        sigma2 = 1/m * np.dot(residual.T, residual)
        psi = np.linalg.det(Phi) ** (1/m) * sigma2
        gamma = C_inv.dot(tilde_Y - np.dot(tilde_F, beta))

        self.C_inv = np.copy(C_inv)
        self.R = np.copy(R)
        self.beta = np.copy(beta)
        self.gamma = np.copy(gamma)
        self.tilde_F = np.copy(tilde_F)
        return psi

    def main_dace(self):
        '''
        :return:
        '''

        F = self.regr_func_eval(self.x)
        func_eval = partial(Kriging.psi_eval, self, F)
        delta, delta_refine = 0.2, 16
        iter_max = 100
        # The upper and lower bounds are given in Engineering Design via Surrogate Modelling: A Practical Guide.
        lb = -2 * np.ones((self.x.shape[0], 1))
        ub = 2 * np.ones((self.x.shape[0], 1))
        theta, iter_num, delta = GPS.move(self.theta, delta, func_eval, iter_max, delta_refine, lb, ub)
        self.theta = np.copy(theta)
        # Finally we have theta, we need to update all info one more time.
        psi = Kriging.psi_eval(self, F, self.theta)

    def kriging_eval(self, x):
        x = x.reshape(-1, 1)
        '''
        Evaluate the kriging interpolation at untried site x.
        :param x: The site.
        :return:  The Kriging interpolation value at x.
        '''
        return np.dot(self.regr_func_eval(x), self.beta) + \
               np.dot(self.corr_func_vector_eval(self.theta, x, self.x), self.gamma)

    def kriging_gradient(self, x):

        return





