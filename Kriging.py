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

        self.original_x = np.copy(x)
        self.original_y = np.copy(y)

        self.corr_keyword = corr_keyword
        self.regr_keyword = regr_keyword
        self.x = np.copy(X)
        self.y = np.copy(Y)
        self.theta = np.copy(theta0)

        self.regr_func_eval, self.regr_func_grad = Kriging.eval_regression_basis(self)
        self.corr_func_scalar_eval, self.corr_func_vector_eval, self.corr_func_grad = Kriging.correlation_basis(self)

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
                :return: A function that evaluates the n+1 linear basis functions for point x, return as a n by n+1 matrix.
                '''
                assert isinstance(x, np.ndarray), 'The input of sites matrix x is not an array!'
                assert x.ndim == 2, 'The input x should be columnwise vector, 2D!'
                n, m = x.shape
                F = np.hstack((np.ones((m, 1)), x.T))
                return F

            def linear_basis_gradient(x):
                '''
                Determine the Jacobian matrix at x based on n+1 linear basis functions.
                :param x: The site x.
                :return: The Jacobian matrix, n+1 by n.
                '''
                assert isinstance(x, np.ndarray), 'The input of sites matrix x is not an array!'
                assert x.ndim == 2, 'The input x should be columnwise vector, 2D!'
                n, m = x.shape
                JF = np.vstack(( np.zeros((1, n)), np.identity(n) ))
                return JF

            return linear_basis, linear_basis_gradient

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
                if comb.any():  # For n = 1, there is no 1C2, so comb will be empty.
                    comb_matrix = x[comb - 1, :]  # combination numbers used as index should seduce 1 in python.
                    cross_term = comb_matrix[:, 0, :] * comb_matrix[:, 1, :]
                    F = np.hstack((F, cross_term.T))
                return F

            def quadratic_basis_gradient(x):
                '''
                Determine the Jacobian matrix at x based on (n+1)(n+2)/2 quadratic basis functions.
                :param x: The site x.
                :return:  The Jacobian matrix, (n+1)(n+2)/2 by n.
                '''
                assert isinstance(x, np.ndarray), 'The input of sites matrix x is not an array!'
                assert x.ndim == 2, 'The input x should be column-wise vector, 2D!'
                n, m = x.shape

                # linear part
                JF = np.vstack(( np.zeros((1, n)), np.identity(n) ))

                # quadratic part
                # np.diag must be 1 dimension row vector
                JF = np.vstack(( JF, np.diag(2 * x.T[0]) ))

                # cross-terms
                nlist = np.arange(1, n + 1)
                comb = np.array([list(c) for c in combinations(nlist, 2)])  # we use 2 because its quadratic.
                if comb.any():
                    for i in range(comb.shape[0]):
                        temp = np.zeros((1, n))
                        temp[comb[i]-1] = x[np.flip(comb[i]-1, axis=0)]
                        JF = np.vstack(( JF, temp))
                return JF

            return quadratic_basis, quadratic_basis_gradient

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
                :param theta: Weight parameters for different dimensions, should be a n by 1.
                :param x: One site, has the form as n by , vector.
                :param y: Another site, has the form as n by , vector.
                :return: An exponential correlation function
                '''
                assert theta.ndim == 2, 'The shape of theta should be (n,1).'

                theta = theta.reshape(-1, 1).T  # transform theta to be a 2D, n by 1 vector.
                exp_theta = np.power(10, theta)
                diff = np.abs(x-y)
                return np.exp(-exp_theta.dot(diff))

            def correxp_vector(theta, x, Y):
                '''
                Evaluate the exponential correlation between site x and other sites Y.
                Important: notice that theta should be in exp form
                :param theta: Weight parameters for different dimensions, should be a n by 1.
                :param x: One site, has the form as n by , vector.
                :param y: Another site, has the form as n by m vector.
                :return: An exponential correlation function
                '''
                theta = theta.reshape(-1, 1).T  # transform theta to be a 2D, n by 1 vector.
                exp_theta = np.power(10, theta)
                x = x.reshape(-1, 1)  # transform x into a 2D, n by 1 vector
                diff = np.abs(Y - x)
                return np.exp(-exp_theta.dot(diff))

            def correxp_gradient(theta, x, Y):
                '''
                Evaluate the gradient of exponential correlation between x and other sites Y.
                :param theta: Weight parameters for different dimensions, should be a n by 1.
                :param x: The site x.
                :param Y: The evaluated sites.
                :return:  The Jacobian of correlation function.
                '''
                theta = theta.reshape(-1, 1).T  # transform theta to be a 2D, n by 1 vector.

                exp_theta = np.power(10, theta)
                x = x.reshape(-1, 1)

                return np.exp( -exp_theta.dot(np.sign(x-Y)) )

            return correxp_scalar, correxp_vector, correxp_gradient

        elif self.corr_keyword == 'corrgauss':

            def corrgauss_scalar(theta, x, y):
                '''
                Evaluate the exponential correlation between two sites.
                Important: notice that theta should be in exp form
                :param theta: Weight parameters for different dimensions, should be a n by 1.
                :param x: One site, has the form as n by , vector.
                :param y: Another site, has the form as n by , vector.
                :return: An gaussian correlation function
                '''
                assert theta.ndim == 2, 'The shape of theta should be (n,1).'
                theta = theta.reshape(-1, 1).T  # transform theta to be a 2D, n by 1 vector.
                exp_theta = np.power(10, theta)
                return np.exp(-exp_theta.dot((x-y)**2))

            def corrgauss_vector(theta, x, Y):
                '''
                Evaluate the exponential correlation between site x and other sites Y.
                :param theta: Weight parameters for different dimensions, should be a n by 1.
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


            def corrgrauss_gradient(theta, x, Y):
                '''
                Evaluate the gradient of exponential correlation between x and other sites Y.
                :param theta: Weight parameters for different dimensions, should be a n by 1.
                :param x: The site x.
                :param Y: The evaluated sites.
                :return:  The Jacobian of correlation function.
                '''
                theta = theta.reshape(-1, 1).T  # transform theta to be a 2D, n by 1 vector.

                exp_theta = np.power(10, theta)
                x = x.reshape(-1, 1)
                diff = 2 * np.abs(x - Y) * np.sign(x - Y)
                return np.exp( -exp_theta.dot(diff) )

            return corrgauss_scalar, corrgauss_vector, corrgrauss_gradient

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
        mu = (10 + m) * 1e-5
        Phi = np.copy(Phi + mu * np.identity(m))
        C = np.linalg.cholesky(Phi)
        C_inv = np.linalg.inv(C)
        tilde_F = np.dot(C_inv, F)
        tilde_Y = np.dot(C_inv, self.y.reshape(-1, 1))
        Q, R = np.linalg.qr(tilde_F)
        beta = np.dot(np.dot(np.linalg.inv(R), Q.T), tilde_Y)
        residual = tilde_Y - np.dot(tilde_F, beta)
        sigma2 = 1/m * np.dot(residual.T, residual)
        psi = np.linalg.det(Phi) ** (1/m) * sigma2
        gamma = np.dot(C_inv.T, tilde_Y - np.dot(tilde_F, beta))

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
        iter_max = 1000
        # The upper and lower bounds are given in Engineering Design via Surrogate Modelling: A Practical Guide.
        lb = -2 * np.ones((self.x.shape[0], 1))
        ub = 2 * np.ones((self.x.shape[0], 1))
        theta, iter_num, delta = GPS.move(self.theta, delta, func_eval, iter_max, delta_refine, lb, ub)
        self.theta = np.copy(theta)
        # Finally we have theta, we need to update all info one more time.
        psi = Kriging.psi_eval(self, F, self.theta)

    def  kriging_eval(self, x):
        x = x.reshape(-1, 1)
        '''
        Evaluate the kriging interpolation at untried site x.
        :param x: The site.
        :return:  The Kriging interpolation value at x.
        '''
        normalized_x = (x - np.mean(self.original_x, axis=1).reshape(-1, 1)) / np.std(self.original_x, axis=1).reshape(-1, 1)

        normalized_y = np.dot(self.regr_func_eval(normalized_x), self.beta) + \
               np.dot(self.corr_func_vector_eval(self.theta, normalized_x, self.x), self.gamma)
        return normalized_y * np.std(self.original_y) + np.mean(self.original_y)

    def kriging_gradient(self, x):

        return





