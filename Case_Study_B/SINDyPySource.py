from scipy import integrate
import numpy as np
from sklearn.linear_model import Lasso
import itertools
from scipy.signal import savgol_filter
import os

print("Current place of this file:" + os.getcwd())

class theta:
    def __init__(self, var_list, order):
        self.var_list = np.array(var_list)
        self.order = order
    def get_order(self):
        return self.order

    def get_powers(self): # I is length of var_list, K is order
        """I (int): length of var_list
        K (int): order
        returns: matrix in order of increasing power will all combinations of powers. """
        I = len(self.var_list)
        K = self.order
        pwrs = []
        for k in range(K+1):
            for powers in itertools.combinations_with_replacement(range(I), k):
                pwr = [0]*I
                for i in powers:
                    pwr[i] += 1
                pwrs.append(pwr)
        pwrs = np.array(pwrs)
        return pwrs.astype(np.float64)

    def library(self):
        l = []
        var_list = self.var_list

        var_list = np.array(var_list).astype(np.float64)
        pwrs = self.get_powers()
        for p in pwrs:
            term = np.ones_like(var_list[0])
            for i in range(len(var_list)):
                term *= var_list[i]**p[i]
            l.append(term)
        library = np.array(l)
        library = library[1:, :]
        return library
class dmethods:

    def __init__(self, x, t):
        """x: list of state variables (list of list)
        t: list of time stamps. """
        self.dt = t[1] - t[0]
        self.x = x
        self.t = t


    def differential(self, var):
        "returns approximate differentation using central finite difference method."
        "t: list of floats "
        "x: list of floats: one state variable at different times."

        f = np.gradient(var, self.dt)
        return f

    def forward_difference(self):
        "returns differentation of each state variable (ex: [grad x1, grad x2, grad x3, ...])"
        x = self.x
        f = []
        for var in x:
            f.append(self.differential(var))
        f = np.array(f)
        f = np.clip(f, -1e6, 1e6)
        return f
class STLSQ:

    def __init__(self, diff, theta_lib, lbd=.01):
        """initializes parameters:
        diff is differential of all state variables.
        lbd: float representing bias penalty.
        feature library: is the list of candidate functions.
        """
        if lbd < 0:
            raise ValueError("threshold must be a positive number.")
        self.diff = diff
        self.lbd = lbd
        self.theta_lib = theta_lib

    def sparse_regression(self, fi=None, norm_optimization=False, threshold=None):
        """returns the final coefficients for the system using pure SINDy (fi=None) or PIN-SINDy."""
        diff = self.diff
        theta_lib = self.theta_lib
        model = Lasso(alpha=self.lbd, fit_intercept=False)
        if norm_optimization:
            norm = np.linalg.norm(theta_lib, ord=2, axis=1)
            norm[norm==0] = 1
            theta_lib_scaled = theta_lib/norm[:,None]
            coef_scaled=[]
            if fi is None:
                for i in range(len(diff)):
                    coef_scaled.append(model.fit(theta_lib_scaled.T, diff[i]).coef_) # (1, 5000) and (5000, 5) makes (1, 5) yes!
                coef_scaled = np.array(coef_scaled)
                coef = coef_scaled / norm
            else:
                for i in range(len(diff)):
                    coef_scaled.append(model.fit(theta_lib_scaled.T, diff[i] - fi[i] @ theta_lib_scaled).coef_)
                coef_scaled = np.array(coef_scaled)
                coef = fi + coef_scaled / norm
        else:
           coef=[]
           if fi is None:
                for i in range(len(diff)):
                    coef.append(model.fit(theta_lib.T, diff[i]).coef_) # (1, 5000) and (5000, 5) makes (1, 5) yes!
                coef = np.array(coef)
           else:
                for i in range(len(diff)):
                    coef.append(model.fit(theta_lib.T, diff[i] - fi[i] @ theta_lib).coef_)
                coef= np.array(coef)
                coef = fi + coef
        if threshold:
            coef[coef <= np.abs(threshold)] = 0
        return coef

class SINDY:

    def __init__(self, dmethod, theta_instance, var_list, lbd=.01, norm_optimization=False, threshold=None):
        self.f = dmethod.forward_difference()
        self.theta_lib = theta_instance.library()
        self.var_list = np.array(var_list)
        self.order = theta_instance.get_order()
        self.theta = theta_instance
        self.lbd = lbd
        self.dmethod = dmethod
        self.norm_optimization = norm_optimization
        self.threshold = threshold


    def get_coef(self, round_coef=True):
        if round_coef:
            for e in self.coef:
                for i in range(len(e)):
                    e[i] = round(e[i], 2)
        return self.coef

    def model(self, fi=None):
        """creates differential equation system using sparse regression for state variables."""
        f = self.f
        theta_lib = self.theta_lib
        s = STLSQ(f, theta_lib, self.lbd)
        coef = s.sparse_regression(fi, norm_optimization = self.norm_optimization, threshold=self.threshold)
        model = []
        for i in range(len(coef)):
            model_term = np.dot(theta_lib.T, coef[i])
            model.append(model_term)

        #update self.coef
        coef = np.array(coef)
        self.coef = coef
        return {'coef': coef, 'model': model }

    def simulate(self, t_span, u0, t_data, method="LSODA", fi=None):

        pwrs = self.theta.get_powers()
        self.t_span = t_span
        self.u0 = u0
        self.t_data = t_data

        f = self.f
        theta_lib = self.theta_lib
        s = STLSQ(f, theta_lib, self.lbd)
        coef = s.sparse_regression(fi)

        model = []
        for i in range(len(coef)):
            model_term = np.dot(theta_lib.T, coef[i])
            model.append(model_term)
        model = np.array(model)

        #update self.coef
        coef = np.array(coef)
        self.coef = coef

        def f(t, y):
            """function version of library. Takes positional argument y(list of state variable floats, len = # of state variables, time t(float), and np array
            of sparse regression coefs(float np array)
            t is a list of all the time steps in original data."""

            theta_vals = []
            for p in pwrs:
                assert len(p) == len(y), "oops! recalculate theta library."
                term = 1.0
                for i in range(len(p)):
                    term *= y[i]**p[i]
                theta_vals.append(term)

            theta_vals = np.array(theta_vals)
            theta_vals = theta_vals[1:]

            forcing = np.array([np.dot(theta_vals, coef_vector) for coef_vector in coef])
            return forcing

        sol = integrate.solve_ivp(f, t_span, u0, method=method, t_eval = t_data)
        return sol


    def best_lbd(self, X, true_coef, fi=None):
        lbd_guess = np.logspace(-6, 1, 50)
        min_err = np.inf
        lbd_opt = 0
        for lbd in lbd_guess:
            sindy_test = SINDY(self.dmethod, self.theta, self.var_list, lbd=lbd)
            if fi is None:
                model = sindy_test.model()
            else:
                model = sindy_test.model(fi=fi)
            coef_sindy = sindy_test.get_coef()
            coef_err = np.mean((true_coef - coef_sindy)**2)
            if coef_err < min_err:
                min_err = coef_err
                lbd_opt = lbd
        final_sindy = SINDY(self.dmethod, self.theta, self.var_list, lbd=lbd_opt)
        if fi is None:
            final_model = final_sindy.model()
        else:
            final_model = final_sindy.model(fi=fi)
        final_sol = final_sindy.simulate(self.t_span, self.u0, self.t_data).y
        return {'model':final_model, 'min_err': min_err, 'lbd': lbd_opt, 'sol': final_sol}
