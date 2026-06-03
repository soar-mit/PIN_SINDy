from matplotlib.pylab import norm
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
import itertools
from scipy.signal import savgol_filter
from matplotlib.gridspec import GridSpec

# Goal: generate Pure SINDy Model for Lotka-Volterra Equations

#source code (figure out how to import later):
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

        smooth_var = savgol_filter(var, window_length=11, polyorder=3)
        f = np.gradient(smooth_var, self.dt)
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

    def sparse_regression(self, fi=None):
        """returns the final coefficients for the system using pure SINDy (fi=None) or PIN-SINDy."""
        diff = self.diff
        theta_lib = self.theta_lib
        model = Lasso(alpha=self.lbd, fit_intercept=False)
        coef=[]
        if fi is None:
            for i in range(len(diff)):
                coef.append(model.fit(theta_lib.T, diff[i]).coef_) # (1, 5000) and (5000, 5) makes (1, 5) yes!
            coef = np.array(coef)
        else:
            for i in range(len(diff)):
                coef.append(model.fit(theta_lib.T, diff[i] - fi[i] @ theta_lib).coef_)
            coef = fi + coef
        return coef
class SINDY:

    def __init__(self, dmethod, theta_instance, var_list, lbd=.01):
        self.f = dmethod.forward_difference()
        self.theta_lib = theta_instance.library()
        self.var_list = np.array(var_list)
        self.order = theta_instance.get_order()
        self.theta = theta_instance
        self.lbd = lbd
        self.dmethod = dmethod

    def model(self, fi=None):
        """creates differential equation system using sparse regression for state variables."""
        f = self.f
        theta_lib = self.theta_lib
        s = STLSQ(f, theta_lib, self.lbd)
        coef = s.sparse_regression(fi)
        model = []
        for i in range(len(coef)):
            model_term = np.dot(theta_lib.T, coef[i])
            model.append(model_term)

        #update self.coef
        coef = np.array(coef)
        self.coef = coef
        return {'coef': coef, 'model': model }

    def get_coef(self, round_coef=True):
        if round_coef:
            for e in self.coef:
                for i in range(len(e)):
                    e[i] = round(e[i], 2)
        return self.coef


    def simulate(self, t_span, u0, t_data, method="LSODA"):

        assert len(self.coef) != 0, "create sindy model first before simulating."
        coef = self.coef
        pwrs = self.theta.get_powers()
        self.t_span = t_span
        self.u0 = u0
        self.t_data = t_data

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


    def best_lbd(self, X, true_coef):
        x, y = X
        lbd_guess = np.logspace(-6, 1, 50)
        min_err = np.inf
        lbd_opt = 0
        divergence = []
        inequal_shapes = 0
        for lbd in lbd_guess:
            sindy = SINDY(self.dmethod, self.theta, self.var_list, lbd=lbd)
            model = sindy.model()
            coef_sindy = sindy.get_coef()
            coef_err = np.mean((true_coef - coef_sindy)**2)
            if coef_err < min_err:
                min_err = coef_err
                lbd_opt = lbd
        final_sindy = SINDY(self.dmethod, self.theta, self.var_list, lbd=lbd_opt)
        final_model = final_sindy.model()
        final_sol = final_sindy.simulate(self.t_span, self.u0, self.t_data).y
        return {'model':final_model, 'min_err': min_err, 'lbd': lbd_opt, 'inequal_shapes': inequal_shapes, 'diverging_lbds': divergence, 'sol': final_sol}

# determine what output you would like to see:
plot_error = True  #plots the error of the SINDy solution over time
plot_results = True #plots the SINDy solution and the true solution over time.

# create training data:
def true_forcing(t, x):
    dx = .7*x[0] - .5*x[1]*x[0]
    dy = -.3*x[1] + .2*x[1]*x[0]
    return [dx, dy]

true_coef = np.array([[.7, 0, 0, -.5, 0], [0, -.3, 0, .2, 0]])
fi = None

x0 = [5, 2]

t = np.linspace(0, 10, 1000)
t_span = (0, 10)

true_sol = integrate.solve_ivp(true_forcing, t_span, x0, method="LSODA", t_eval=t)
x, y = true_sol.y
trueT = true_sol.t
X = np.array((x, y))

#create SINDy object to train data

dmethod = dmethods(X, t)
diff = dmethod.forward_difference()

theta_instance = theta(X, 2)

lbd_guess = np.logspace(-10, 2, 50)
x_err = np.inf
y_err = np.inf

print(theta_instance.library().shape)

sindy = SINDY(dmethod, theta_instance, X)
model = sindy.model(fi)
sol = sindy.simulate(t_span, x0, t)

best_lbd = sindy.best_lbd(X, true_coef)

solx, soly = best_lbd['sol']

lbd = best_lbd['lbd']


## plot results
if plot_results:
    fig, axs = plt.subplots(2,1, sharex=True)
    axs[0].plot(t, x, linestyle="dashed", label="observed x")
    axs[0].plot(t, solx, label="sindy x")
    axs[0].legend()
    axs[0].set_ylabel("x")

    axs[1].plot(t, y, linestyle="dashed", label="observed y")
    axs[1].plot(t, soly, label="sindy y")
    axs[1].legend()
    axs[1].set_ylabel("y")
    axs[1].set_xlabel("t")

    plt.suptitle(f"PIN-SINDY \n lbd={lbd: .2e} \n SINDy coefficients: {sindy.get_coef()} \n true coefficients: {true_coef} ")
    plt.tight_layout()
    plt.show()

# plot error
if plot_error:
    fig, axs = plt.subplots(1, 2, sharex=True)
    axs[0].plot(t, x-solx, label="x error")
    axs[1].plot(t, y-soly, label="y error")
    plt.suptitle(f"Error \n maximum x error: {max(abs(x-solx))} maximum y error: {max(abs(y - soly))}")
    axs[0].legend()
    axs[1].legend()
    plt.tight_layout()

    plt.show()
