from matplotlib.pylab import norm
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
from SINDyPySource import SINDY, theta, dmethods

# Goal: generate Pure SINDy Model for Lotka-Volterra Equations


# determine what output you would like to see:
plot_error = True  #plots the error of the SINDy solution over time
plot_results = True #plots the SINDy solution and the true solution over time.
opt_lbd_coef = True
opt_lbd_sim = True

# create training data:
def true_forcing(t, x):
    dx = .7*x[0] - .5*x[1]*x[0]
    dy = -.3*x[1] + .2*x[1]*x[0]
    return [dx, dy]

true_coef = np.array([[.7, 0, 0, -.5, 0], [0, -.3, 0, .2, 0]])

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
lbd = .01

sindy = SINDY(dmethod, X, t, t_span, x0, theta_instance=theta_instance,
              lbd=lbd, regressor="lstsq")
model = sindy.model()
solx, soly = sindy.simulate().y


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

    plt.suptitle(f"Pure SINDY \n lbd={lbd: .2e} \n SINDy coefficients: {np.round(sindy.get_coef(), 2)} \n true coefficients: {true_coef} ")
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
if opt_lbd_coef:
    lbds = np.logspace(-5, 0, 50)
    min_error = np.inf
    opt_lbd = 0
    zetas = []
    for lbd in lbds:
        sindy_lbd = SINDY(dmethod, X, t, t_span, x0, theta_instance=theta_instance,
                          lbd=lbd, regressor="lstsq")
        model_lbd = sindy_lbd.model()
        coef_lbd = sindy_lbd.get_coef()
        Norm = np.linalg.norm(coef_lbd)
        if Norm == 0:
            norm = 1
        zeta = np.linalg.norm(abs(coef_lbd - true_coef))/Norm
        zetas.append(zeta)
        if zeta <= min_error:
            min_error = zeta
            opt_lbd = lbd
    zetas = np.array(zetas)

    plt.plot(lbds, zetas)
    plt.plot(opt_lbd, min_error, color="r")
    plt.xlabel("lambda values")
    plt.ylabel("coefficient error")
    plt.tight_layout()
    plt.show()
