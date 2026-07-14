from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from Brunton_result_rep.SINDyPySource import SINDY, theta, dmethods

# Goal: generate PIN-SINDy Model for Lotka-Volterra Equations


# determine what output you would like to see:
plot_error = False  #plots the error of the SINDy solution over time
plot_opt_lbd = False
plot_results = True #plots the SINDy solution and the true solution over time.
plot_true_vs_prior = False #plots the true solution and the prior solution over time.

# create training data:
def true_forcing(t, x):
    dx = .7*x[0] - .5*x[1]*x[0]
    dy = -.3*x[1] + .2*x[1]*x[0]
    return [dx, dy]

def prior_forcing(t, x):
    dx = x[0] - x[1]*x[0]
    dy = - x[1] + x[1]*x[0]
    return [dx, dy]


true_coef = np.array([[.7, 0, 0, -.5, 0], [0, -.3, 0, .2, 0]])
fi = np.array([[1, 0, 0, -1, 0], [0, -1, 0, 1, 0]])

x0 = [5, 2]

t = np.linspace(0, 10, 1000)
t_span = (0, 10)

true_sol = integrate.solve_ivp(true_forcing, t_span, x0, method="LSODA", t_eval=t)
x, y = true_sol.y
trueT = true_sol.t
X = np.array((x, y))


#plot true vs prior equations
if plot_true_vs_prior:
    prior_sol = integrate.solve_ivp(prior_forcing, t_span, x0, t_eval=t)
    priorX, priorY = prior_sol.y
    priorT = prior_sol.t

    plt.rcParams["font.family"] = "serif"

    fig = plt.figure()
    axs = GridSpec(2, 2, width_ratios=[1, 1])
    axs1 = fig.add_subplot(axs[0, 1])
    axs2= fig.add_subplot(axs[1, 1], sharex=axs1)
    axs3 = fig.add_subplot(axs[:, 0])


    axs1.plot(trueT, x, 'k-', label='true equations')
    axs1.plot(priorT, priorX, 'b-', label='prior equations')
    axs1.set_ylabel("x")

    axs2.plot(trueT, y, 'k-')
    axs2.plot(priorT, priorY, 'b-')
    axs2.set_ylabel("y")


    fig.supxlabel("t")

    axs3.plot(x, y, 'k-', label="true equations")
    axs3.plot(priorX, priorY, 'b-', label="prior equations")
    axs3.set_ylabel("y")
    axs3.set_xlabel("x")

    plt.legend()
    plt.tight_layout()
    plt.show()


#create PIN-sindy object to train data

dmethod = dmethods(X, t)
diff = dmethod.forward_difference()

theta_instance = theta(X, 2)

lbd_guess = np.logspace(-10, 2, 50)
x_err = np.inf
y_err = np.inf

print(theta_instance.library().shape)
lbd=1e-6
sindy = SINDY(dmethod, theta_instance, X, lbd=lbd)
model = sindy.model(fi)
sol = sindy.simulate(t_span, x0, t)
solx, soly = sol.y

## plot results
if plot_results:
    fig, axs = plt.subplots(2,1, sharex=True)
    axs[0].plot(t, x, linestyle="dashed", label="observed x")
    axs[0].plot(t, solx, label="pin-sindy x")
    axs[0].legend()
    axs[0].set_ylabel("x")

    axs[1].plot(t, y, linestyle="dashed", label="observed y")
    axs[1].plot(t, soly, label="pin-sindy y")
    axs[1].legend()
    axs[1].set_ylabel("y")
    axs[1].set_xlabel("t")

    plt.suptitle(f"PIN-SINDY \n lbd={lbd: .2e} \n PIN-SINDy coefficients: {sindy.get_coef()} \n true coefficients: {true_coef} ")
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

if plot_opt_lbd:
    lbd_space = np.logspace(-7, 3, 50)
    zeta = []

    for lbd in lbd_space:
        sindy = SINDY(dmethod, theta_instance, X, lbd)
        model = sindy.model(fi)
        sol = sindy.simulate(t_span, x0, t, fi=fi)
        sindy_coef = sindy.get_coef()

        zeta_term = np.linalg.norm(sindy_coef - true_coef, 2)/ np.linalg.norm(sindy_coef+fi, 2)
        zeta.append(zeta_term)

    zeta = np.array(zeta)

    plt.plot(lbd_space, zeta, color='orange')
    plt.xlabel('lambda values')
    plt.ylabel('coefficient error (scaled)')
    plt.title('PIN-SINDy Lotka-Volterra System Error')
    plt.grid()
    plt.show()
