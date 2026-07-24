from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from SINDyPySource import SINDY, theta, dmethods

# Goal: generate Pure SINDy and PIN-SINDy model for comparison. for Lotka-Volterra Equations
true_vs_prior = True
plot_model = False
plot_error = False

# create training data:
def true_forcing(t, x):
    dx = .5*x[0] - .2*x[0]*x[1] - x[0]**2*x[1]
    dy = -.3*x[1] + .4*x[0]*x[1] - x[1]**2*x[0]
    return [dx, dy]

def prior_forcing(t, x):
    dx = x[0] - x[0]*x[1]
    dy = x[0]*x[1] - x[1]
    return [dx, dy]

true_coef = np.zeros((2, 9))
true_coef[0, 0] = .5
true_coef[0, 3] = -.2
true_coef[0, 6] = -1


true_coef[1, 1] = -.3
true_coef[1, 3] = .4
true_coef[1, 7] = -1

fi = np.zeros((2, 9))
fi[0, 0] = 1
fi[0, 3] = -1
fi[1, 1] = -1
fi[1, 3] = 1

x0 = [5, 2]

t = np.linspace(0, 20, 20000)
t_span = (0, 20)

true_sol = integrate.solve_ivp(true_forcing, t_span, x0, t_eval=t)
x, y = true_sol.y
X = np.array((x, y))

#create PIN-SINDy object to train data

dmethod = dmethods(X, t)

theta_instance = theta(X, 3)

sindy_pin = SINDY(dmethod, X, t, t_span, x0, theta_instance=theta_instance, lbd=.01, regressor="lasso")
model_pin = sindy_pin.model(fi)
sol_pin = sindy_pin.simulate(fi).y
print("pin-sindy model simulated...")

#create pure SINDy object

sindy_pure = SINDY(dmethod, X, t, t_span, x0, theta_instance=theta_instance, lbd=1, regressor="lasso")
print("sindy model created...")
model_pure = sindy_pure.model()
sol_pure = sindy_pure.simulate().y

## plot results


plt.rcParams["font.family"] = "serif"

if true_vs_prior:

    prior_sol = integrate.solve_ivp(prior_forcing, t_span, x0, t_eval=t)

    fig = plt.figure()
    axs = GridSpec(2, 2, width_ratios=[1, 1])
    axs1 = fig.add_subplot(axs[0, 1])
    axs2= fig.add_subplot(axs[1, 1], sharex=axs1)
    axs3 = fig.add_subplot(axs[:, 0])

    x, y = X
    priorx, priory = prior_sol.y

    axs1.plot(t, x, 'b-', label='true')
    axs1.plot(t, priorx, 'y-', linestyle="dashed", label='prior')
    axs1.set_ylabel("x")
    axs1.legend()

    axs2.plot(t, y, 'b-')
    axs2.plot(t, priory, 'y-', linestyle="dashed")
    axs2.set_ylabel("y")
    axs2.legend()


    fig.supxlabel("t")

    axs3.plot(x, y, 'b-', label="true")
    axs3.plot(priorx, priory, 'y-', linestyle="dashed", label="prior")
    axs3.set_ylabel("y")
    axs3.set_xlabel("x")
    axs3.legend()

    plt.suptitle(f"true vs prior equations")
    plt.tight_layout()
    plt.show()

if plot_model:
    print("plotting model...")
    fig, axs = plt.subplots(2,1)

    pinx, piny = sol_pin
    purex, purey = sol_pure

    axs[0].plot(t, pinx, 'b-', label='PIN-SINDy')
    axs[0].plot(t, purex, 'y-', label='pure SINDy')
    axs[0].plot(t, x, 'r-', linestyle="dashed", label="model")
    axs[0].set_ylabel("x")
    axs[0].legend()

    axs[1].plot(t, piny, 'b-', label="PIN-SINDy")
    axs[1].plot(t, purey, 'y-', label="pure SINDy")
    axs[1].plot(t, y, 'r-', linestyle="dashed", label="model")
    axs[1].set_ylabel("y")
    axs[1].legend()


    fig.supxlabel("t")

    plt.suptitle(f"pure SINDy coef: {sindy_pure.get_coef()} \n PIN-SINDy coef: {sindy_pin.get_coef()} \n real coef: {true_coef}")
    plt.tight_layout()
    plt.show()

if plot_error:

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].plot(t, abs(x - pinx)/x)
    axs[0, 0].set_ylabel("pin-sindy x")
    axs[1, 0].plot(t, abs(y-piny)/y)
    axs[1,0].set_ylabel("pin-sindy y")
    axs[1,0].set_xlabel("t")
    axs[1, 0].set_title("pin-sindy model")

    axs[0, 1].plot(t, abs(x - purex)/x)
    axs[0, 1].set_ylabel("pure sindy x")
    axs[1, 1].plot(t, abs(y - purey)/y)
    axs[1, 1].set_ylabel("pure sindy y")
    axs[1, 1]. set_xlabel("t")
    axs[0, 1].set_title("pure sindy model")
    plt.tight_layout()
    plt.show()

print(f'sindy difference: {abs(true_coef - sindy_pure.get_coef())}')
print(f'pin sindy difference: {abs(true_coef - sindy_pin.get_coef())}')
