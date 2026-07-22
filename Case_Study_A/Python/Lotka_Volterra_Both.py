from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from SINDyPySource import SINDY, theta, dmethods

# Goal: generate Pure SINDy and PIN-SINDy model for comparison. for Lotka-Volterra Equations


# create training data:
def true_forcing(t, x):
    dx = .01*x[0] - .01*x[1]*x[0]
    dy = -.01*x[1] + .01*x[1]*x[0]
    return [dx, dy]

def prior_forcing(t, x):
    dx = x[0] - x[1]*x[0]
    dy = - x[1] + x[1]*x[0]
    return [dx, dy]


true_coef = np.array([[.01, 0, .01, -.01, 0], [0, -.01, 0, .01, 0]])
fi = np.array([[1, 0, 0, -1, 0], [0, -1, 0, 1, 0]])

x0 = [5, 2]

t = np.linspace(0, 10, 1000)
t_span = (0, 10)

true_sol = integrate.solve_ivp(true_forcing, t_span, x0, method="LSODA", t_eval=t)
x, y = true_sol.y
trueT = true_sol.t
X = np.array((x, y))

#create PIN-SINDy object to train data

dmethod = dmethods(X, t)
diff = dmethod.forward_difference()

theta_instance = theta(X, 2)

sindy_pin = SINDY(dmethod, X, t, t_span, x0, theta_instance=theta_instance,
                  lbd=.01, regressor="lstsq")

model_pin = sindy_pin.model(fi)

sol_pin = sindy_pin.simulate(fi).y
#create pure SINDy object

sindy_pure = SINDY(dmethod, X, t, t_span, x0, theta_instance=theta_instance, lbd=.01, regressor="lstsq")
model_pure = sindy_pure.model()
sol_pure = sindy_pure.simulate().y

## plot results

prior_sol = integrate.solve_ivp(prior_forcing, t_span, x0, t_eval=t)


plt.rcParams["font.family"] = "serif"

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

fig = plt.figure()
axs = GridSpec(2, 2, width_ratios=[1, 1])
axs1 = fig.add_subplot(axs[0, 1])
axs2= fig.add_subplot(axs[1, 1], sharex=axs1)
axs3 = fig.add_subplot(axs[:, 0])

pinx, piny = sol_pin
purex, purey = sol_pure

axs1.plot(t, pinx, 'b-', label='PIN-SINDy')
axs1.plot(t, purex, 'y-', linestyle="dashed", label='pure SINDy')
axs1.set_ylabel("x")
axs1.legend()

axs2.plot(t, piny, 'b-')
axs2.plot(t, purey, 'y-', linestyle="dashed")
axs2.set_ylabel("y")
axs2.legend()


fig.supxlabel("t")

axs3.plot(pinx, piny, 'b-', label="PIN-SINDy")
axs3.plot(purex, purey, 'y-', linestyle="dashed", label="pure SINDy")
axs3.set_ylabel("y")
axs3.set_xlabel("x")
axs3.legend()

plt.suptitle(f"pure SINDy coef: {sindy_pure.get_coef()} \n PIN-SINDy coef: {sindy_pin.get_coef()} \n real coef: {true_coef}")
plt.tight_layout()
plt.show()
