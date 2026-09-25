import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from Case_Study_B.SINDyPySource import theta, dmethods, SINDY

# #create toy problem:

def f(t, x):
    #x is list of state variables - in this case [[x1], [x2]]
    dx1 = -x[0] + x[0]*x[1]
    dx2 = -x[1]**2
    return [dx1, dx2]

t = np.linspace(0, 10, 1000)
x0 = [1, 1]
tspan = (t[0], t[-1])

x = integrate.solve_ivp(f, tspan, x0, t_eval=t).y
x1 = np.array(x[0])
x2 = np.array(x[1])


# # create SINDy model

dmethod = dmethods(x, t)

theta_instance = theta(x, 2)

sindy = SINDY(dmethod, theta_instance, x, lbd=1e-4)
mod = sindy.model()['model']


sol = sindy.simulate(tspan, x0, t)
sol1 = sol.y[0]
sol2 = sol.y[1]

print(sindy.get_coef())

# check 1 - plot derivatives:

x1_prime, x2_prime = dmethod.forward_difference()

fig, axs = plt.subplots(2, 1)


# axs[0].plot(t, x1_prime, label="dmethods x1 prime")
# axs[0].plot(t, x2_prime, label="dmethods x2 prime")

# axs[1].plot(t, mod[0], label="sindy x1 prime")
# axs[1].plot(t, mod[1], label="sindy x2 prime")


## check 2 - plot actual solution:

axs[0].plot(t, x1, label="original x1")
axs[0].plot(t, x2, label = "original x2")

axs[1].plot(t, sol1, label="sindy solution x2")
axs[1].plot(t, sol2, label="sindy solution x2")

plt.legend()
plt.show()
