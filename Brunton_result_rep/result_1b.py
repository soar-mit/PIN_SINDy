"""
Result 1: Damped harmonic oscillator with linear terms.
 correctly identify model with up to second or third order polynomials in library.
 Should be accurate with up to fifth order polynomial terms.

X0 = (2, 2)or X0 = (2, 2, 2)

t_span = (0, 25). Time steps not given so try 2500.

"""

import numpy as np
import matplotlib.pyplot as plt
from Case_Study_A.Python.SINDyPySource import SINDY, dmethods, theta
from scipy import integrate

def linear(t, x):
    dx = -.1*x[0] - 2*x[1]
    dy = -.1*x[1] + 2*x[0]
    dz = -.3*x[2]
    return [dx, dy, dz]

t_span =  (0, 50)
t = np.linspace(0, 50, 5000)
x0 = (2, 2, 1)

x = integrate.solve_ivp(linear, t_span, x0, t_eval=t).y

dmethod = dmethods(x, t)
library = theta(x, 2)
sindy = SINDY(dmethod, x, t, t_span, x0, method="RK45", theta_instance=library,
                 lbd=.00004)

model = sindy.model()
print(sindy.get_coef())
sol = sindy.simulate().y


fig = plt.figure()
axs0 = fig.add_subplot(121)
axs1 = fig.add_subplot(122, projection='3d')

axs0.plot(t, sol[0], color="r", label="x_1")
axs0.plot(t, sol[1], color="b", label="x_2")
axs0.plot(t, sol[2], color="g", label="x_3")
axs0.plot(t, x[0], color="k", linestyle="dashed", label="model")
axs0.plot(t, x[1], color="k", linestyle="dashed", label="model")
axs0.plot(t, x[2], color="k", linestyle="dashed", label="model")
axs0.set_xlabel("time")
axs0.set_ylabel("x_k")

axs1.plot(x[0], x[1], x[2], color="k", linestyle="dashed")
axs1.plot(sol[0], sol[1], sol[2], color="r")
axs1.set_xlabel("x_1")
axs1.set_ylabel("x_2")
axs1.set_zlabel("x_3")
axs1.set_xbound(-2, 2)
axs1.set_ybound(-2, 2)
axs1.set_zbound(0, 1)

plt.legend()
plt.show()
