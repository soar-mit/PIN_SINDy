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
    dx = -.1*x[0] + 2*x[1]
    dy = -.1*x[1] - 2*x[0]
    return [dx, dy]

t_span =  (0, 25)
t = np.linspace(0, 25, 2500)
x0 = (2, 2)

X = integrate.solve_ivp(linear, t_span, x0, t_eval=t).y

dmethod = dmethods(X, t)
library = theta(X, 3)
sindy_l = SINDY(dmethod, X, t, t_span, x0, method="RK45", theta_instance=library,
                 lbd=.0001)

model_l = sindy_l.model()
print(sindy_l.get_coef())
sol_l = sindy_l.simulate().y
print(sindy_l.simulate().success)

def cubic(t, x):
    dx = -.1*x[0]**3 + 2*x[1]**3
    dy = -.1*x[1]**3 - 2*x[0]**3
    return [dx, dy]

X_c = integrate.solve_ivp(cubic, t_span, x0, t_eval=t).y

dmethod_c = dmethods(X_c, t)
library_c = theta(X_c, 3)
sindy_c = SINDY(dmethod_c, X_c, t, t_span, x0,
                method="RK45", theta_instance=library_c, lbd=.0001)

model_c = sindy_c.model()
print(sindy_c.get_coef())
sol_c = sindy_c.simulate().y
print(sindy_c.simulate().success)

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(t, sol_l[0], label="x1", color="r")
axs[0,0].plot(t, sol_l[1], label="x2", color="b")
axs[0,0].plot(t, X[0], label="model", color="k", linestyle="dashed")
axs[0,0].plot(t, X[1], label="model", color="k", linestyle="dashed")
axs[0,0].set_xlabel("t")
axs[0,0].set_ylabel("x_k")
axs[0,0].set_title("Linear System")

axs[0, 1].plot(t, sol_c[0], color="r")
axs[0,1].plot(t, sol_c[1], color="b")
axs[0,1].plot(t, X_c[0], color="k", linestyle="dashed")
axs[0,1].plot(t, X_c[1], color="k", linestyle="dashed")
axs[0,1].set_xlabel("t")
axs[0,1].set_ylabel("x_k")
axs[0, 1].set_title("Cubic System")

axs[1, 0].plot(sol_l[0], sol_l[1], color="r", label="x_k")
axs[1, 0].plot(X[0], X[1], label="model", color="k", linestyle="dashed")
axs[1, 0].set_xlabel("x_1")
axs[1, 0].set_ylabel("x_2")

axs[1, 1].plot(sol_c[0], sol_c[1], color="r")
axs[1, 1].plot(X_c[0], X_c[1], color="k", linestyle="dashed")
axs[1, 1].set_xlabel("x_1")
axs[1, 1].set_ylabel("x_2")

plt.show()
