
import numpy as np
import matplotlib.pyplot as plt
from PIN_SINDy.Case_Study_A.Python.SINDyPySource import SINDY, dmethods, theta
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d  import Line3DCollection

def lorentz(t, x):
    dx = -10*x[0] + 10*x[1]
    dy = 28*x[0] - x[0]*x[2] - x[1]
    dz = x[0]*x[1] - (8/3)*x[2]
    return [dx, dy, dz]

t_span = (0, 100)
t_span_short = (0, 20)
t_span_long = (0, 250)
t = np.linspace(0, 100, 100000)
t_short = np.linspace(0, 20, 20000)
t_long = np.linspace(0, 250, 250000)
x0 = (-8, 7, 27)

x = integrate.solve_ivp(lorentz, t_span, x0, t_eval=t).y

dmethod = dmethods(x, t)
library = theta(x, 2)
sindy = SINDY(dmethod, x, norm_optimization=False, t_eval=t, t_span=t_span, u0=x0, method="RK45", theta_instance = library, lbd=10,
              regressor="lasso", threshold=1e-3)

model = sindy.model()
print(np.round(sindy.get_coef(), 2))
sim = sindy.simulate()
sol = sim.y

fig = plt.figure()
axs0 = fig.add_subplot(211, projection='3d')
axs1 = fig.add_subplot(212, projection='3d')

axs0.plot(sol[0], sol[1], sol[2])
axs0.set_xlabel('x')
axs0.set_ylabel('y')
axs0.set_zlabel('z')
axs0.set_title("sindy solution. time span 0-100")

axs1.plot(x[0], x[1], x[2])
axs1.set_xlabel('x')
axs1.set_ylabel('y')
axs1.set_zlabel('z')
axs1.set_title("model. time span 0-100")

plt.show()
print(np.round(sindy.get_coef(), 2))
