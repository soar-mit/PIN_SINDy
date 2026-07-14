import numpy as np
import matplotlib.pyplot as plt
from SINDyPySource import SINDY, dmethods, theta
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection

def lorentz(t, x):
    dx = -10*x[0] + 10*x[1]
    dy = 28*x[0] - x[0]*x[2] - x[1]
    dz = x[0]*x[1] - (8/3)*x[2]
    return [dx, dy, dz]

t_span = (0, 100)
t = np.linspace(0, 100, 10000)
x0 = (-8, 7, 27)

x = integrate.solve_ivp(lorentz, t_span, x0, t_eval=t)

dmethod = dmethods(x, t)
library = theta(x, 5)
sindy = SINDY(dmethod, x, t_eval=t, t_span=t_span, u0=x0, method="RK45", theta_instance = library, lbd=.001
              stre="first derivative second order")

model = sindy.model()
print(model.get_coef())
sim = sindy.simulate()
sol = sim.y

fig = plt.figure()
axs0 = fig.add_subplot((2, 1, 1), projection='3d')
axs1 = fig.add_subplot((2, 1, 2), projection='3d')

solpoints = np.array([sol[0], sol[1], sol[2]]).T.reshape(-1, 1, 3)
solsegments = np.concatenate([solpoints[:-1], solpoints[1:]], axis=1)

axs0.plot(sol[0], sol[1], sol[2])
axs0.set_xlabel('x')
axs0.set_ylabel('y')
axs0.set_zlabel('z')
lc = Line3DCollection(solsegments, cmap='jet')
lc.set_array(sol[2][:-1])  # color by z
axs0.add_collection3d(lc)
