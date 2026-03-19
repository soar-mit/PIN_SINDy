import csv
import numpy as np
from sklearn.linear_model import Lasso
import itertools
from scipy import integrate
import matplotlib.pyplot as plt
from SINDyPySource import theta, dmethods, SINDY
import math


#SINDy Test - 1 Variable

## create test function: dx/dt = 3x ---> x = e^3t u0 = 1

t = np.linspace(0, 5, 50)

x = [(math.e)**(3*i) for i in t]
x = [x]

## create SINDy Function:

theta_lib = theta(x, 2)
dmethod = dmethods(x, t)

sindy = SINDY(dmethod, theta_lib, x, lbd=.05)
model = sindy.model()

sol1 = model['model']

# graph differentials:

xdiff = [3*(math.e)**(3*i) for i in t]

plt.plot(t, xdiff, label="x differential")

plt.plot(t, sol1[0], label ="SINDy")
plt.plot(t, dmethod.forward_difference()[0], label="x1 gradient")
plt.legend()

plt.show()
