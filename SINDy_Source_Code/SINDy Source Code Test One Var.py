import numpy as np
import matplotlib.pyplot as plt
from Case_Study_B.SINDyPySource import theta, dmethods, SINDY, STLSQ
import math


#SINDy Test - 1 Variable

## create test function: dx/dt = 3x ---> x = e^3t u0 = 1

t = np.linspace(0, 5, 50)

x = [(math.e)**(3*i) for i in t]
x = [x]

## create SINDy Function:

theta_lib = theta(x, 1)
dmethod = dmethods(x, t)

sindy = SINDY(dmethod, theta_lib, x, lbd=1)
model = sindy.model()

sol1 = model['model']

t_span = (t[0], t[-1])
u0 = [1]
sim = (sindy.simulate(t_span, u0, t)).y[0]

lib = theta_lib.library()
diff = dmethod.forward_difference()


stlsq = STLSQ(diff, lib)

# graph differentials - test 'passed'


# plt.plot(t, sol1[0], label ="SINDy")
# plt.plot(t, dmethod.forward_difference()[0], label="x1 gradient")
# plt.legend()

# plt.show()

#graph solution - test 'passed'

plt.plot(t, x[0], label="original data")
plt.plot(t, sim, label="SINDy Solution")
plt.legend()
plt.show()
