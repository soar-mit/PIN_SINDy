from SINDyPySource import SINDY, theta, dmethods
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import itertools
np.set_printoptions(precision=2, suppress=True)

true_vs_prior = True

R = 100
L = 1
C = 1

def true_equation(t, I):
    ddI = -R/L*I[1] - 1/(L*C)*I[0]
    return [I[1], ddI]


t = np.linspace(0, 20, 2000)
t_span = [0, 200]
I0 = (1, 1)

I, dI = integrate.solve_ivp(true_equation, t_span, I0, t_eval=t).y

if true_vs_prior:
    def prior_equation(t, I):
        ddI = -I[0] - I[1]
        return [I[1], ddI]



    Iprior, dIprior = integrate.solve_ivp(prior_equation, t_span, I0, method="LSODA", t_eval=t).y

    fig = plt.figure()
    axs = GridSpec(2, 2, width_ratios=[1, 1])
    axs_1 = fig.add_subplot(axs[0, 1])
    axs_2 = fig.add_subplot(axs[1, 1], sharex=axs_1)
    axs0 = fig.add_subplot(axs[:, 0])

    axs_1.plot(t, I, color="gold")
    axs_1.plot(t, Iprior, linestyle="dashed", color="thistle")
    axs_1.set_ylabel('I')

    axs_2.plot(t, dI, color="gold")
    axs_2.plot(t, dIprior, linestyle="dashed", color="thistle")
    axs_2.set_ylabel('dI')
    axs_2.set_xlabel('t')

    axs0.plot(I, dI, label="true", color="khaki")
    axs0.plot(Iprior, dIprior, label="prior", linestyle="dashed", color="thistle")

    fig.legend()
    plt.show()

## create fi and true coefficient matrices
fi = np.zeros((2, 5))
true_coef = np.zeros((2, 5))

true_coef[0, 1] = 1
true_coef[1, 0] = -1/(L*C)
true_coef[1, 1] = -R/L

fi[0, 1] = 1
fi[1, 0] = -1
fi[1, 1] = -1

## create SINDy objects
diff = dmethods(I, t)
theta_library = theta(I, 2)

sindy_p = SINDY(diff, I, t, t_span, I0, theta_instance=theta_library, lbd=.01, regressor="lstsq")
sindy_s = SINDY(diff, I, t, t_span, I0, theta_instance=theta_library, lbd=.01, regressor="lstsq")

coef_p = sindy_p.model(fi)['coef']
coef_s = sindy_s.model(fi)['coef']
print(true_coef)
print(coef_p)
print(coef_s)
