from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIN_SINDy.Case_Study_B.SINDyPySource import SINDY, theta, dmethods

# Goal: generate Pure SINDy and PIN-SINDy model for comparison. for Lotka-Volterra Equations
true_vs_prior = False
coef_error_norm = True
coef_error_each = False

# create training data:
def true_forcing(t, x):
    dx = .5*x[0] - .2*x[0]*x[1] - .1*x[0]**2
    dy = -.3*x[1] + .4*x[0]*x[1] - .1*x[1]**2
    return [dx, dy]

def prior_forcing(t, x):
    dx = x[0] - x[0]*x[1]
    dy = x[0]*x[1] - x[1]
    return [dx, dy]

true_coef = np.zeros((2, 9))
true_coef[0, 0] = .5
true_coef[0, 3] = -.2
true_coef[0, 2] = -.1


true_coef[1, 1] = -.3
true_coef[1, 3] = .4
true_coef[1, 4] = -.1

fi = np.zeros((2, 9))
fi[0, 0] = 1
fi[0, 3] = -1
fi[1, 1] = -1
fi[1, 3] = 1

x0 = [5, 2]

t = np.linspace(0, 20, 2000)
t_span = (0, 20)

true_sol = integrate.solve_ivp(true_forcing, t_span, x0, t_eval=t)
x, y = true_sol.y
X = np.array((x, y))

#create PIN-SINDy object to train data

dmethod = dmethods(X, t)

theta_instance = theta(X, 3)


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

lbds = np.logspace(-8, 1, 100)

if coef_error_each:


    zeta_sindy_x = []
    zeta_pin_x = []
    zeta_sindy_y = []
    zeta_pin_y = []
    zeta_sindy_x_xy = []
    zeta_pin_x_xy = []
    zeta_sindy_y_xy = []
    zeta_pin_y_xy = []
    zeta_sindy_x2 = []
    zeta_pin_x2 = []
    zeta_sindy_y2 = []
    zeta_pin_y2 = []

    for lbd in lbds:
        sindy_l = SINDY(dmethod, X, t_eval=t, t_span=t_span, u0=x0, theta_instance=theta_instance,
                        norm_optimization=False, lbd=lbd, regressor = "lasso")
        coef_sindy_l = sindy_l.model()['coef']
        coef_pin_l = sindy_l.model(fi)['coef']

        zeta_sindy_x.append(abs((coef_sindy_l[0,0]-true_coef[0, 0])/coef_sindy_l[0,0]))
        zeta_pin_x.append(abs((coef_pin_l[0,0]-true_coef[0, 0])/coef_pin_l[0,0]))

        zeta_sindy_y.append(abs((coef_sindy_l[1,1]-true_coef[1,1])/coef_sindy_l[1,1]))
        zeta_pin_y.append(abs((coef_pin_l[1,1]-true_coef[1,1])/coef_pin_l[1,1]))

        zeta_sindy_x_xy.append(abs((coef_sindy_l[0,3]-true_coef[0,3])/coef_sindy_l[0,3]))
        zeta_pin_x_xy.append(abs((coef_pin_l[0,3] - true_coef[0,3])/coef_pin_l[0,3]))

        zeta_sindy_y_xy.append(abs((coef_sindy_l[1,3] - true_coef[1,3])/coef_sindy_l[1,3]))
        zeta_pin_y_xy.append(abs((coef_pin_l[1,3] - true_coef[1,3])/coef_pin_l[1,3]))

        zeta_sindy_x2.append(abs((coef_sindy_l[0,2] - true_coef[0,2])/coef_sindy_l[0,2]))
        zeta_pin_x2.append(abs((coef_pin_l[0,2] - true_coef[0,2])/coef_pin_l[0,2]))

        zeta_sindy_y2.append(abs((coef_sindy_l[1,4] - true_coef[1,4])/coef_sindy_l[1,4]))
        zeta_pin_y2.append(abs((coef_pin_l[1,4] - true_coef[1,4])/coef_pin_l[1,4]))

    ## helper function for graphing
    fix, axs = plt.subplots(3)

    def plot_coef_error(zeta_sindy, zeta_pin, coef_name, axis, axes):
        zeta_sindy = np.array(zeta_sindy)
        zeta_pin = np.array(zeta_pin)
        label_s = 'sindy'
        label_p = 'pin'

        index_sindy = np.argmin(zeta_sindy)
        index_pin = np.argmin(zeta_pin)

        axes[axis].plot(lbds, zeta_sindy, color='lightcoral', label=label_s)
        axes[axis].plot(lbds, zeta_pin, color='lightskyblue', label=label_p)
        axes[axis].plot(lbds[index_sindy], zeta_sindy[index_sindy], color='firebrick', marker='o', markersize=5)
        axs[axis].plot(lbds[index_pin], zeta_pin[index_pin], color='royalblue', marker='o', markersize=5)
        axes[axis].set_xscale('log')
        axes[axis].set_xlabel('λ values')
        axes[axis].set_ylabel(f'coef error for coef: {coef_name}')

        axes[axis].annotate(
    f'sindy min: {zeta_sindy[index_sindy]:.3g}\nλ={lbds[index_sindy]:.3g}',
    xy=(lbds[index_sindy], zeta_sindy[index_sindy]),       # point being annotated
    xytext=(15, 15), textcoords='offset points',              # offset of text box
    fontsize=8, color='firebrick',
    arrowprops=dict(arrowstyle='->', color='firebrick', lw=1)
)

# annotate pin-sindy minimum
        axes[axis].annotate(
    f'pin min: {zeta_pin[index_pin]:.3g}\nλ={lbds[index_pin]:.3g}',
    xy=(lbds[index_pin], zeta_pin[index_pin]),
    xytext=(15, -25), textcoords='offset points',
    fontsize=8, color='royalblue',
    arrowprops=dict(arrowstyle='->', color='royalblue', lw=1)
)

    plot_coef_error(zeta_sindy_x, zeta_pin_x, 'x', 0, axs)
    plot_coef_error(zeta_sindy_x_xy, zeta_pin_x_xy, 'xy', 1, axs)
    plot_coef_error(zeta_sindy_x2, zeta_sindy_x2, 'x^2', 2, axs)

    plt.legend()
    plt.tight_layout()
    plt.show()

    fig, axs= plt.subplots(3)

    plot_coef_error(zeta_sindy_y, zeta_pin_y, 'y', 0, axs)
    plot_coef_error(zeta_sindy_y_xy, zeta_pin_y_xy, 'xy', 1, axs)
    plot_coef_error(zeta_sindy_y2, zeta_pin_y2, 'y^2', 2, axs)

    plt.legend()
    plt.tight_layout()
    plt.show()

if coef_error_norm:
    error_cap = 1
    zeta_sindy = []
    zeta_pin = []

    min_error_sindy = np.inf
    opt_lbd_sindy = 0
    lbds_sindy_ok = 0

    min_error_pin = np.inf
    opt_lbd_pin = 0
    lbds_pin_ok = 0

    for lbd in lbds:
        sindy_l = SINDY(dmethod, X, t_eval=t, t_span=t_span, u0=x0, theta_instance=theta_instance,
                        norm_optimization=True, lbd=lbd, regressor = "lstsq")
        coef_sindy_l = sindy_l.model()['coef']
        coef_pin_l = sindy_l.model(fi)['coef']
        sindy_norm = abs(np.linalg.norm(coef_sindy_l))
        if sindy_norm == 0:
            sindy_norm = 1
        zeta_term_sindy = np.linalg.norm(abs(coef_sindy_l - true_coef)) / sindy_norm
        zeta_sindy.append(zeta_term_sindy)

        if zeta_term_sindy <= min_error_sindy:
            min_error_sindy = zeta_term_sindy
            opt_lbd_sindy = lbd

        if zeta_term_sindy < error_cap:
            lbds_sindy_ok += 1

        pin_norm = abs(np.linalg.norm(coef_pin_l))
        if pin_norm == 0:
            pin_norm = 1

        zeta_term_pin = np.linalg.norm(abs(coef_pin_l - true_coef)) / pin_norm
        zeta_pin.append(zeta_term_pin)

        if zeta_term_pin <= min_error_pin:
            min_error_pin = zeta_term_pin
            opt_lbd_pin = lbd
        if zeta_term_pin < error_cap:
            lbds_pin_ok += 1

    zeta_sindy = np.array(zeta_sindy)
    zeta_pin = np.array(zeta_pin)

    plt.plot(lbds, zeta_sindy, color='lightcoral', label='sindy')
    plt.plot(opt_lbd_sindy, min_error_sindy, marker='o', markersize=5, color='firebrick')
    plt.xscale('log')
    plt.xlabel('λ values')
    plt.ylabel('normalized coefficient error')
    plt.annotate(
            f'sindy min: {min_error_sindy:.3g}\nλ={opt_lbd_sindy:.3g}',
            xy=(opt_lbd_pin, min_error_pin),
            xytext=(15, -25), textcoords='offset points',
            fontsize=8, color='firebrick',
            arrowprops=dict(arrowstyle='->', color='firebrick', lw=1)
        )

    plt.plot(lbds, zeta_pin, color='lightskyblue', label='pin-sindy')
    plt.plot(opt_lbd_pin, min_error_pin, marker='o', markersize=5, color='royalblue')
    plt.annotate(
        f'pin min: {min_error_pin:.3g}\nλ={opt_lbd_pin:.3g}',
        xy=(opt_lbd_pin, min_error_pin),
        xytext=(15, 25), textcoords='offset points',
        fontsize=8, color='royalblue',
        arrowprops=dict(arrowstyle='->', color='royalblue', lw=1)
    )

    plt.legend()
    plt.tight_layout()
    plt.show()
