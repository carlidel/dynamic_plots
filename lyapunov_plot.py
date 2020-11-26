import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import henon_map as hm
import dynamicimshow as dim
from matplotlib.widgets import TextBox, Slider, Button


def test_li(xmin, xmax, ymin, ymax, epsilon=0.0, mu=0.0, th1=0.0, th2=0.0, n_turns=1000, err=0.0001, special_view=None):
    sampling = 300          # SAMPLING FOR THE GIVEN VIEW
    steps = n_turns         # NUMBER OF STEPS
    initial_err = err       # ERROR DISLOCATION
    x0 = np.linspace(xmin, xmax, sampling+2)[1:-1]
    y0 = np.linspace(ymin, ymax, sampling+2)[1:-1]
    xx, yy = np.meshgrid(x0, y0)

    if special_view == "x_px":
        xxf = xx.flatten()
        yyf = np.zeros_like(xxf)
        pxf = yy.flatten()
        pyf = np.zeros_like(yyf)
    elif special_view == "y_py":
        yyf = xx.flatten()
        xxf = np.zeros_like(yyf)
        pyf = yy.flatten()
        pxf = np.zeros_like(xxf)
    else:
        xxf = xx.flatten() * np.cos(th1)
        yyf = yy.flatten() * np.cos(th2)
        pxf = xx.flatten() * np.sin(th1)
        pyf = yy.flatten() * np.sin(th2)

    xxf = np.concatenate((xxf, xxf))
    yyf = np.concatenate((yyf, yyf))
    pxf = np.concatenate((pxf, pxf))
    pyf = np.concatenate((pyf, pyf))

    r_alpha = np.random.uniform(0, np.pi/2, sampling**2)
    th1_alpha = np.random.uniform(0, np.pi*2, sampling**2)
    th2_alpha = np.random.uniform(0, np.pi*2, sampling**2)

    xxf[len(xxf)//2:] += initial_err * np.cos(r_alpha) * np.cos(th1_alpha)
    pxf[len(pxf)//2:] += initial_err * np.cos(r_alpha) * np.sin(th1_alpha)
    yyf[len(yyf)//2:] += initial_err * np.sin(r_alpha) * np.cos(th2_alpha)
    pyf[len(pxf)//2:] += initial_err * np.sin(r_alpha) * np.sin(th2_alpha)

    engine = hm.partial_track.generate_instance(xxf, pxf, yyf, pyf)
    x, px, y, py, _ = engine.compute(steps, epsilon, mu)

    x = x[:len(x)//2] - x[len(x)//2:]
    px = px[:len(px)//2] - px[len(px)//2:]
    y = y[:len(y)//2] - y[len(y)//2:]
    py = py[:len(py)//2] - py[len(py)//2:]
    data = np.log10((np.power(x, 2) + np.power(px, 2) +
                     np.power(y, 2) + np.power(py, 2)) / initial_err)
    return data.reshape(sampling, sampling)


if __name__ == "__main__":
    initial_epsilon = 0.0
    initial_mu = 0.0
    initial_nturns = 1000
    initial_err = 0.0001
    ex0 = (0, 1, 0, 1)
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.3)
    f = test_li
    cbarlabel = "Lyapunov error (log scale)"
    img = ax.imshow(f(ex0[0], ex0[1], ex0[2], ex0[3]),
                    extent=(ex0[0], ex0[1], ex0[2], ex0[3]), origin="lower")
    cbar = fig.colorbar(img, label=cbarlabel)
    axcmap = fig.axes[-1]
    axbox1 = fig.add_axes([0.20, 0.05 - 0.04, 0.10, 0.075])
    axbox2 = fig.add_axes([0.35, 0.05 - 0.04, 0.10, 0.075])
    axbox3 = fig.add_axes([0.50, 0.05 - 0.04, 0.10, 0.075])
    axbox4 = fig.add_axes([0.65, 0.05 - 0.04, 0.10, 0.075])
    axcolor = 'lightgoldenrodyellow'
    axth2 = fig.add_axes([0.25, 0.15 - 0.04, 0.6, 0.03], facecolor=axcolor)
    axth1 = fig.add_axes([0.25, 0.20 - 0.04, 0.6, 0.03], facecolor=axcolor)

    axbut1 = fig.add_axes([0.02, 0.2 - 0.04, 0.10, 0.05])
    axbut2 = fig.add_axes([0.02, 0.1 - 0.04, 0.10, 0.05])

    bth1 = Slider(axth1, '$\\theta_1$ (rad)', 0.0,
                  2.0, valinit=0.0, valstep=0.01)
    bth2 = Slider(axth2, '$\\theta_2$ (rad)', 0.0,
                  2.0, valinit=0.0, valstep=0.01)

    text_box_eps = TextBox(axbox1, "$\\varepsilon$")
    text_box_eps.set_val(str(initial_epsilon))
    
    text_box_mu = TextBox(axbox2, "$\\mu$")
    text_box_mu.set_val(str(initial_mu))

    text_box_t = TextBox(axbox3, "$t$")
    text_box_t.set_val(str(initial_nturns))

    text_box_err = TextBox(axbox4, "err_0")
    text_box_err.set_val(str(initial_err))
    
    button_1 = Button(axbut1, "$X$ - $P_X$")
    button_2 = Button(axbut2, "$Y$ - $P_Y$")

    ax.set_title("Lyapunov error")

    print(fig.axes)
    plotter = dim.dynamic_imshow(
        ax, fig, cbar, cbarlabel, axcmap, bth1, bth2, text_box_eps, text_box_mu, text_box_t, text_box_err, button_1, button_2, f, (ex0[0], ex0[1], ex0[2], ex0[3]), initial_epsilon, initial_mu, initial_nturns, initial_err)
    plt.show()
