import numpy as np
import matplotlib.pyplot as plt
import henon_map as hm
import dynamicimshow as dim
from matplotlib.widgets import TextBox, Slider, Button
import warnings

def test_inversion(xmin, xmax, ymin, ymax, epsilon=0.0, mu=0.0, th1=0.0, th2=0.0, n_turns=1000, err=None, special_view=None):
    sampling = 300   # SAMPLING FOR THE GIVEN VIEW
    inversion = n_turns # HOW MANY TURNS?

    if mu != 0.0:
        warnings.warn("Inverse map with octupolar kick not yet implemented! mu!=0 will have no actual effect!")

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

    engine = hm.partial_track.generate_instance(xxf, pxf, yyf, pyf)
    _, _, _, _, _ = engine.compute(inversion, epsilon)
    x, px, y, py, _ = engine.inverse_compute(inversion, epsilon)
    data = np.log10(
        np.sqrt(
            + np.power(x - xxf, 2) 
            + np.power(px - pxf, 2) 
            + np.power(y - yyf, 2) 
            + np.power(py - pyf, 2)))
    return data.reshape((sampling, sampling))


if __name__ == "__main__":
    initial_epsilon = 0.0
    initial_mu = 0.0
    initial_nturns = 1000
    ex0 = (0,1,0,1)
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.3)
    f = test_inversion
    cbarlabel = "Inversion error (log scale)"
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

    text_box_err = TextBox(axbox4, "")
    text_box_err.set_val(str(-1))

    button_1 = Button(axbut1, "$X$ - $P_X$")
    button_2 = Button(axbut2, "$Y$ - $P_Y$")

    ax.set_title("Inversion error")

    print(fig.axes)
    plotter = dim.dynamic_imshow(
        ax, fig, cbar, cbarlabel, axcmap, bth1, bth2, text_box_eps, text_box_mu, text_box_t, text_box_err, button_1, button_2, f, (ex0[0], ex0[1], ex0[2], ex0[3]), initial_epsilon, initial_mu, initial_nturns, None)
    plt.show()
