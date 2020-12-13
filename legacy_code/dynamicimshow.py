import numpy as np
import henon_map as hm
import matplotlib.pyplot as plt

def test_func(xmin, xmax, ymin, ymax):
    sampling = 100
    x = np.linspace(xmin, xmax, sampling+2)[1:-1]
    y = np.linspace(ymin, ymax, sampling+2)[1:-1]
    xx, yy = np.meshgrid(x, y, sparse=True, indexing="ij")
    return np.sin(xx) * np.cos(yy)


class dynamic_imshow:
    def __init__(self, ax, fig, cbar, cbarlabel, cbar_ax, slider_th1, slider_th2, text_box_eps, text_box_mu, text_box_t, text_box_err, button_1, button_2, data_func, starting_extent, epsilon, mu, t, err):
        # Fig stuff
        self.ax = ax
        self.fig = fig
        self.cbar = cbar
        self.cbarlabel = cbarlabel
        self.cbar_ax = cbar_ax
        self.slider_th1 = slider_th1
        self.slider_th2 = slider_th2
        self.text_box_eps = text_box_eps
        self.text_box_mu = text_box_mu
        self.text_box_t = text_box_t
        self.text_box_err = text_box_err
        self.button_1 = button_1
        self.button_2 = button_2
        self.special_view = None
        self.xlim = ax.get_xlim()
        self.ylim = ax.get_ylim()
        # function and extent
        self.data_func = data_func
        self.starting_extent = starting_extent
        # values
        self.epsilon = epsilon
        self.mu = mu
        self.t = t
        self.err = err
        if err is None:
            self.is_there_err = False
        else:
            self.is_there_err = True
        self.th1 = 0.0
        self.th2 = 0.0
        # set axis
        self.ax.set_xlabel("$X$ [a.u.]")
        self.ax.set_ylabel("$Y$ [a.u.]")
        # connect
        self.connect()

    def update_epsilon(self, string):
        print("Updated epsilon!!")
        self.epsilon = float(string)
        self.ax.set_xlim(self.ax.get_xlim()[0] - self.ax.get_xlim()[0] * 0.0000001,
                         self.ax.get_xlim()[1])

    def update_mu(self, string):
        print("Updated mu!!")
        self.mu = float(string)
        self.ax.set_xlim(self.ax.get_xlim()[0] - self.ax.get_xlim()[0] * 0.0000001,
                         self.ax.get_xlim()[1])

    def update_t(self, string):
        print("Updated t!!")
        self.t = int(string)
        self.ax.set_xlim(self.ax.get_xlim()[0] - self.ax.get_xlim()[0] * 0.0000001,
                         self.ax.get_xlim()[1])

    def update_error(self, string):
        if self.is_there_err:
            print("Updated initial error!!")
            self.err = float(string)
            self.ax.set_xlim(self.ax.get_xlim()[
                             0] - self.ax.get_xlim()[0] * 0.0000001, self.ax.get_xlim()[1])
        else:
            pass

    def update_theta(self, val):
        print("Updated theta!!")
        
        if self.special_view is not None:
            print("Reset special view!")
            self.special_view = None

        self.th1 = self.slider_th1.val * np.pi
        self.th2 = self.slider_th2.val * np.pi
        if self.th1 == 0.0 or self.th1 == 2.0 * np.pi:
            self.ax.set_xlabel("$X$ [a.u.]")
        elif self.th1 == 0.5 * np.pi:
            self.ax.set_xlabel("$P_X$ [a.u.]")
        elif self.th1 == 1.0 * np.pi:
            self.ax.set_xlabel("$-X$ [a.u.]")
        elif self.th1 == 1.5 * np.pi:
            self.ax.set_xlabel("$-P_X$ [a.u.]")
        else:
            self.ax.set_xlabel("$r_x$ [a.u]")

        if self.th2 == 0.0 or self.th2 == 2.0 * np.pi:
            self.ax.set_ylabel("$Y$ [a.u.]")
        elif self.th2 == 0.5 * np.pi:
            self.ax.set_ylabel("$P_Y$ [a.u.]")
        elif self.th2 == 1.0 * np.pi:
            self.ax.set_ylabel("$-Y$ [a.u.]")
        elif self.th2 == 1.5 * np.pi:
            self.ax.set_ylabel("$-P_Y$ [a.u.]")
        else:
            self.ax.set_ylabel("$r_y$ [a.u]")

        self.ax.set_xlim(self.ax.get_xlim()[0] - self.ax.get_xlim()[0] * 0.00001,
                         self.ax.get_xlim()[1])

    def on_press(self, event):
        print("You scrolled the wheel! Going back to home!")
        self.ax.set_xlim(self.starting_extent[0],
                         self.starting_extent[1])
        self.ax.set_ylim(self.starting_extent[2],
                         self.starting_extent[3])

    def s_x_px(self, event):
        print("Set special view X - PX")
        self.special_view = "x_px"
        self.ax.set_xlabel("$X$ [a.u.]")
        self.ax.set_ylabel("$P_X$ [a.u.]")
        self.ax.set_xlim(self.ax.get_xlim()[0] - self.ax.get_xlim()[0] * 0.00001,
                         self.ax.get_xlim()[1])

    def s_y_py(self, event):
        print("Set special view Y - PY")
        self.special_view = "y_py"
        self.ax.set_xlabel("$Y$ [a.u.]")
        self.ax.set_ylabel("$P_Y$ [a.u.]")
        self.ax.set_xlim(self.ax.get_xlim()[0] - self.ax.get_xlim()[0] * 0.00001,
                         self.ax.get_xlim()[1])


    def ax_update(self, event):
        self.ax.set_autoscale_on(False)
        print("Exectuing redraw with the following extent:")
        print("x_min={}, x_max={}, y_min={}, y_max={}".format(
            event.get_xlim()[0],
            event.get_xlim()[1],
            event.get_ylim()[0],
            event.get_ylim()[1]))
        data = self.data_func(
            self.ax.get_xlim()[0],
            self.ax.get_xlim()[1],
            self.ax.get_ylim()[0],
            self.ax.get_ylim()[1],
            self.epsilon, self.mu,
            self.th1, self.th2,
            self.t, self.err, self.special_view)
        im = self.ax.images[-1]
        im.set_data(data[::,::])
        im.set_extent((event.get_xlim()[0],
                       event.get_xlim()[1],
                       event.get_ylim()[0],
                       event.get_ylim()[1]))
        self.ax.figure.canvas.draw_idle()
        new_cbar = self.fig.colorbar(im, cax=self.cbar_ax, label=self.cbarlabel)
        self.cbar = new_cbar
        #self.cbar.set_clim(vmin=np.min(data), vmax=np.max(data))
        #cbar_ticks = np.linspace(np.min(data), np.max(data), num=9, endpoint=True)
        #self.cbar.set_ticks(cbar_ticks)
        #self.cbar.draw_all()
    
    def connect(self):
        self.x_com = self.ax.callbacks.connect("xlim_changed", self.ax_update)
        self.y_com = self.ax.callbacks.connect("ylim_changed", self.ax_update)
        self.press_com = self.fig.canvas.mpl_connect(
            'scroll_event', self.on_press)
        self.text_box_eps.on_submit(self.update_epsilon)
        self.text_box_mu.on_submit(self.update_mu)
        self.text_box_t.on_submit(self.update_t)
        self.text_box_err.on_submit(self.update_error)
        self.slider_th1.on_changed(self.update_theta)
        self.slider_th2.on_changed(self.update_theta)
        self.button_1.on_clicked(self.s_x_px)
        self.button_2.on_clicked(self.s_y_py)
