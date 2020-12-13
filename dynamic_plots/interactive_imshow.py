import numpy as np
import henon_map as hm
import matplotlib.pyplot as plt

class interactive_imshow:
    def __init__(self, function_list, function_names, starting_extent):
        self.function_list = function_list
        self.function_names = function_names
        self.ex0 = starting_extent

        self.epsilon = 0.0
        self.mu = 0.0
        self.turns = 1000
        
        self.fig, self.ax = plt.subplots()
        self.fig.subplots_adjust(bottom=0.3)

        img = self.ax.imshow(self.function_list[0](
            self.ex0[0], self.ex0[1], self.ex0[2], self.ex0[3]),
            extent=(self.ex0[0], self.ex0[1], self.ex0[2], self.ex0[3]), 
            origin="lower"
        )
        
        self.cbar = fig.colorbar(img, label=cbarlabel)
        self.axcmap = fig.axes[-1]
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
