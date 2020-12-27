"""This module contains an example function wrapping of my Henon map engine,
in order to test and visualize the graphical interface made with plotly+dash
"""
import numpy as np
# Personal package henon_map
import henon_map as hm
# Local module function_wrap.py
from function_wrap import function_wrap, function_multi


def henon_call(init_list, n_turns, start_point, epsilon, mu):
    engine = hm.partial_track.generate_instance(
        init_list[0],
        init_list[1],
        init_list[2],
        init_list[3],
    )
    engine.total_iters = start_point
    x, px, y, py, _ = engine.compute(n_turns, epsilon, mu)
    return x, px, y, py


def henon_inverse(init_list, n_turns, start_point, epsilon, mu):
    if not((mu is None) or (mu == 0.0)):
        raise NotImplementedError
    engine = hm.partial_track.generate_instance(
        init_list[0],
        init_list[1],
        init_list[2],
        init_list[3],
    )
    engine.total_iters = start_point
    x, px, y, py, _ = engine.inverse_compute(n_turns, epsilon)
    return x, px, y, py


def henon_tracking(init_list, n_turns, epsilon, mu):
    engine = hm.partial_track.generate_instance(
        init_list[0],
        init_list[1],
        init_list[2],
        init_list[3],
    )
    x, px, y, py, steps = engine.compute(n_turns, epsilon, mu)
    return steps


def henon_inverse_error(init_list, n_turns, epsilon, mu):
    if not((mu is None) or (mu == 0.0)):
        raise NotImplementedError
    engine = hm.partial_track.generate_instance(
        init_list[0],
        init_list[1],
        init_list[2],
        init_list[3],
    )
    engine.compute(n_turns, epsilon)
    x, px, y, py, _ = engine.inverse_compute(n_turns, epsilon)
    data = np.sqrt(
        np.power(init_list[0] - x, 2)
        + np.power(init_list[1] - px, 2)
        + np.power(init_list[2] - y, 2)
        + np.power(init_list[3] - py, 2)
    )
    return data


def henon_lyapunov(init_list, n_turns, epsilon, mu):
    initial_err = 1e-10
    x0 = np.concatenate((init_list[0], init_list[0]))
    px0 = np.concatenate((init_list[1], init_list[1]))
    y0 = np.concatenate((init_list[2], init_list[2]))
    py0 = np.concatenate((init_list[3], init_list[3]))

    alpha = np.random.uniform(0, np.pi/2, len(init_list[0]))
    th1 = np.random.uniform(0, np.pi*2, len(init_list[0]))
    th2 = np.random.uniform(0, np.pi*2, len(init_list[0]))

    x0[len(x0)//2:] += initial_err * np.cos(alpha) * np.cos(th1)
    px0[len(x0)//2:] += initial_err * np.cos(alpha) * np.sin(th1)
    y0[len(x0)//2:] += initial_err * np.sin(alpha) * np.cos(th2)
    py0[len(x0)//2:] += initial_err * np.sin(alpha) * np.sin(th2)

    engine = hm.partial_track.generate_instance(x0, px0, y0, py0)
    x, px, y, py, _ = engine.compute(n_turns, epsilon, mu)
    data = np.sqrt(
        + np.power(x[:len(x)//2] - x[len(x)//2:], 2)
        + np.power(px[:len(x)//2] - px[len(x)//2:], 2)
        + np.power(y[:len(x)//2] - y[len(x)//2:], 2)
        + np.power(py[:len(x)//2] - py[len(x)//2:], 2)
    ) / initial_err
    return data


henon_wrap = function_wrap("Henon Map", henon_call, [
                           "epsilon", "mu"], [0.0, 0.0])
henon_wrap.set_inverse(henon_inverse)
henon_wrap.set_indicator(henon_tracking, "tracking")
henon_wrap.set_indicator(henon_lyapunov, "lyapunov")
henon_wrap.set_indicator(henon_inverse_error, "inversion error")

henon_wrap_2d = function_multi(henon_wrap)
