"""This module contains an example function wrapping of my Henon map engine,
in order to test and visualize the graphical interface made with plotly+dash
"""
from numba import njit
import numpy as np
# Personal package henon_map
import henon_map as hm
from numpy.linalg.linalg import LinAlgError
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
    initial_err = 1e-12
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
    data = np.log10(np.sqrt(
        + np.power(x[:len(x)//2] - x[len(x)//2:], 2)
        + np.power(px[:len(x)//2] - px[len(x)//2:], 2)
        + np.power(y[:len(x)//2] - y[len(x)//2:], 2)
        + np.power(py[:len(x)//2] - py[len(x)//2:], 2)
    ) / initial_err) / n_turns
    return data


@njit(parallel=True)
def sali(x, px, y, py):
    # build displacement vectors
    v1x = x[len(x)//3:(len(x)//3)*2] - x[:len(x)//3]
    v1px = px[len(x)//3:(len(x)//3)*2] - px[:len(x)//3]
    v1y = y[len(x)//3:(len(x)//3)*2] - y[:len(x)//3]
    v1py = py[len(x)//3:(len(x)//3)*2] - py[:len(x)//3]
    v2x = x[(len(x)//3)*2:] - x[:len(x)//3]
    v2px = px[(len(x)//3)*2:] - px[:len(x)//3]
    v2y = y[(len(x)//3)*2:] - y[:len(x)//3]
    v2py = py[(len(x)//3)*2:] - py[:len(x)//3]
    # compute norm
    norm1 = np.sqrt(np.power(v1x, 2) + np.power(v1px, 2) + np.power(v1y, 2) + np.power(v1py, 2))
    norm2 = np.sqrt(np.power(v2x, 2) + np.power(v2px, 2) + np.power(v2y, 2) + np.power(v2py, 2))
    # normalize
    v1x /= norm1
    v1px /= norm1
    v1y /= norm1
    v1py /= norm1
    v2x /= norm2
    v2px /= norm2
    v2y /= norm2
    v2py /= norm2
    # return minimum
    return np.sqrt(np.minimum(
        np.power(v1x + v2x, 2) + np.power(v1px + v2px, 2) +
        np.power(v1y + v2y, 2) + np.power(v1py + v2py, 2),
        np.power(v1x - v2x, 2) + np.power(v1px - v2px, 2) +
        np.power(v1y - v2y, 2) + np.power(v1py - v2py, 2)
    ))


def gali(x, px, y, py):
    # build displacement vectors
    v1x = x[len(x)//5:(len(x)//5)*2] - x[:len(x)//5]
    v1px = px[len(x)//5:(len(x)//5)*2] - px[:len(x)//5]
    v1y = y[len(x)//5:(len(x)//5)*2] - y[:len(x)//5]
    v1py = py[len(x)//5:(len(x)//5)*2] - py[:len(x)//5]
    v2x = x[(len(x)//5)*2:(len(x)//5)*3] - x[:len(x)//5]
    v2px = px[(len(x)//5)*2:(len(x)//5)*3] - px[:len(x)//5]
    v2y = y[(len(x)//5)*2:(len(x)//5)*3] - y[:len(x)//5]
    v2py = py[(len(x)//5)*2:(len(x)//5)*3] - py[:len(x)//5]
    v3x = x[(len(x)//5)*3:(len(x)//5)*4] - x[:len(x)//5]
    v3px = px[(len(x)//5)*3:(len(x)//5)*4] - px[:len(x)//5]
    v3y = y[(len(x)//5)*3:(len(x)//5)*4] - y[:len(x)//5]
    v3py = py[(len(x)//5)*3:(len(x)//5)*4] - py[:len(x)//5]
    v4x = x[(len(x)//5)*4:] - x[:len(x)//5]
    v4px = px[(len(x)//5)*4:] - px[:len(x)//5]
    v4y = y[(len(x)//5)*4:] - y[:len(x)//5]
    v4py = py[(len(x)//5)*4:] - py[:len(x)//5]
    # compute norm
    norm1 = np.sqrt(np.power(v1x, 2) + np.power(v1px, 2) +
                    np.power(v1y, 2) + np.power(v1py, 2))
    norm2 = np.sqrt(np.power(v2x, 2) + np.power(v2px, 2) +
                    np.power(v2y, 2) + np.power(v2py, 2))
    norm3 = np.sqrt(np.power(v3x, 2) + np.power(v3px, 2) +
                    np.power(v3y, 2) + np.power(v3py, 2))
    norm4 = np.sqrt(np.power(v4x, 2) + np.power(v4px, 2) +
                    np.power(v4y, 2) + np.power(v4py, 2))
    # normalize
    v1x /= norm1
    v1px /= norm1
    v1y /= norm1
    v1py /= norm1
    v2x /= norm2
    v2px /= norm2
    v2y /= norm2
    v2py /= norm2
    v3x /= norm3
    v3px /= norm3
    v3y /= norm3
    v3py /= norm3
    v4x /= norm4
    v4px /= norm4
    v4y /= norm4
    v4py /= norm4
    # Compose matrix
    matrix = np.array(
        [[v1x, v2x, v3x, v4x],
         [v1px, v2px, v3px, v4px],
         [v1y, v2y, v3y, v4y],
         [v1py, v2py, v3py, v4py]]
    )
    matrix = np.swapaxes(matrix, 1, 2)
    matrix = np.swapaxes(matrix, 0, 1)

    bool_mask = np.all(np.logical_not(np.isnan(matrix)), axis=(1,2))
    # SVD
    try:
        _, s, _ = np.linalg.svd(matrix[bool_mask], full_matrices=True)
    except:
        print("Failed! looking for error point:")
        for i, m in enumerate(matrix[bool_mask]):
            print(i, m)
            print(bool_mask[i])
            print(np.all(np.logical_not(np.isnan(m))))
            try:
                np.linalg.svd(m, full_matrices=True)
            except:
                print("^^^ This is the broken one! ^^^")
                raise LinAlgError
    result = np.zeros((len(x)//5))
    result[np.logical_not(bool_mask)] = np.nan
    result[bool_mask] = np.prod(s, axis=-1)
    return result
    

def henon_sali(init_list, n_turns, epsilon, mu):
    initial_err = 1e-12
    tau = 2
    n_iters = n_turns // tau
    x0 = np.concatenate((init_list[0], init_list[0], init_list[0]))
    px0 = np.concatenate((init_list[1], init_list[1], init_list[1]))
    y0 = np.concatenate((init_list[2], init_list[2], init_list[2]))
    py0 = np.concatenate((init_list[3], init_list[3], init_list[3]))

    x0[len(x0)//3:(len(x0)//3)*2] += initial_err
    y0[(len(x0)//3)*2:] += initial_err

    engine = hm.partial_track.generate_instance(x0, px0, y0, py0)
    s = sali(x0, px0, y0, py0)
    for i in range(n_iters):
        x, px, y, py, _ = engine.compute(tau, epsilon, mu)
        s = np.amin([s, sali(x, px, y, py)], axis=0)
    return s


def henon_gali(init_list, n_turns, epsilon, mu):
    initial_err = 1e-12
    tau = 5
    n_iters = n_turns // tau
    x0 = np.concatenate(
        (init_list[0], init_list[0], init_list[0], init_list[0], init_list[0]))
    px0 = np.concatenate(
        (init_list[1], init_list[1], init_list[1], init_list[1], init_list[1]))
    y0 = np.concatenate(
        (init_list[2], init_list[2], init_list[2], init_list[2], init_list[2]))
    py0 = np.concatenate(
        (init_list[3], init_list[3], init_list[3], init_list[3], init_list[3]))

    x0[(len(x0)//5)*1:(len(x0)//5)*2] += initial_err
    px0[(len(x0)//5)*2:(len(x0)//5)*3] += initial_err
    y0[(len(x0)//5)*3:(len(x0)//5)*4] += initial_err
    py0[(len(x0)//5)*4:] += initial_err

    engine = hm.partial_track.generate_instance(x0, px0, y0, py0)
    g = gali(x0, px0, y0, py0)
    for i in range(n_iters):
        x, px, y, py, _ = engine.compute(tau, epsilon, mu)
        g = np.amin([g, gali(x, px, y, py)], axis=0)
    return g


henon_wrap = function_wrap("Henon Map", henon_call, [
                           "epsilon", "mu"], [0.0, 0.0])
henon_wrap.set_inverse(henon_inverse)
henon_wrap.set_indicator(henon_tracking, "tracking")
henon_wrap.set_indicator(henon_lyapunov, "lyapunov")
henon_wrap.set_indicator(henon_inverse_error, "inversion error")
henon_wrap.set_indicator(henon_sali, "SALI")
henon_wrap.set_indicator(henon_gali, "GALI")

henon_wrap_2d = function_multi(henon_wrap)
