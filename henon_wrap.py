"""This module contains an example function wrapping of my Henon map engine,
in order to test and visualize the graphical interface made with plotly+dash
"""
from numba import njit, prange
import numba
import numpy as np
# Personal package henon_map
import henon_map as hm
from numpy.linalg.linalg import LinAlgError
# Local module function_wrap.py
from function_wrap import function_wrap, function_multi


def henon_call(init_list, n_turns, start_point, epsilon, mu, **kwargs):
    engine = hm.partial_track.generate_instance(
        init_list[0],
        init_list[1],
        init_list[2],
        init_list[3],
    )
    engine.total_iters = start_point
    x, px, y, py, _ = engine.compute(n_turns, epsilon, mu)
    return x, px, y, py


def henon_inverse(init_list, n_turns, start_point, epsilon, mu, **kwargs):
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


def henon_tracking(init_list, n_turns, epsilon, mu, **kwargs):
    engine = hm.partial_track.generate_instance(
        init_list[0],
        init_list[1],
        init_list[2],
        init_list[3],
    )
    x, px, y, py, steps = engine.compute(n_turns, epsilon, mu)
    return steps


def henon_inverse_error(init_list, n_turns, epsilon, mu, **kwargs):
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


def henon_inverse_error_with_forward_kick(init_list, n_turns, epsilon, mu, **kwargs):
    if not((mu is None) or (mu == 0.0)):
        raise NotImplementedError

    if "kick_magnitude" not in kwargs:
        kick_magnitude = 1e-14
    else:
        kick_magnitude = kwargs["kick_magnitude"]
    
    engine = hm.partial_track.generate_instance(
        init_list[0],
        init_list[1],
        init_list[2],
        init_list[3],
    )
    for i in range(n_turns):
        engine.compute(1, epsilon)
        alpha = np.random.uniform(0, np.pi/2, len(init_list[0]))
        th1 = np.random.uniform(0, np.pi*2, len(init_list[0]))
        th2 = np.random.uniform(0, np.pi*2, len(init_list[0]))
        engine.add_kick(
            kick_magnitude * np.cos(alpha) * np.cos(th1),
            kick_magnitude * np.cos(alpha) * np.sin(th1),
            kick_magnitude * np.sin(alpha) * np.cos(th2),
            kick_magnitude * np.sin(alpha) * np.sin(th2)
        )
    x, px, y, py, _ = engine.inverse_compute(n_turns, epsilon)
    data = np.sqrt(
        np.power(init_list[0] - x, 2)
        + np.power(init_list[1] - px, 2)
        + np.power(init_list[2] - y, 2)
        + np.power(init_list[3] - py, 2)
    )
    return data


def henon_lyapunov(init_list, n_turns, epsilon, mu, **kwargs):
    if "initial_err" not in kwargs:
        initial_err = 1e-12
    else:
        initial_err = kwargs["initial_err"]
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


def henon_lyapunov_multi(init_list, n_turns, epsilon, mu, **kwargs):
    if "initial_err" not in kwargs:
        initial_err = 1e-12
    else:
        initial_err = kwargs["initial_err"]
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
    x, px, y, py, _ = engine.compute(n_turns, epsilon, mu)
    data = np.log10((
        np.sqrt(
            + np.power(x[:len(x)//5] - x[(len(x)//5)*1:(len(x)//5)*2], 2)
            + np.power(px[:len(x)//5] - px[(len(x)//5)*1:(len(x)//5)*2], 2)
            + np.power(y[:len(x)//5] - y[(len(x)//5)*1:(len(x)//5)*2], 2)
            + np.power(py[:len(x)//5] - py[(len(x)//5)*1:(len(x)//5)*2], 2)
        ) +
        np.sqrt(
            + np.power(x[:len(x)//5] - x[(len(x)//5)*2:(len(x)//5)*3], 2)
            + np.power(px[:len(x)//5] - px[(len(x)//5)*2:(len(x)//5)*3], 2)
            + np.power(y[:len(x)//5] - y[(len(x)//5)*2:(len(x)//5)*3], 2)
            + np.power(py[:len(x)//5] - py[(len(x)//5)*2:(len(x)//5)*3], 2)
        ) +
        np.sqrt(
            + np.power(x[:len(x)//5] - x[(len(x)//5)*3:(len(x)//5)*4], 2)
            + np.power(px[:len(x)//5] - px[(len(x)//5)*3:(len(x)//5)*4], 2)
            + np.power(y[:len(x)//5] - y[(len(x)//5)*3:(len(x)//5)*4], 2)
            + np.power(py[:len(x)//5] - py[(len(x)//5)*3:(len(x)//5)*4], 2)
        ) +
        np.sqrt(
            + np.power(x[:len(x)//5] - x[(len(x)//5)*4:], 2)
            + np.power(px[:len(x)//5] - px[(len(x)//5)*4:], 2)
            + np.power(y[:len(x)//5] - y[(len(x)//5)*4:], 2)
            + np.power(py[:len(x)//5] - py[(len(x)//5)*4:], 2)
        )) / (initial_err * 4)) / n_turns
    return data


def henon_lyapunov_square_error(init_list, n_turns, epsilon, mu, **kwargs):
    if "initial_err" not in kwargs:
        initial_err = 1e-12
    else:
        initial_err = kwargs["initial_err"]
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
    x, px, y, py, _ = engine.compute(n_turns, epsilon, mu)

    t11 = x[(len(x0)//5)*1:(len(x0)//5)*2] - x[:(len(x0)//5)*1]
    t12 = px[(len(x0)//5)*1:(len(x0)//5)*2] - px[:(len(x0)//5)*1]
    t13 = y[(len(x0)//5)*1:(len(x0)//5)*2] - y[:(len(x0)//5)*1]
    t14 = py[(len(x0)//5)*1:(len(x0)//5)*2] - py[:(len(x0)//5)*1]

    t21 = x[(len(x0)//5)*2:(len(x0)//5)*3] - x[:(len(x0)//5)*1]
    t22 = px[(len(x0)//5)*2:(len(x0)//5)*3] - px[:(len(x0)//5)*1]
    t23 = y[(len(x0)//5)*2:(len(x0)//5)*3] - y[:(len(x0)//5)*1]
    t24 = py[(len(x0)//5)*2:(len(x0)//5)*3] - py[:(len(x0)//5)*1]

    t31 = x[(len(x0)//5)*3:(len(x0)//5)*4] - x[:(len(x0)//5)*1]
    t32 = px[(len(x0)//5)*3:(len(x0)//5)*4] - px[:(len(x0)//5)*1]
    t33 = y[(len(x0)//5)*3:(len(x0)//5)*4] - y[:(len(x0)//5)*1]
    t34 = py[(len(x0)//5)*3:(len(x0)//5)*4] - py[:(len(x0)//5)*1]

    t41 = x[(len(x0)//5)*4:(len(x0)//5)*5] - x[:(len(x0)//5)*1]
    t42 = px[(len(x0)//5)*4:(len(x0)//5)*5] - px[:(len(x0)//5)*1]
    t43 = y[(len(x0)//5)*4:(len(x0)//5)*5] - y[:(len(x0)//5)*1]
    t44 = py[(len(x0)//5)*4:(len(x0)//5)*5] - py[:(len(x0)//5)*1]

    tm = np.transpose(np.array([
        [t11, t12, t13, t14],
        [t21, t22, t23, t24],
        [t31, t32, t33, t34],
        [t41, t42, t43, t44]
    ]), axes=(2, 0, 1))
    tmt = np.transpose(tm, axes=(0, 2, 1))

    return np.trace(np.matmul(tmt, tm), axis1=1, axis2=2)


def henon_invariant_lyapunov(init_list, n_turns, epsilon, mu, **kwargs):
    return np.sqrt(henon_lyapunov_square_error(init_list, n_turns, epsilon, mu, **kwargs))


def faddeev_leverrier(m, grade):
    assert grade > 0
    step = 1
    B = m.copy()
    p = np.trace(B, axis1=1, axis2=2)
    while step != grade:
        step += 1
        B = np.matmul(m, B - np.array([np.identity(B.shape[-1]) for i in range(B.shape[0])]) * p[:,None,None])
        p = np.trace(B, axis1=1, axis2=2) * (1 / step)
    return p * ((-1) ** (grade + 1))


def henon_invariant_lyapunov_spec_grade(init_list, n_turns, epsilon, mu, **kwargs):
    if "initial_err" not in kwargs:
        initial_err = 1e-12
    else:
        initial_err = kwargs["initial_err"]
    if "grade" not in kwargs:
        grade = 1
    else:
        grade = kwargs["grade"]
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
    x, px, y, py, _ = engine.compute(n_turns, epsilon, mu)

    t11 = x[(len(x0)//5)*1:(len(x0)//5)*2] - x[:(len(x0)//5)*1]
    t12 = px[(len(x0)//5)*1:(len(x0)//5)*2] - px[:(len(x0)//5)*1]
    t13 = y[(len(x0)//5)*1:(len(x0)//5)*2] - y[:(len(x0)//5)*1]
    t14 = py[(len(x0)//5)*1:(len(x0)//5)*2] - py[:(len(x0)//5)*1]

    t21 = x[(len(x0)//5)*2:(len(x0)//5)*3] - x[:(len(x0)//5)*1]
    t22 = px[(len(x0)//5)*2:(len(x0)//5)*3] - px[:(len(x0)//5)*1]
    t23 = y[(len(x0)//5)*2:(len(x0)//5)*3] - y[:(len(x0)//5)*1]
    t24 = py[(len(x0)//5)*2:(len(x0)//5)*3] - py[:(len(x0)//5)*1]

    t31 = x[(len(x0)//5)*3:(len(x0)//5)*4] - x[:(len(x0)//5)*1]
    t32 = px[(len(x0)//5)*3:(len(x0)//5)*4] - px[:(len(x0)//5)*1]
    t33 = y[(len(x0)//5)*3:(len(x0)//5)*4] - y[:(len(x0)//5)*1]
    t34 = py[(len(x0)//5)*3:(len(x0)//5)*4] - py[:(len(x0)//5)*1]

    t41 = x[(len(x0)//5)*4:(len(x0)//5)*5] - x[:(len(x0)//5)*1]
    t42 = px[(len(x0)//5)*4:(len(x0)//5)*5] - px[:(len(x0)//5)*1]
    t43 = y[(len(x0)//5)*4:(len(x0)//5)*5] - y[:(len(x0)//5)*1]
    t44 = py[(len(x0)//5)*4:(len(x0)//5)*5] - py[:(len(x0)//5)*1]

    tm = np.transpose(np.array([
        [t11, t12, t13, t14],
        [t21, t22, t23, t24],
        [t31, t32, t33, t34],
        [t41, t42, t43, t44]
    ]), axes=(2, 0, 1))
    tmt = np.transpose(tm, axes=(0, 2, 1))

    return faddeev_leverrier(np.matmul(tmt,tm), grade)


def henon_megno_single(init_list, n_turns, epsilon, mu, **kwargs):
    if "initial_err" not in kwargs:
        initial_err = 1e-12
    else:
        initial_err = kwargs["initial_err"]
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
    sum = np.ones_like(init_list[0])
    x1 = x0
    px1 = px0
    y1 = y0
    py1 = py0
    for i in range(0, n_turns):
        x2, px2, y2, py2, _ = engine.compute(1, epsilon, mu)
        sum += (
            (i + 1) * np.log(
                (
                    + np.power(x2[:len(x2)//2] - x2[len(x2)//2:], 2)
                    + np.power(px2[:len(x2)//2] - px2[len(x2)//2:], 2)
                    + np.power(y2[:len(x2)//2] - y2[len(x2)//2:], 2)
                    + np.power(py2[:len(x2)//2] - py2[len(x2)//2:], 2)
                ) / (
                    + np.power(x1[:len(x1)//2] - x1[len(x1)//2:], 2)
                    + np.power(px1[:len(x1)//2] - px1[len(x1)//2:], 2)
                    + np.power(y1[:len(x1)//2] - y1[len(x1)//2:], 2)
                    + np.power(py1[:len(x1)//2] - py1[len(x1)//2:], 2)
                ) 
            )
        )
        x1 = x2.copy()
        px1 = px2.copy()
        y1 = y2.copy()
        py1 = py2.copy()
    return sum / n_turns


def prev_2_pow(x):
    return 1 if x == 0 else 2**((x - 1).bit_length() - 1)   


@njit(parallel=True)
def select_indices(l, c, r, data):
    p1 = np.empty(len(l), dtype=numba.int32)
    p2 = np.empty(len(l), dtype=numba.int32)
    vp1 = np.empty(len(l))
    vp2 = np.empty(len(l))
    for i in prange(data.shape[1]):
        if data[l[i], i] > data[r[i], i]:
            p1[i] = l[i]
            p2[i] = c[i]
            vp1[i] = data[p1[i], i]
            vp2[i] = data[p2[i], i]
        else:
            p1[i] = c[i]
            p2[i] = r[i]
            vp1[i] = data[p1[i], i]
            vp2[i] = data[p2[i], i]
    return p1, p2, vp1, vp2


@njit()
def interpolation(data, indices):
    values = np.empty(len(indices))
    for i in range(len(indices)):
        if np.any(np.isnan(data[:, i])):
            values[i] = np.nan
        elif indices[i] == 0:
            values[i] = np.nan
        else:
            #if indices[i] == 0:
            #    indices[i] += 1
            if indices[i] == len(data) - 1:
                indices[i] -= 1
            #print("-------------")
            #print(indices[i]-1,indices[i],indices[i]+1)
            cf1 = np.absolute(data[indices[i] - 1][i])
            cf2 = np.absolute(data[indices[i]][i])
            cf3 = np.absolute(data[indices[i] + 1][i])
            #print(cf1,cf2,cf3)
            if cf3 > cf1:
                p1 = cf2
                p2 = cf3
                nn = indices[i]
            else:
                p1 = cf1
                p2 = cf2
                nn = indices[i] - 1            
            p3 = np.cos(2 * np.pi / len(data))
            values[i] = (
                (nn / len(data)) + (1/np.pi) * np.arcsin(
                    np.sin(2*np.pi/len(data)) * 
                    ((-(p1+p2*p3)*(p1-p2) + p2*np.sqrt(p3**2*(p1+p2)**2 - 2*p1*p2*(2*p3**2-p3-1)))/(p1**2 + p2**2 + 2*p1*p2*p3))
                )
            )
            #print(nn)
            #print(values[i] * len(data))
            #print("-------------")
    return values


def henon_tune_map_x(init_list, n_turns, epsilon, mu, **kwargs):
    n_turns = prev_2_pow(n_turns)
    engine = hm.partial_track.generate_instance(
        init_list[0],
        init_list[1],
        init_list[2],
        init_list[3],
    )
    x, px, _, _ = engine.compute(n_turns, epsilon, mu, full_track=True)
    signal = x + 1j * px
    fft = np.absolute(np.fft.fft(
        signal * np.hanning(signal.shape[0])[:, None], axis=0))
    data1 = np.argmax(fft, axis=0)  # / n_turns
    data1 = interpolation(fft, data1)
    return data1


def henon_tune_map_y(init_list, n_turns, epsilon, mu, **kwargs):
    n_turns = prev_2_pow(n_turns)
    engine = hm.partial_track.generate_instance(
        init_list[0],
        init_list[1],
        init_list[2],
        init_list[3],
    )
    _, _, y, py = engine.compute(n_turns, epsilon, mu, full_track=True)
    signal = y + 1j * py
    fft = np.absolute(np.fft.fft(
        signal * np.hanning(signal.shape[0])[:, None], axis=0))
    data1 = np.argmax(fft, axis=0)  # / n_turns
    data1 = interpolation(fft, data1)
    return data1


def henon_frequency_map_x_only(init_list, n_turns, epsilon, mu, **kwargs):
    n_turns = prev_2_pow(n_turns)
    engine = hm.partial_track.generate_instance(
        init_list[0],
        init_list[1],
        init_list[2],
        init_list[3],
    )
    x, px, _, _ = engine.compute(n_turns, epsilon, mu, full_track=True)
    signal = x + 1j * px
    fft = np.absolute(np.fft.fft(
        signal * np.hanning(signal.shape[0])[:, None], axis=0))
    data1 = np.argmax(fft, axis=0)  # / n_turns
    data1 = interpolation(fft, data1)
    x, px, _, _ = engine.compute(n_turns, epsilon, mu, full_track=True)
    signal = x + 1j * px
    fft = np.absolute(np.fft.fft(
        signal * np.hanning(signal.shape[0])[:, None], axis=0))
    data2 = np.argmax(fft, axis=0)  # / n_turns
    data2 = interpolation(fft, data2)
    return np.absolute(data1 - data2)


def henon_frequency_map_y_only(init_list, n_turns, epsilon, mu, **kwargs):
    n_turns = prev_2_pow(n_turns)
    engine = hm.partial_track.generate_instance(
        init_list[0],
        init_list[1],
        init_list[2],
        init_list[3],
    )
    _, _, y, py = engine.compute(n_turns, epsilon, mu, full_track=True)
    signal = y + 1j * py
    fft = np.absolute(np.fft.fft(
        signal * np.hanning(signal.shape[0])[:, None], axis=0))
    data1 = np.argmax(fft, axis=0)  # / n_turns
    data1 = interpolation(fft, data1)
    _, _, y, py = engine.compute(n_turns, epsilon, mu, full_track=True)
    signal = y + 1j * py
    fft = np.absolute(np.fft.fft(
        signal * np.hanning(signal.shape[0])[:, None], axis=0))
    data2 = np.argmax(fft, axis=0)  # / n_turns
    data2 = interpolation(fft, data2)
    return np.absolute(data1 - data2)


def henon_frequency_map(init_list, n_turns, epsilon, mu, **kwargs):
    n_turns = prev_2_pow(n_turns)
    engine = hm.partial_track.generate_instance(
        init_list[0],
        init_list[1],
        init_list[2],
        init_list[3],
    )
    x, px, y, py = engine.compute(n_turns, epsilon, mu, full_track=True)
    signal = x + 1j * px
    fft = np.absolute(np.fft.fft(
        signal * np.hanning(signal.shape[0])[:, None], axis=0))
    data1 = np.argmax(fft, axis=0)# / n_turns
    data1 = interpolation(fft, data1)
    signal2 = y + 1j * py
    fft2 = np.absolute(np.fft.fft(
        signal2 * np.hanning(signal2.shape[0])[:, None], axis=0))
    data2 = np.argmax(fft2, axis=0)# / n_turns
    data2 = interpolation(fft2, data2)
    x, px, y, py = engine.compute(n_turns, epsilon, mu, full_track=True)
    signal = x + 1j * px
    fft = np.absolute(np.fft.fft(
        signal * np.hanning(signal.shape[0])[:, None], axis=0))
    data3 = np.argmax(fft, axis=0)# / n_turns
    data3 = interpolation(fft, data3)
    signal2 = y + 1j * py
    fft2 = np.absolute(np.fft.fft(
        signal2 * np.hanning(signal2.shape[0])[:, None], axis=0))
    data4 = np.argmax(fft2, axis=0)# / n_turns
    data4 = interpolation(fft2, data4)
    return np.sqrt(np.power(data3 - data1, 2) + np.power(data4 - data2, 2))


def henon_resonance(init_list, n_turns, epsilon, mu, **kwargs):
    if "min_res_order" not in kwargs:
        min_res_order = 3
    else:
        min_res_order = kwargs["min_res_order"]
    if "max_res_order" not in kwargs:
        max_res_order = 5
    else:
        max_res_order = kwargs["max_res_order"]

    if "tolerance" not in kwargs:
        tolerance = 0.001
    else:
        tolerance = kwargs["tolerance"]

    n_turns = prev_2_pow(n_turns)
    engine = hm.partial_track.generate_instance(
        init_list[0],
        init_list[1],
        init_list[2],
        init_list[3],
    )
    x, px, y, py = engine.compute(n_turns, epsilon, mu, full_track=True)
    signal = x + 1j * px
    fft = np.absolute(np.fft.fft(
        signal * np.hanning(signal.shape[0])[:, None], axis=0))
    data1 = np.argmax(fft, axis=0)#   / n_turns
    #data1[data1==0] = np.nan
    data1 = interpolation(fft, data1)
    signal2 = y + 1j * py
    fft2 = np.absolute(np.fft.fft(
        signal2 * np.hanning(signal2.shape[0])[:, None], axis=0))
    data2 = np.argmax(fft2, axis=0)#   / n_turns
    #data2[data2==0] = np.nan
    data2 = interpolation(fft2, data2)

    tune = np.zeros_like(data1)
    for k in range(min_res_order, max_res_order + 1):
        for i in range(k + 1): # nx = 0, ny = 4; nx = 1, ny = 3
            tx = i
            ty = k - i
            mask1 = np.absolute(
                np.modf(data1 * tx + data2 * ty)[0]) <= tolerance
            mask2 = np.absolute(
                np.modf(data1 * tx - data2 * ty)[0]) <= tolerance
            mask3 = np.absolute(
                np.modf(-data1 * tx + data2 * ty)[0]) <= tolerance
            mask4 = np.absolute(
                np.modf(-data1 * tx - data2 * ty)[0]) <= tolerance
            tune[np.logical_or(np.logical_or(mask1, mask2),
                               np.logical_or(mask3, mask4))] = k
    tune[tune == 0] = np.nan
    return tune

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

    bool_mask = np.all(np.logical_not(np.isnan(matrix)), axis=(1, 2))
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


def gali3(x, px, y, py):
    # build displacement vectors
    v1x = x[len(x)//4:(len(x)//4)*2] - x[:len(x)//4]
    v1px = px[len(x)//4:(len(x)//4)*2] - px[:len(x)//4]
    v1y = y[len(x)//4:(len(x)//4)*2] - y[:len(x)//4]
    v1py = py[len(x)//4:(len(x)//4)*2] - py[:len(x)//4]

    v2x = x[(len(x)//4)*2:(len(x)//4)*3] - x[:len(x)//4]
    v2px = px[(len(x)//4)*2:(len(x)//4)*3] - px[:len(x)//4]
    v2y = y[(len(x)//4)*2:(len(x)//4)*3] - y[:len(x)//4]
    v2py = py[(len(x)//4)*2:(len(x)//4)*3] - py[:len(x)//4]
    
    v3x = x[(len(x)//4)*3:] - x[:len(x)//4]
    v3px = px[(len(x)//4)*3:] - px[:len(x)//4]
    v3y = y[(len(x)//4)*3:] - y[:len(x)//4]
    v3py = py[(len(x)//4)*3:] - py[:len(x)//4]
    # compute norm
    norm1 = np.sqrt(np.power(v1x, 2) + np.power(v1px, 2) +
                    np.power(v1y, 2) + np.power(v1py, 2))
    norm2 = np.sqrt(np.power(v2x, 2) + np.power(v2px, 2) +
                    np.power(v2y, 2) + np.power(v2py, 2))
    norm3 = np.sqrt(np.power(v3x, 2) + np.power(v3px, 2) +
                    np.power(v3y, 2) + np.power(v3py, 2))
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
    # Compose matrix
    matrix = np.array(
        [[v1x, v2x, v3x],
         [v1px, v2px, v3px],
         [v1y, v2y, v3y],
         [v1py, v2py, v3py]]
    )
    matrix = np.swapaxes(matrix, 1, 2)
    matrix = np.swapaxes(matrix, 0, 1)

    bool_mask = np.all(np.logical_not(np.isnan(matrix)), axis=(1, 2))
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
    result = np.zeros((len(x)//4))
    result[np.logical_not(bool_mask)] = np.nan
    result[bool_mask] = np.prod(s, axis=-1)
    return result


def gali2(x, px, y, py):
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
    norm1 = np.sqrt(np.power(v1x, 2) + np.power(v1px, 2) +
                    np.power(v1y, 2) + np.power(v1py, 2))
    norm2 = np.sqrt(np.power(v2x, 2) + np.power(v2px, 2) +
                    np.power(v2y, 2) + np.power(v2py, 2))
    # normalize
    v1x /= norm1
    v1px /= norm1
    v1y /= norm1
    v1py /= norm1
    v2x /= norm2
    v2px /= norm2
    v2y /= norm2
    v2py /= norm2
    # Compose matrix
    matrix = np.array(
        [[v1x, v2x],
         [v1px, v2px],
         [v1y, v2y],
         [v1py, v2py]]
    )
    matrix = np.swapaxes(matrix, 1, 2)
    matrix = np.swapaxes(matrix, 0, 1)

    bool_mask = np.all(np.logical_not(np.isnan(matrix)), axis=(1, 2))
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
    result = np.zeros((len(x)//3))
    result[np.logical_not(bool_mask)] = np.nan
    result[bool_mask] = np.prod(s, axis=-1)
    return result
    

def henon_sali(init_list, n_turns, epsilon, mu, **kwargs):
    if "inital_err" in kwargs:
        initial_err = 1e-12
    else:
        initial_err = kwargs["initial_err"]
    if "tau" in kwargs:
        tau = 5
    else:
        tau = kwargs["tau"]
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


def henon_gali(init_list, n_turns, epsilon, mu, **kwargs):
    if "inital_err" in kwargs:
        initial_err = 1e-12
    else:
        initial_err = kwargs["initial_err"]
    if "tau" in kwargs:
        tau = 5
    else:
        tau = kwargs["tau"]
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


def henon_gali_3(init_list, n_turns, epsilon, mu, **kwargs):
    if "inital_err" in kwargs:
        initial_err = 1e-12
    else:
        initial_err = kwargs["initial_err"]
    if "tau" in kwargs:
        tau = 5
    else:
        tau = kwargs["tau"]
    n_iters = n_turns // tau
    x0 = np.concatenate(
        (init_list[0], init_list[0], init_list[0], init_list[0]))
    px0 = np.concatenate(
        (init_list[1], init_list[1], init_list[1], init_list[1]))
    y0 = np.concatenate(
        (init_list[2], init_list[2], init_list[2], init_list[2]))
    py0 = np.concatenate(
        (init_list[3], init_list[3], init_list[3], init_list[3]))

    x0[(len(x0)//4)*1:(len(x0)//4)*2] += initial_err
    px0[(len(x0)//4)*2:(len(x0)//4)*3] += initial_err
    y0[(len(x0)//4)*3:] += initial_err

    engine = hm.partial_track.generate_instance(x0, px0, y0, py0)
    g = gali3(x0, px0, y0, py0)
    for i in range(n_iters):
        x, px, y, py, _ = engine.compute(tau, epsilon, mu)
        g = np.amin([g, gali3(x, px, y, py)], axis=0)
    return g


def henon_gali_2(init_list, n_turns, epsilon, mu, **kwargs):
    if "inital_err" in kwargs:
        initial_err = 1e-12
    else:
        initial_err = kwargs["initial_err"]
    if "tau" in kwargs:
        tau = 5
    else:
        tau = kwargs["tau"]
    n_iters = n_turns // tau
    x0 = np.concatenate(
        (init_list[0], init_list[0], init_list[0]))
    px0 = np.concatenate(
        (init_list[1], init_list[1], init_list[1]))
    y0 = np.concatenate(
        (init_list[2], init_list[2], init_list[2]))
    py0 = np.concatenate(
        (init_list[3], init_list[3], init_list[3]))

    x0[(len(x0)//3)*1:(len(x0)//3)*2] += initial_err
    y0[(len(x0)//3)*2:] += initial_err

    engine = hm.partial_track.generate_instance(x0, px0, y0, py0)
    g = gali2(x0, px0, y0, py0)
    for i in range(n_iters):
        x, px, y, py, _ = engine.compute(tau, epsilon, mu)
        g = np.amin([g, gali2(x, px, y, py)], axis=0)
    return g


henon_wrap = function_wrap("Henon Map", henon_call, [
                           "epsilon", "mu"], [0.0, 0.0])
henon_wrap.set_inverse(henon_inverse)
henon_wrap.set_indicator(henon_tracking, "tracking")
henon_wrap.set_indicator(
    henon_lyapunov, 
    "lyapunov",
    ["initial_err"],
    [1e-14])
henon_wrap.set_indicator(
    henon_lyapunov_multi, 
    "lyapunov multi", 
    ["initial_err"], 
    [1e-14])
henon_wrap.set_indicator(
    henon_lyapunov_square_error,
    "lyapunov square error",
    ["initial_err"],
    [1e-14])
henon_wrap.set_indicator(
    henon_invariant_lyapunov,
    "lyapunov invariant",
    ["initial_err"],
    [1e-14])
henon_wrap.set_indicator(
    henon_invariant_lyapunov_spec_grade,
    "lyapunov invariant grade",
    ["initial_err", "grade"],
    [1e-14, 1])
henon_wrap.set_indicator(
    henon_megno_single,
    "megno",
    ["initial_err"],
    [1e-14])
henon_wrap.set_indicator(henon_inverse_error, "inversion error")
henon_wrap.set_indicator(
    henon_inverse_error_with_forward_kick,
    "inversion error with kick",
    ["kick_magnitude"],
    [1e-14])
henon_wrap.set_indicator(henon_frequency_map, "Frequency Map")
henon_wrap.set_indicator(henon_frequency_map_x_only, "Frequency Map X")
henon_wrap.set_indicator(henon_frequency_map_y_only, "Frequency Map Y")
henon_wrap.set_indicator(
    henon_resonance,
    "resonance map",
    ["min_res_order", "max_res_order", "tolerance"],
    [3, 5, 0.001])
henon_wrap.set_indicator(henon_tune_map_x, "tune x")
henon_wrap.set_indicator(henon_tune_map_y, "tune y")
henon_wrap.set_indicator(
    henon_sali, 
    "SALI", 
    ["initial_err", "tau"], 
    [1e-14, 5])
henon_wrap.set_indicator(
    henon_gali, 
    "GALI", 
    ["initial_err", "tau"], 
    [1e-14, 5])
henon_wrap.set_indicator(
    henon_gali_2, 
    "GALI 2 dims only", 
    ["initial_err", "tau"], 
    [1e-14, 5])
henon_wrap.set_indicator(
    henon_gali_3, 
    "GALI 3 dims only", 
    ["initial_err", "tau"], 
    [1e-14, 5])

henon_wrap_2d = function_multi(henon_wrap)
