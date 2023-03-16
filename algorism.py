from scipy.spatial.distance import pdist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def embedding(series, tau: int, dim: int, *args, **kwargs) -> np.ndarray:
    """
    series: 対象の時系列
    tau: 遅れ時間（何ステップシフトするか）
    dim: 埋め込み次元
    """
    _N = len(series)
    state_space = np.empty((_N - tau * (dim - 1), dim))

    for i in range(dim):
        state_space[:, i] = series[tau * i: _N - tau * (dim - 1 - i)]

    return state_space


def plot_state_space(state_space, azim=10) -> None:
    cols = state_space.shape[1]
    fig = plt.figure()
    if cols == 2:
        plt.plot(state_space[:, 0], state_space[:, 1])
    elif cols == 3:
        ax = fig.add_subplot(projection='3d')
        ax.plot(state_space[:, 0], state_space[:, 1], state_space[:, 2])
        ax.view_init(azim=azim)


def runge_kutta_3d(differential_equations, dt, N, init=(0., 0., 0.)):
    def gen_diff(differential_equations, x, y, z):
        k_1 = differential_equations[0](x, y, z) * dt
        l_1 = differential_equations[1](x, y, z) * dt
        m_1 = differential_equations[2](x, y, z) * dt

        k_2 = differential_equations[0](
            x + k_1 / 2, y + l_1 / 2, z + m_1 / 2) * dt
        l_2 = differential_equations[1](
            x + k_1 / 2, y + l_1 / 2, z + m_1 / 2) * dt
        m_2 = differential_equations[2](
            x + k_1 / 2, y + l_1 / 2, z + m_1 / 2) * dt

        k_3 = differential_equations[0](
            x + k_2 / 2, y + l_2 / 2, z + m_2 / 2) * dt
        l_3 = differential_equations[1](
            x + k_2 / 2, y + l_2 / 2, z + m_2 / 2) * dt
        m_3 = differential_equations[2](
            x + k_2 / 2, y + l_2 / 2, z + m_2 / 2) * dt

        k_4 = differential_equations[0](x + k_3, y + l_3, z + m_3) * dt
        l_4 = differential_equations[1](x + k_3, y + l_3, z + m_3) * dt
        m_4 = differential_equations[2](x + k_3, y + l_3, z + m_3) * dt

        k = (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6
        el = (l_1 + 2 * l_2 + 2 * l_3 + l_4) / 6
        m = (m_1 + 2 * m_2 + 2 * m_3 + m_4) / 6

        return np.array([k, el, m])
    xyz = np.empty((N + 1, 3))
    xyz[0] = init
    for i in range(N):
        diff = gen_diff(differential_equations,
                        xyz[i, 0], xyz[i, 1], xyz[i, 2])
        xyz[i + 1] = xyz[i] + diff

    return xyz


def gen_lorenz_arr(s=10, r=30, b=8 / 3, dt=0.01, steps=30000, init=(1., 1., 1.)):
    drop = int(1000 / dt)
    return runge_kutta_3d(differential_equations=[
        lambda x, y, z: s * (y - x),
        lambda x, y, z: r * x - y - x * z,
        lambda x, y, z: x * y - b * z
    ], dt=dt, N=steps + drop, init=init)[drop:]


def gen_logistic(a=4, steps=30000, init=0.1):
    _logistic_map = np.empty((steps + 1))
    _logistic_map[0] = init
    for i in range(steps):
        _logistic_map[i + 1] = a * _logistic_map[i] * (1 - _logistic_map[i])
    return _logistic_map


def gen_henon_map(a=1.4, b=0.3, steps=20000, init=(1., 1.)):
    def henon(x, y, a, b):
        _x = 1 - a * x ** 2 + y
        _y = b * x
        return _x, _y
    _henon_map = np.empty((steps + 1001, 2))

    _henon_map[0] = init
    for t in range(steps + 1000):
        _henon_map[t + 1] = henon(_henon_map[t][0], _henon_map[t][1], a, b)

    return _henon_map[1001:]


def _entry_pre(i, j, n):
    return (i - 1) + j * n - 2 * j - (j * (j + 1)) // 2


def _entry_after(i, j, n):
    return n * i + j - ((i + 2) * (i + 1)) // 2


def jacobian(embedded: np.ndarray, dist: np.ndarray, t: int, s: float, M: int):
    n = len(embedded)
    dist_index = np.array([_entry_pre(t, j, n) for j in range(
        t)] + [_entry_after(t, j, n) for j in range(t + 1, n)])[:- s - 1]

    neighbor_index = np.array(dist[dist_index].argsort()[:M], dtype='int32')

    y = embedded[neighbor_index] - embedded[t]
    z = embedded[neighbor_index + s] - embedded[t + s]

    w_sigma = 0
    c_sigma = 0
    for i in range(len(y)):
        w_sigma += np.tile(y[i], (len(y[i]), 1)).T @ np.diag(y[i])
        c_sigma += np.tile(z[i], (len(z[i]), 1)).T @ np.diag(y[i])
    w = w_sigma / M
    c = c_sigma / M

    return np.linalg.lstsq(w.T, c.T, rcond=None)[0].T


def lyap_j(embedded: np.ndarray, s: float, M: int, sampling_rate: float, *args, **kwargs):
    if s < sampling_rate:
        raise Exception("s < sampling_rate")

    s = int(s / sampling_rate)
    N = len(embedded) - s - 1

    dist = pdist(embedded.astype("float32"))
    J_t = jacobian(embedded, dist, 0, s, M)
    J_daggers = [J_t.T]
    Q_k, R_k = np.linalg.qr(J_t)
    S = np.log(np.abs(np.diag(R_k)))
    for t in range(1, N):
        J_t = jacobian(embedded, dist, t, s, M)
        J_daggers.append(J_t.T)
        Q_k, R_k = np.linalg.qr(J_t @ Q_k)
        S += np.log(np.abs(np.diag(R_k)))

    for _ in range(0, N):
        J_t_dagger = J_daggers.pop()
        Q_k, R_k = np.linalg.qr(J_t_dagger @ Q_k)
        S += np.log(np.abs(np.diag(R_k)))
    return S / (2 * N)


def fractal_gp(state_space, scale_range: tuple[float, float], split=100) -> pd.DataFrame:
    logr = np.exp(np.linspace(
        np.log(scale_range[0]), np.log(scale_range[1]), split))
    N = len(state_space)

    logCr = (np.expand_dims(pdist(state_space), axis=1) <=
             np.expand_dims(logr, axis=0)).sum(axis=0) / N**2
    logCr[logCr <= 0] = np.nan
    logr = np.log(logr)
    logCr = np.log(logCr)
    slope = np.abs(np.pad(np.diff(logCr), [1, 0], 'constant', constant_values=(
        np.nan, np.nan))) / np.pad(np.diff(logr), [1, 0], 'constant', constant_values=(np.nan, np.nan))
    slope[slope == 0] = np.nan
    return pd.DataFrame(np.concatenate([logr.reshape((-1, 1)), logCr.reshape((-1, 1)), slope.reshape((-1, 1))], 1), columns=['log r', 'log C^m(r)', 'Slope'])
