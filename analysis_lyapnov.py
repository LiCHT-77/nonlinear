from algorism import embedding, lyap_j
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def traverse_lyap(series: np.ndarray, params: dict):
    dynamic = None
    for key, param in params.items():
        if type(param) is list:
            dynamic = key

    if dynamic is None:
        return

    params = [{**params, **{dynamic: val}} for val in params[dynamic]]
    lyaps = []
    for param in params:
        embedded = embedding(series, **param)
        lyapnov_component = lyap_j(
            embedded,
            **param
        )
        lyaps.append(lyapnov_component)

    return np.array([[param[dynamic] for param in params], lyaps])


def lyap(series: np.ndarray, dt: float):
    results = []
    M = list(range(2, 11))
    col = 3
    fig = plt.figure(figsize=(12, 2 * (len(M) // col)))
    fig.subplots_adjust(hspace=0.6, wspace=0.4)
    for i, m in enumerate(M):
        exponents = lyap_converge(series, dt, m)
        results.append(exponents)
        ax = fig.add_subplot(len(M) // col + int(len(M) %
                             col != 0), col, i + 1)
        lyap_plot(exponents[0], exponents[1], f"m={m}", ax)

    return results, fig


def traverse(series, dim: int, window: int = 90):
    Ns = [i for i in range(300, len(series), 10)]
    param = {
        "tau": window // dim,
        "dim": dim,
        "s": 1,
        "M": dim + 1,
        "sampling_rate": 1
    }

    lyaps = []
    for n in Ns:
        embedded = embedding(series[:n], **param)
        lyapnov_component = lyap_j(
            embedded,
            **param
        )
        lyaps.append(lyapnov_component)

    return Ns, np.array(lyaps)


def lyap_converge(series: np.ndarray, dt: float, m: int, window: int, M: int):
    Ns = [i for i in range(window + 100, len(series), 100)]
    param = {
        "tau": window // m,
        "dim": m,
        "s": dt,
        "M": M,
        "sampling_rate": dt
    }
    lyaps = []
    for n in Ns:
        embedded = embedding(series[:n], **param)
        lyapnov_component = lyap_j(
            embedded,
            **param
        )
        lyaps.append(lyapnov_component)

    return Ns, np.array(lyaps)


def lyap_plot(Ns: np.ndarray, lyaps: np.ndarray, title: str, ax=None):
    if ax is None:
        fig = plt.figure()
        fig.subplots_adjust(top=0.99)
        fig.tight_layout()
        ax = fig.add_subplot(111)

    for i in range(len(lyaps[0])):
        ax.plot(Ns, lyaps[:, i])
    ax.set_xlabel("N")
    ax.set_ylabel("Lyapnov exponent")
    ax.set_title(title, y=-0.6)
    ax.set_ylim([0, 0.5])

    return ax


if __name__ == "__main__":
    sampling_rate = 1
    path = "./sample/temperature.csv"
    df = pd.read_csv(path, index_col=0)
    # col = "那覇"
    col = "札幌"
    # col = "東京"
    series = df.interpolate(
        method='linear', limit_direction='forward', limit_area='inside')[col].values[:4000]

    m = 3
    window = 30 * m
    M = 4
    Ns, ly = lyap_converge(series, sampling_rate, m, window, M)
    lyap_plot(Ns, ly, "temperature")
    plt.show()
