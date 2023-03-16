from algorism import embedding, gen_lorenz_arr
from lyapnov_spectram import lyap_j
import numpy as np


def resampling(series, rate, newrate):
    return series[[i for i in range(0, len(series), int(newrate / rate))]]


def traverse_lyap(series: np.ndarray, params: dict):
    dynamic = None
    for key, param in params.items():
        if type(param) is list:
            dynamic = key

    if dynamic is None:
        return

    params = [{**params, **{dynamic: val}} for val in params[dynamic]]
    for param in params:
        embedded = embedding(series, **param)
        lyap = lyap_j(
            embedded,
            **param
        )
        print(f"{dynamic}={param[dynamic]}:{lyap}")


if __name__ == "__main__":
    N = 1000
    sampling_rate = 0.01
    resampling_rate = 0.1
    steps = int(N / sampling_rate)
    lorenz_x = gen_lorenz_arr(
        dt=sampling_rate, steps=steps, init=(0.1, 0.1, 0.1))[:, 0]
    lorenz_x = resampling(lorenz_x, resampling_rate, sampling_rate)

    tau_params = {
        "tau": [i for i in range(10, 20)],
        "dim": 4,
        "s": 0.1,
        "M": 5,
        "sampling_rate": resampling_rate
    }
    traverse_lyap(lorenz_x, tau_params)

    dim_params = {
        "tau": 15,
        "dim": [i for i in range(2, 5)],
        "s": 0.1,
        "M": 5,
        "sampling_rate": resampling_rate
    }
    # traverse_lyap(lorenz_x, dim_params)

    s_params = {
        "tau": 15,
        "dim": 3,
        "s": [i / 10 for i in range(1, 5)],
        "M": 5,
        "sampling_rate": resampling_rate
    }
    # traverse_lyap(lorenz_x, s_params)

    M_params = {
        "tau": 15,
        "dim": 3,
        "s": 0.1,
        "M": [i for i in range(4, 10)],
        "sampling_rate": resampling_rate
    }
    # traverse_lyap(lorenz_x, M_params)
