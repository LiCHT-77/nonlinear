from lyapnov_lorenz import resampling, traverse_lyap
import sys
import pandas as pd

if __name__ == "__main__":
    sampling_rate = 0.01
    resampling_rate = 0.1
    path = sys.argv[1]
    series = pd.read_csv(path, index_col=0).iloc[:, 0]
    if sampling_rate < resampling_rate:
        series = resampling(series, sampling_rate, resampling_rate)

    N = len(series)
    print(f"N={N}")

    dims = [i for i in range(2, 10)]
    params = {
        "tau": 1,
        "dim": dims,
        "s": resampling_rate,
        "M": max(dims),
        "sampling_rate": resampling_rate
    }
    traverse_lyap(series, params)
