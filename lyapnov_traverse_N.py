from lyapnov_lorenz import resampling, lyap_j
from algorism import embedding
import sys
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    sampling_rate = 0.01
    resampling_rate = 0.1
    path = sys.argv[1]
    series = pd.read_csv(path, index_col=0).iloc[:, 0]
    if sampling_rate < resampling_rate:
        series = resampling(series, sampling_rate, resampling_rate)

    Ns = [i for i in range(100, len(series), 10)]
    param = {
        "tau": 1,
        "dim": 3,
        "s": resampling_rate,
        "M": 3,
        "sampling_rate": resampling_rate
    }
    lyaps = []
    for n in Ns:
        embedded = embedding(series[:n], **param)
        lyap = lyap_j(
            embedded,
            **param
        )
        lyaps.append(lyap)
        print(f"N={n}:{lyap}")

    plt.plot(lyaps)
    plt.show()
