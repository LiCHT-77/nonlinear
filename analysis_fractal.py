import matplotlib.pyplot as plt
from matplotlib import gridspec
from algorism import embedding, fractal_gp, gen_henon_map
import numpy as np
import pandas as pd


def traverse(series, dims: list[int], taus: list[int] = [], window: int = 10):
    if taus == []:
        taus = [window // m for m in dims]

    r_min = 1
    r_max = 5
    fig = plt.figure(figsize=(6, 8))
    fig.subplots_adjust(hspace=0.3, top=0.99)
    fig.tight_layout()
    spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[2, 1])
    ax_correlation = fig.add_subplot(spec[0])
    ax_correlation.tick_params(labelsize=12)
    ax_correlation.set_xlabel('log r', fontsize=14)
    ax_correlation.set_ylabel('log C^m(r)', fontsize=14)
    ax_correlation.set_xlim([r_min, r_max])
    ax_correlation.set_title(
        'Correlation integration', y=-0.2)
    ax_slope = fig.add_subplot(spec[1])
    ax_slope.tick_params(labelsize=12)
    ax_slope.set_xlabel('log r', fontsize=14)
    ax_slope.set_ylabel('Slope', fontsize=14)
    ax_slope.set_xlim([r_min, r_max])
    ax_slope.set_ylim([0, 10])
    ax_slope.set_title(
        'Slope of correlation integration', y=-0.4)
    for m, tau in list(zip(dims, taus)):
        state_space = embedding(series, tau, m)
        ri = fractal_gp(state_space, (np.exp(r_min), np.exp(r_max)), 100)
        ri.to_csv("./out/fractal.csv")
        ax_correlation.scatter(
            ri['log r'], ri['log C^m(r)'], marker='.', label=f'm = {m}')
        ax_slope.scatter(ri['log r'], ri['Slope'],
                         marker='.', label=f'm = {m}')
        ax_correlation.legend()
        ax_slope.legend()
    ax_slope.set_yticks(np.arange(0, 15, 5))
    ax_slope.axvline(2.7, color="black")
    ax_slope.axvline(3.2, color="black")
    plt.show()
    return fig


def correlation_exponent(ri, period):
    var = ri["Slope"].rolling(period).var()
    var = var.rename("var")
    mean = ri["Slope"].rolling(period).mean()
    mean = mean.rename("mean")
    diff = ri["Slope"].diff(period)
    diff = diff.rename("diff")
    aggregate = pd.concat([var, mean, diff], axis=1)
    passed = aggregate[(aggregate["var"] <= 0.005)].tail(1)
    print(passed)

    if len(passed) != 0:
        return passed["mean"].values[0]
    else:
        return np.nan


def converge(series, dims: list[int], taus: list[int] = [], window: int = 10):
    if taus == []:
        taus = [window // m for m in dims]

    r_min = 1
    r_max = 5
    ces = []
    for m, tau in list(zip(dims, taus)):
        state_space = embedding(series, tau, m)
        ri = fractal_gp(state_space, (np.exp(r_min), np.exp(r_max)), 100)
        ce = correlation_exponent(ri, 10)
        ces.append(ce)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_ylim([0, 2])
    ax.plot(dims, ces)
    plt.show()


if __name__ == "__main__":
    path = "./sample/temperature.csv"
    dt = 0.1
    N = 5000
    taus = [7, 30, 90]
    max_dim = 9

    plot_rows = 2
    plot_cols = len(taus)
    df = pd.read_csv(path, index_col=0)
    col = "那覇"
    col = "札幌"
    col = "東京"
    series = df.interpolate(
        method='linear', limit_direction='forward', limit_area='inside')[col].values
    t = np.arange(0, 4000)

    traverse(series, list(range(3, 11)), taus=[90 for _ in range(3, 11)])
