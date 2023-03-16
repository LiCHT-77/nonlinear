from algorism import embedding
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    path = "./sample/temperature.csv"
    df = pd.read_csv(path, index_col=0)
    # col = "那覇"
    col = "札幌"
    # col = "東京"
    series = df.interpolate(
        method='linear', limit_direction='forward', limit_area='inside')[col].values[:2000]

    fig = plt.figure(figsize=(10, 3.5))
    fig.subplots_adjust(top=1, bottom=0.15, right=0.96, left=0.05, wspace=0.5)
    taus = [1, 90, 180]
    for i, tau in enumerate(taus):
        embedded = embedding(series, tau, 2)
        ax = fig.add_subplot(1, 3, i + 1)
        ax.set_aspect('equal', adjustable='box')
        ax.plot(embedded[:, 0], embedded[:, 1])
        ax.set_title(f"τ = {tau}", y=-0.4)
        ax.set_xlabel("x(t)")
        ax.set_ylabel("x(t + τ)")
    fig.savefig(f"./out/embedding_{col}.png")

    plt.show()
