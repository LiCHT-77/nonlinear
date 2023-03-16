import numpy as np
from algorism import gen_lorenz_arr, embedding
import matplotlib.pyplot as plt

if __name__ == "__main__":
    lorenz = gen_lorenz_arr(steps=5000)
    embedded = embedding(lorenz[:, 0], 10, 3)
    fig = plt.figure()
    fig.subplots_adjust(wspace=0.24, top=1, bottom=0, left=0.05)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.plot(lorenz[:, 0], lorenz[:, 1], lorenz[:, 2])
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.set_xlabel("x(t)")
    ax2.set_ylabel("x(t+τ)")
    ax2.set_zlabel("x(t+2τ)")
    ax2.plot(embedded[:, 0], embedded[:, 1], embedded[:, 2])
    fig.savefig("./out/embedding_demo")