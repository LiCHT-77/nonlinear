from algorism import gen_henon_map
from lyapnov_lorenz import traverse_lyap

if __name__ == "__main__":
    N = 10000
    henon_x = gen_henon_map(steps=N)[:, 0]

    tau_params = {
        "tau": [i for i in range(1, 5)],
        "dim": 2,
        "s": 1,
        "M": 2,
        "sampling_rate": 1
    }
    traverse_lyap(henon_x, tau_params)

    dim_params = {
        "tau": 1,
        "dim": [i for i in range(2, 5)],
        "s": 1,
        "M": 2,
        "sampling_rate": 1
    }
    traverse_lyap(henon_x, dim_params)
