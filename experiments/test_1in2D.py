from string_bonds.states import StringBondState
import numpy as np
from xmps.Hamiltonians import Hamiltonian
from xmps.iOptimize import find_ground_state
from ansatze import *
from minimizer.minimizer import Minimize
from scipy.optimize import minimize
import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")




def obj(p, U, h):
    p1, p2 = np.split(p, 2)
    U1, U2 = U(p1), U(p2)
    state = StringBondState(U1, U2)
    return state.energy(h, opy=np.eye(4))*2-1


def take_monotonic(X):
    xs = [X[0]]
    for x in X[0:]:
        if x < xs[-1]:
            xs.append(x)
    return np.array(xs)


def pad_p(p, new_size):
    p1, p2 = np.split(p, 2)
    pad_width = (new_size - len(p)) // 2
    return np.concatenate([np.pad(p1, (0, pad_width)), np.pad(p2, (0, pad_width))])


depths = [25]#[1, 3, 5, 7, 9, 11, 13]
tries = 5
λs = np.linspace(0.1, 1.4, 10)
traces = []
exacts = []
for λ in λs:
    h = Hamiltonian({"ZZ": -1, "X": λ}).to_matrices(2)[0]
    ress = []
    exacts.append(find_ground_state(h, 2, noisy=True, tol=1e-5)[1][-1])

    for i, depth in enumerate(depths):
        best = None
        for trie in range(tries):
            p = np.random.randn(30)
            try:
                res = Minimize(obj, p, args=(lambda Jh: U_full(Jh), h))
                if best is None or res.res.fun < best.res.fun:
                    best = res
            except np.linalg.LinAlgError:
                print('linalg error')

        xs = take_monotonic(best.last_stored_results)
        ress.append(xs[-1])
    traces.append(ress)

for i, trace in enumerate(np.array(traces).T):
    plt.plot(λs, trace, label=f'1in2d', marker='x')
plt.plot(λs, np.array(exacts), marker='x', label='mps')


plt.xlabel("$\lambda$")
plt.ylabel("energy density")
plt.title(f"TFIM energy density")
plt.legend()

plt.savefig("figs/energies_match.pdf")
np.save('data/exacts', np.array(exacts))
np.save('data/1d', np.array(traces))

