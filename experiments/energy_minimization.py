from string_bonds.states import StringBondState
import numpy as np
from xmps.Hamiltonians import Hamiltonian
from scipy.linalg import expm
from xmps.spin import paulis
from minimizer.minimizer import Minimize
from functools import reduce

from scipy.optimize import minimize
import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")

X, Y, Z = paulis(0.5)
CX = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 0, 1],
               [0, 0, 1, 0]])
H = np.array([[1, 1], [1, -1]])/np.sqrt(2)


def Rx(θ):
    """this is about 100 times faster than exponentiating!"""
    return np.array(
        [[np.cos(θ / 2), -1j * np.sin(θ / 2)], [-1j * np.sin(θ / 2), np.cos(θ / 2)]]
    )


def Ry(θ):
    return np.array([[np.cos(θ / 2), np.sin(θ / 2)], [-np.sin(θ / 2), np.cos(θ / 2)]])


def Rz(θ):
    return np.array(
        [
            [np.cos(θ / 2) - 1j * np.sin(θ / 2), 0],
            [0, np.cos(θ / 2) + 1j * np.sin(θ / 2)],
        ]
    )


def Ui(h):
    """Ui
    :param h: O(3) vector in polar coordinates
    """
    α, θ, ϕ = h
    return Rz(-ϕ) @ Ry(-θ) @ Rz(2 * α) @ Ry(θ) @ Rz(ϕ)


def Rh(J):
    """Rh
    entangling gate
    """
    return expm(
        1j * (J[0] * np.kron(X, X) + J[1] * np.kron(Y, Y) + J[2] * np.kron(Z, Z))
    )


def U_full(Jh):
    """U: full parametrisation of two qubit unitary
    :param Jh: [J1, J2, J3, hu11, hu12, hu13, hu21, hu22, hu23, hd12, ...]
    """
    J = Jh[:3]
    h1_u = Jh[3:6]
    h2_u = Jh[6:9]
    h1_d = Jh[9:12]
    h2_d = Jh[12:]
    return np.kron(Ui(h1_d), Ui(h2_d)) @ Rh(J) @ np.kron(Ui(h1_u), Ui(h2_u))


def U_shallow(Jh, depth=2, jsks=None):
    """shallow parametrisation - alternating CXs and local X rotations"""
    Rp = [Rx, Ry, Rz]
    jsks = np.zeros((depth, 2)).astype(int) + 1 if jsks is None else jsks  # all Ry
    return reduce(
        lambda x, y: x @ y,
        [
            CX @ np.kron(Rp[j](θ), Rp[k](φ))
            for (θ, φ), (j, k) in zip([Jh[i : i + 2] for i in range(len(Jh) - 1)], jsks)
        ],
    )


h = Hamiltonian({"XX": 1, "Z": 1}).to_matrices(2)[0]


def obj(p, U):
    p1, p2 = p[:15], p[15:]
    U1, U2 = U(p1), U(p2)
    state = StringBondState(U1, U2)
    return state.energy(h)


def take_monotonic(X):
    xs = [X[0]]
    for x in X[0:]:
        if x < xs[-1]:
            xs.append(x)
    return np.array(xs)


p = np.random.randn(30)
res = Minimize(obj, p, args=(lambda Jh: U_shallow(Jh, 1),))
xs = take_monotonic(res.last_stored_results)
plt.plot(xs, label="depth 1")

res = Minimize(obj, p, args=(lambda Jh: U_shallow(Jh, 2),))
xs = take_monotonic(res.last_stored_results)
plt.plot(xs, label="depth 2")

res = Minimize(obj, p, args=(lambda Jh: U_full(Jh),))
xs = take_monotonic(res.last_stored_results)
plt.plot(xs, label="full")

plt.xlabel("# function evaluations")
plt.ylabel("energy density")
plt.title("TFIM at $\\lambda=1$")
plt.legend()

plt.savefig("energies.pdf")
