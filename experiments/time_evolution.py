from string_bonds.states import StringBondState
from minimizer.minimizer import Minimize
from xmps.Hamiltonians import Hamiltonian
from scipy.linalg import expm
from ansatze import *
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")

np.random.seed(500)


hx = Hamiltonian({"ZZ": -1, "Z": 1}).to_matrices(2)[0]
hy = Hamiltonian({"ZZ": -1, "Z": 1}).to_matrices(2)[0]

dt = 1e-3
Ux = expm(-1j * dt * hx)
Uy = expm(-1j * dt * hy)


def objx(p, p_, U, Wx=Ux):
    """objx. The objective function for a quarter timestep, i.e. the even/odd x trotter step

    Args:
        p: The parameters of the first unitary
        p_: The parameters of the second unitary
        U: The function mapping parameters -> unitary
        Wx: The (quarter) timestep evolution operator expm(-1j*dt*hx)
    """
    p1, p2 = np.split(p, 2)
    U1, U2 = U(p1), U(p2)
    state1 = StringBondState(U1, U2)

    p1, p2 = np.split(p_, 2)
    U1, U2 = U(p1), U(p2)
    state2 = StringBondState(U1, U2)

    ρx = Wx @ state1.rhox() @ Wx.conj().T
    σx = state2.rhox()

    return np.real(
        np.trace(ρx @ ρx) + np.trace(σx @ σx) - 2 * np.real(np.trace(ρx @ σx))
    )

itebd_objx(np.random.randn(30), np.random.randn(30), U_full, Ux)
raise Exception

def energy(p, U):
    p1, p2 = np.split(p, 2)
    U1, U2 = U(p1), U(p2)
    state = StringBondState(U1, U2)
    return state.energy(hx, hy)


def ev(p, U, op):
    p1, p2 = np.split(p, 2)
    U1, U2 = U(p1), U(p2)
    state = StringBondState(U1, U2)
    return state.E(op)


def objy(p, p_, U, Wy=Uy):
    """objx. The objective function for a quarter timestep, i.e. the even/odd x trotter step

    Args:
        p: The parameters of the first unitary
        p_: The parameters of the second unitary
        U: The function mapping parameters -> unitary
        Wx: The (quarter) timestep evolution operator expm(-1j*dt*hx)
    """
    p1, p2 = np.split(p, 2)
    U1, U2 = U(p1), U(p2)
    state1 = StringBondState(U1, U2)

    p1, p2 = np.split(p_, 2)
    U1, U2 = U(p1), U(p2)
    state2 = StringBondState(U1, U2)

    ρy = Wy @ state1.rhoy() @ Wy.conj().T
    σy = state2.rhoy()

    return np.real(
        np.trace(ρy @ ρy) + np.trace(σy @ σy) - 2 * np.real(np.trace(ρy @ σy))
    )


def time_evolve(p0, N, dt, U, ops=[X, Y, Z], get_energy=True):
    """Time evolve the state specified by p0 for N timesteps of size dt

    Args:
        p0: The initial state parameters
        N: The number of (quarter) timesteps
        dt: The timestep
        U: The function mapping parameters -> unitaries
    """
    p = p0
    energies = []
    evs = []
    for _ in range(N):
        if get_energy:
            energies.append(energy(p, U))
        if ops:
            evs.append([ev(p, U, op) for op in ops])

        new_objx = lambda p_new: objx(p, p_new, U)
        new_objy = lambda p_new: objy(p, p_new, U)
        resx = Minimize(new_objx, p)
        resy = Minimize(new_objy, resx.res.x)
        p = resy.res.x
    return p, np.array(evs), np.array(energies)


depths = [8]
N = 10
for depth in depths:
    #p0 = np.concatenate([np.zeros(15), np.random.randn(15)])#np.random.randn(30)
    p0 = np.random.randn(30)
    #p0 = np.random.randn(4 * depth)
    p, evs, energies = time_evolve(p0, N, dt, lambda Jhs: U_full(Jhs))
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(evs)
    ax[1].plot(energies)
    ax[1].set_xlabel("t")

    ax[0].set_ylabel("$\\langle X\\rangle$")
    ax[1].set_ylabel("$\\langle H\\rangle$")
    plt.tight_layout()
    plt.savefig("figs/time_evolution.pdf")
