import numpy as np
from xmps.iMPS import iMPS
from minimizer.minimizer import Minimize
from xmps.spin import paulis
from xmps.Hamiltonians import Hamiltonian
from string_bonds.states import merge, MatrixProductState
from ansatze import U_full as U
from ansatze import *
from scipy.linalg import expm
import matplotlib.pyplot as plt
import tqdm 
plt.style.use("seaborn-whitegrid")

np.random.seed(500)
X, Y, Z = paulis(0.5)


x = iMPS().random(2, 5).left_canonicalise()

p = np.random.randn(15)
mps = MatrixProductState(U(p))


h = Hamiltonian({"ZZ": -1, "Z": 1}).to_matrices(2)[0]

dt = 1e-3
U = expm(-1j * dt * h)

def obj_local(p, p_, U, W=U):
    """obj. The objective function for a half timestep, i.e. the even/odd trotter step

    Args:
        p: The parameters of the first unitary
        p_: The parameters of the second unitary
        U: The function mapping parameters -> unitary
        Wx: The (quarter) timestep evolution operator expm(-1j*dt*hx)
    """
    state1 = MatrixProductState(U(p))

    state2 = MatrixProductState(U(p_))

    ρ = W @ state1.rho() @ W.conj().T
    σ = state2.rho()

    return np.real(
        np.trace(ρ @ ρ) + np.trace(σ @ σ) - 2 * np.real(np.trace(ρ @ σ))
    )

def energy(p, U):
    state = MatrixProductState(U(p))
    return state.energy(h)

def ev(p, U, op):
    state = MatrixProductState(U(p))
    return state.E(op)

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
    obj = obj_local
    for _ in tqdm.tqdm(range(N)):
        if get_energy:
            energies.append(energy(p, U))
        if ops:
            evs.append([ev(p, U, op) for op in ops])

        new_obj = lambda p_new: obj(p, p_new, U)
        res = Minimize(new_obj, p, tol=1e-8)
        p = res.res.x
    return p, np.array(evs), np.array(energies)

N = 1000
T = [k*dt for k in range(N)]
p0 = np.random.randn(15)
print(f"Time evolution, {T[-1]}, {N} timesteps")

p, evs, energies = time_evolve(p0, N, dt, lambda Jhs: U_full(Jhs))

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(T, evs)
ax[1].plot(T, energies)
ax[1].set_xlabel("t")

ax[0].set_ylabel("$\\langle \\sigma \\rangle$")
ax[1].set_ylabel("$\\langle H\\rangle$")
plt.tight_layout()
plt.savefig("experiments/1d/figs/1d_time_evolution_local_shorter1e3.pdf")
np.save('data/1d_energies_local', energies)
np.save('data/1d_evs_local', evs)
