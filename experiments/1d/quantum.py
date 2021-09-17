import cirq
from qmps.tools import (
    environment_to_unitary,
    unitary_to_tensor
)

from scipy.linalg import cholesky, expm
from xmps.iMPS import TransferMatrix, iMPS
from xmps.Hamiltonians import Hamiltonian
from ansatze import U_full
import numpy as np
from scipy.optimize import minimize
from functools import partial
import sys
import matplotlib.pyplot as plt
import tqdm
from string_bonds.states import merge, MatrixProductState

# - building time evo circuits using an alternative type of time evolution scheme using local reduced density matrices:

# - FIRST ORDER TROTTER CASE:

class Full_U_Gate(cirq.Gate):
    def __init__(self, p):
        self.p = p
        self.U = U_full(p)

    def num_qubits(self):
        return 2

    def _circuit_diagram_info_(self, args):
        return ["A","A"]

    def _unitary_(self):
        return self.U
 

class Full_V_Gate(cirq.Gate):
    def __init__(self, v):
        self.v = v
        self.n_qbs = int(np.log2(v.shape[0])) 

    def num_qubits(self):
        return self.n_qbs

    def _circuit_diagram_info_(self, args):
        return ["V"] * self.n_qbs

    def _unitary_(self):
        return self.v


class Ham(cirq.Gate):
    def __init__(self, H):
        self.h = H

    def num_qubits(self):
        return 2

    def _circuit_diagram_info_(self, args):
        return ["H","H"]

    def _unitary_(self):
        return self.h


def get_env_exact(U):
    """get_env_exact: v. useful for testing. Much faster than variational optimization of the env.

    :param U:
    """
    η, l, r = TransferMatrix(unitary_to_tensor(U)).eigs()
    return environment_to_unitary(cholesky(r).conj().T)


def two_site_unitary_to_tensor_time_evolved(U):
    return U.reshape([2,2,2,2,2,2,2,2])[:,:,:,:,0,0,:,:].reshape(4,4,4).transpose(1,0,2)
    

def two_site_unitary_to_tensor(U):
    return U.reshape([2,2,2,2,2,2])[:,:,:,0,0,:].reshape(2,4,2).transpose(1,0,2)


def get_env_2_site_unitary(U):
    if U.shape[0] == 16:
        AA = two_site_unitary_to_tensor_time_evolved(U) 

    elif U.shape[0] == 8:
        AA = two_site_unitary_to_tensor(U)

    _, _, r = TransferMatrix(AA).eigs()
    return environment_to_unitary(cholesky(r).conj().T)


def build_single_time_evolved_site(U,H0):
    """
-0 |0⟩------|----------------- i0
-           U
-1 |0⟩--|---|---|------------- σ0
-       U       H0
-2 j0 --|-------|------------- σ1
    """
    qubits = cirq.GridQubit.rect(1,3)

    m1 = cirq.Moment([U.on(*qubits[1:3])])
    m2 = cirq.Moment([U.on(*qubits[0:2])])
    m3 = cirq.Moment([H0.on(*qubits[1:3])])

    return  cirq.Circuit((m1,m2,m3))



def build_time_evolved_site(U,H0,H1):
    """
-0 |0⟩--------|---------------  i0
              U
-1 |0⟩----|---|---|-----------  i1
          U       H0
-2 j0 ----|-------|---|-------  σ0
                      H1
-3 j1 ----------------|-------  σ1
    """

    qubits = cirq.GridQubit.rect(1,4)

    m1 = cirq.Moment([U.on(*qubits[1:3])])
    m2 = cirq.Moment([U.on(*qubits[0:2])])
    m3 = cirq.Moment([H0.on(*qubits[1:3])])
    m4 = cirq.Moment([H1.on(*qubits[2:])])

    return  cirq.Circuit((m1,m2,m3,m4))


def build_candidate_site(U):
    """
-0 |0⟩--------|---------------  i0
              U
-1 |0⟩----|---|---------------  σ0
          U       
-2 j0 ----|-------------------  σ1
                    
    """

    qubits = cirq.GridQubit.rect(1,3)

    m1 = cirq.Moment([U.on(*qubits[1:3])])
    m2 = cirq.Moment([U.on(*qubits[0:2])])

    return cirq.Circuit((m1,m2))



def environment_from_circuit(circuit):
    U = cirq.unitary(circuit)
    return get_env_2_site_unitary(U) 
    

def build_time_evolved_state_single(U,H0,V):
    """
    Build this circuit:

-0 |0⟩--------|---------------  i0
-             U
-1 |0⟩----|---|---|-----------  i1
-         U       H0
-2 |0⟩----|-------|-----------  σ0
-    V |              
-3 |0⟩----------------
    
    """

    qubits = cirq.GridQubit.rect(1,4)

    circuit = cirq.Circuit()
    circuit.append(V.on(*qubits[2:]))
    circuit.append(U.on(*qubits[1:3]))
    circuit.append(U.on(*qubits[0:2]))
    circuit.append(H0.on(*qubits[1:3]))

    return circuit

    
def build_no_time_evo_rdm(U, V):
    """
    Build this circuit:

-0 |0⟩--------|---------------
-             U
-1 |0⟩----|---|---------------  σ1
-         U       
-2 |0⟩-|--|-------------------  σ0
-      V              
-3 |0⟩-|----------------------  
    
    """
 
    qubits = cirq.GridQubit.rect(1,4)
    m1 = cirq.Moment([V.on(*qubits[2:])])
    m2 = cirq.Moment([U.on(*qubits[1:3])])
    m3 = cirq.Moment([U.on(*qubits[0:2])])

    return cirq.Circuit((m1,m2, m3))
    

def build_time_evo_rdm(U,V,H):
    """
    Build this circuit:

-0 |0⟩--------------|---|-----
-                   U   H
-1 |0⟩----------|---|---|---|-  σ1
-               U           H
-2 |0⟩-----|----|-------|---|-  σ0
-          U            H
-3 |0⟩-|---|------------|-----  
-      V
-4 |0⟩-|----------------------  
    """

    qubits = cirq.GridQubit.rect(1,5)
    m1 = cirq.Moment([V.on(*qubits[3:])])
    m2 = cirq.Moment([U.on(*qubits[2:4])])
    m3 = cirq.Moment([U.on(*qubits[1:3])])
    m4 = cirq.Moment([U.on(*qubits[0:2])])
    m5 = cirq.Moment([H.on(*qubits[0:2]), H.on(*qubits[2:4])])
    m6 = cirq.Moment([H.on(*qubits[1:3])])

    return cirq.Circuit((m1,m2,m3,m4,m5,m6))


def energy_from_rdm_params(p, H):

    V = get_env_exact(U_full(p))
    c1 = build_no_time_evo_rdm(Full_U_Gate(p), Full_V_Gate(V)) 
    qubits = cirq.GridQubit.rect(1,5)
    sim = cirq.Simulator()

    rho = sim.simulate(c1).density_matrix_of(qubits[1:3])

    return np.trace(rho@H)


def compare_energy_change_trotter(dt,H):
    random_seed = 500
    np.random.seed(random_seed)
    params = np.random.rand(15)
    U = U_full(params)
    V = get_env_exact(U)


    W = expm(-1j * H * dt)
    c1 = build_no_time_evo_rdm(
        Full_U_Gate(params),
        Full_V_Gate(V)
        )

    c2 = build_time_evo_rdm(
        Full_U_Gate(params), 
        Full_V_Gate(V), 
        Ham(W))


    qubits = cirq.GridQubit.rect(1,5)
    sim = cirq.Simulator()

    rho = sim.simulate(c1).density_matrix_of(qubits[1:3])
    sigma = sim.simulate(c2).density_matrix_of(qubits[1:3])

    E1 = np.trace(rho@H)
    E2 = np.trace(sigma@H)

    return E1-E2


def rdm_trace_distance(p_candidate, p_target, W):
    sim = cirq.Simulator()
    qbs = cirq.GridQubit.rect(1,5)
    A = U_full(p_target)
    B = U_full(p_candidate)
    Va = get_env_exact(A)
    Vb = get_env_exact(B)

    c1 = build_time_evo_rdm(Full_U_Gate(p_target), Full_V_Gate(Va), Ham(W))

    c2 = build_no_time_evo_rdm(Full_U_Gate(p_candidate), Full_V_Gate(Vb))

    ρ = sim.simulate(c1).density_matrix_of(qbs[1:3])
    σ = sim.simulate(c2).density_matrix_of(qbs[1:3])

    return np.real(
        np.trace(ρ @ ρ) + np.trace(σ @ σ) - 2 * np.real(np.trace(ρ @ σ))
    )


def rdm_time_evolve_step(p_target, W):

    cost_func = partial(
        rdm_trace_distance,
        p_target = p_target,
        W = W
    )

    def show(x):
        print(np.round(cost_func(x),8))
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K")


    res = minimize(
        cost_func, p_target, method = "Nelder-Mead", callback=show, tol=1e-8
    )

    return res.x


class StepParams:
    def __init__(self, steps, dt, fname):
        self.steps = steps
        self.dt = dt
        self.fname = fname


def find_energy_drift_v2():
    H = Hamiltonian({"ZZ":-1, "Z":1}).to_matrices(2)[0]
    random_seed = 500
    np.random.seed(random_seed)

    params = np.random.rand(15)

    steps = [
        StepParams(100, 1e-2, "dt_1e2"),
        StepParams(200, 5e-3, "dt_5e3"),
        StepParams(1000, 1e-3, "dt_1e3")
    ]
    
    energies = []
    for sp in steps:
        steps = sp.steps
        dt = sp.dt
        fname = sp.fname

        W = expm(-1j * H * dt )

        for _ in tqdm.tqdm(range(steps)):
            energies.append(energy_from_rdm_params(params, H))
            params = rdm_time_evolve_step(params, W)

        plt.plot(energies)
        plt.savefig(f"experiments/1d/figs/rdm_{fname}_energy_drift_double_time_evo_op.png")
        plt.show()



def build_time_evolved_state(U, H0, H1, V):
    """
    Build this circuit:

-0 |0⟩--------|---------------  i0
-             U
-1 |0⟩----|---|---|-----------  i1
-         U       H0
-2 |0⟩----|-------|---|-------  σ0
-      |              H1
-3 |0⟩----------------|-------  σ1
-    V |
-4 |0⟩--------------
-      |
-5 |0⟩--------------
    
    """

    qubits = cirq.GridQubit.rect(1,6)

    circuit = cirq.Circuit()
    circuit.append(V.on(*qubits[2:]))
    circuit.append(U.on(*qubits[1:3]))
    circuit.append(U.on(*qubits[0:2]))
    circuit.append(H0.on(*qubits[1:3]))
    circuit.append(H1.on(*qubits[2:4]))

    return circuit


def build_candidate_state(U, V):
    """
    Build this circuit:

-0 |0⟩--------|---------------  i0
-             U
-1 |0⟩----|---|---------------  σ0
-         U       
-2 |0⟩----|-------------------  σ1
-    V |              
-3 |0⟩---------------
    
    """

    qubits = cirq.GridQubit.rect(1,4)

    m0 = cirq.Moment([V.on(*qubits[2:])])
    m1 = cirq.Moment([U.on(*qubits[1:3])])
    m2 = cirq.Moment([U.on(*qubits[0:2])])

    return cirq.Circuit((m0,m1,m2))




def local_trace_distance_single(A,B,H0):
    """Local trace distance using just a single time evolution operator"""

    sim = cirq.Simulator()
    qbs = cirq.GridQubit.rect(1,6)

    c1 = build_single_time_evolved_site(A, H0)
    v1 = Full_V_Gate(environment_from_circuit(c1))

    c2 = build_candidate_site(A)
    v2 = Full_V_Gate(environment_from_circuit(c2))

    c3 = build_time_evolved_state_single(A, H0, v1)
    c4 = build_candidate_state(B, v2)

    σ = sim.simulate(c3).density_matrix_of(qbs[1:3])
    ρ = sim.simulate(c4).density_matrix_of(qbs[1:3])

    return np.real(
        np.trace(ρ @ ρ) + np.trace(σ @ σ) - 2 * np.real(np.trace(ρ @ σ))
    )



def local_trace_distance(A,B,H0,H1):
    """
    Ham in ["s", "d"]:
        s means that we only apply a single time evolution operator (similar to Fergus' code, and d means we apply two and explicitly increase the bond dim.)
    """
    sim = cirq.Simulator()
    qbs = cirq.GridQubit.rect(1,6)

    c1 = build_time_evolved_site(A, H0, H1)
    v1 = Full_V_Gate(environment_from_circuit(c1))

    c2 = build_candidate_site(A)
    v2 = Full_V_Gate(environment_from_circuit(c2))

    c3 = build_time_evolved_state(A, H0, H1, v1)
    c4 = build_candidate_state(B, v2)

    σ = sim.simulate(c3).density_matrix_of(qbs[2:4])
    ρ = sim.simulate(c4).density_matrix_of(qbs[1:3])

    return np.real(
        np.trace(ρ @ ρ) + np.trace(σ @ σ) - 2 * np.real(np.trace(ρ @ σ))
    )


def local_trace_cost_function(p, A, H0, H1):
    B = Full_U_Gate(p)
    return local_trace_distance(A, B, H0, H1)


def local_trace_cost_function_single(p, A, H0):
    B = Full_U_Gate(p)
    return local_trace_distance(A, B, H0)


def evolve_single_step_single(A_params, H0):
    A = Full_U_Gate(A_params)
    H0 = Ham(H0)

    cost_func = partial(
        local_trace_cost_function_single,
        A = A,
        H0 = H0
    )

    def show(x):
        print(np.round(cost_func(x),8))
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K")

    res = minimize(
        cost_func, A_params, method = "Nelder-Mead", callback=show, tol=1e-8
    )

    return res.x




def evolve_single_step(A_params, H0, H1):
    A = Full_U_Gate(A_params)
    H0 = Ham(H0)
    H1 = Ham(H1)

    cost_func = partial(
        local_trace_cost_function,
        A = A,
        H0 = H0,
        H1 = H1
    )

    def show(x):
        print(np.round(cost_func(x),8))
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K")

    res = minimize(
        cost_func, A_params, method = "Nelder-Mead", callback=show, tol=1e-8
    )

    return res.x


def energy_from_params(params, H):
    sim = cirq.Simulator()

    qbs = cirq.GridQubit.rect(1,4)

    A = Full_U_Gate(params)
    c1 = build_candidate_site(A)
    v1 = Full_V_Gate(environment_from_circuit(c1))
    c2 = build_candidate_state(A, v1)

    ρ = sim.simulate(c2).density_matrix_of(qbs[1:3])

    return np.real(np.trace(H @ ρ))


def energy(p, U, h):
    state = MatrixProductState(U(p))
    return state.energy(h)

def find_energy_drift():
    H = Hamiltonian({"ZZ":-1, "Z":1}).to_matrices(2)[0]

    steps_timestep_filename = [
        (200,5e-3,"dt_5e3"),(1000,1e-3,"dt_1e3")]


    for steps, dt, fname in steps_timestep_filename:
        random_seed = 500
        np.random.seed(random_seed)

        params = np.random.rand(15)

        # steps = 100
        # dt = 1e-2
    
        W = expm(-1j * H * dt )

        energies = []

        for _ in tqdm.tqdm(range(steps)):
            energies.append(energy(params, U_full, H))
            params = evolve_single_step(params, W, W)

        plt.plot(energies)
        plt.savefig(f"experiments/1d/figs/{fname}_energy_drift_double_time_evo_op.png")


def plot_energy_diff():
    H = Hamiltonian({"ZZ":-1, "X":1}).to_matrices(2)[0]
    dts = [0.,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,0.]
    func = partial(
        compare_energy_change_trotter,
        H = H
    )    

    scores = np.vectorize(func)(dts)

    plt.plot(scores)
    plt.show()
    print(scores)


if __name__ == "__main__":
    find_energy_drift_v2()
    





