from states import *
from xmps.Hamiltonians import Hamiltonian
from xmps.spin import paulis
X, Y, Z = paulis(0.5)
I = np.eye(2)


def test_normalised():
    A = iMPS().random(2, 2).left_canonicalise()
    B = iMPS().random(2, 2).left_canonicalise()
    UA = tools.tensor_to_unitary(A[0])
    UB = tools.tensor_to_unitary(B[0])

    state = StringBondState(UA, UB)
    assert np.allclose(state.E(np.eye(2)), 1)
    assert np.allclose(state.E2x(np.eye(4)), 1)
    assert np.allclose(state.E2y(np.eye(4)), 1)

def test_rhox():
    A = iMPS().random(2, 2).left_canonicalise()
    B = iMPS().random(2, 2).left_canonicalise()
    UA = tools.tensor_to_unitary(A[0])
    UB = tools.tensor_to_unitary(B[0])
    state = StringBondState(UA, UB)
    ρ = state.rhox()
    assert np.allclose(ρ.conj().T, ρ)
    assert np.allclose(np.trace(ρ), 1)
    assert np.trace(ρ@ρ) <= np.trace(ρ)
    assert ρ.shape == (4, 4)

def test_rhoy():
    A = iMPS().random(2, 2).left_canonicalise()
    B = iMPS().random(2, 2).left_canonicalise()
    UA = tools.tensor_to_unitary(A[0])
    UB = tools.tensor_to_unitary(B[0])
    state = StringBondState(UA, UB)
    ρ = state.rhoy()
    assert np.allclose(ρ.conj().T, ρ)
    assert np.allclose(np.trace(ρ), 1)
    assert np.trace(ρ@ρ) <= np.trace(ρ)
    assert ρ.shape == (4, 4)

def test_evs_rhos():
    A = iMPS().random(2, 2).left_canonicalise()
    B = iMPS().random(2, 2).left_canonicalise()
    UA = tools.tensor_to_unitary(A[0])
    UB = tools.tensor_to_unitary(B[0])

    state = StringBondState(UA, UB)
    e1 = state.E2x(np.kron(X, I))
    e2 = state.E2y(np.kron(X, I))
    e3 = state.E(X)
    assert np.allclose(e1, e2)
    assert np.allclose(e2, e3)

    rhox = state.rhox()
    rhoy = state.rhoy()

    assert np.allclose(np.trace(np.kron(X, I)@rhox), e1)
    assert np.allclose(np.trace(np.kron(X, I)@rhoy), e1)

    e1_ = state.E2x(np.kron(X, X))
    assert np.allclose(np.trace(np.kron(X, X)@rhox), e1_)

    e1_ = state.E2y(np.kron(X, X))
    assert np.allclose(np.trace(np.kron(X, X)@rhoy), e1_)
