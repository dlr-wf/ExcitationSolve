"""Batched parameter prediction must reproduce the original per-excitation implementation.

`optimal_thetas` shares the AO->MO transform and the reference-determinant energy across the pool
and reads the integrals directly, instead of expanding them into a full (2n)^4 spin-orbital tensor.
That optimisation is only defensible if it changes nothing, so the ORIGINAL tensor-based
implementation is kept here verbatim as a slow-but-obviously-correct oracle and the two are
required to agree BITWISE -- not merely within a tolerance, which would hide a spin-rule error.
"""

import itertools

import numpy as np
import pytest
from pyscf import gto, scf

from excitationsolve import (
    optimal_theta,
    optimal_theta_pyscf,
    optimal_thetas,
    optimal_thetas_pyscf,
)
from excitationsolve.parameter_prediction import (
    _apply_double_excitation,
    _block_to_interleaved,
    _transform_integrals_to_mo,
)


# ---------------------------------------------------------------------------
# Oracle: the pre-batching implementation, kept here verbatim. Materialises the full
# spin-orbital tensor. Deliberately NOT imported from the package -- the package's
# optimal_theta is now a thin wrapper around optimal_thetas, so comparing against it
# would be tautological.
# ---------------------------------------------------------------------------
def _determinant_energy(h1_mo, eri_mo, occ_indices):
    """
    Compute the energy of a single Slater determinant in MO basis.

    Implements (spin-orbital, physicist notation):
        E = sum_{p in occ} h_pp
            + 1/2 sum_{p,q in occ} ( <pq|pq> - <pq|qp> )

    Parameters
    ----------
    h1_mo : np.ndarray, shape (nso, nso)
        One-electron integrals h_{pq}.
    eri_mo : np.ndarray, shape (nso, nso, nso, nso)
        Two-electron integrals in PHYSICIST notation <pq|rs>.
    occ_indices : list[int]
        Occupied spin-orbital indices for this determinant.

    Returns
    -------
    float
        Determinant energy.
    """
    occ = np.array(occ_indices, dtype=int)

    # One-electron part
    e_one = np.sum(h1_mo[occ, occ])

    # Two-electron part: 1/2 sum_{pq} (J_pq - K_pq). In the (sigma,tau,tau,
    # sigma) operator ordering the Coulomb integral is h[p,q,q,p] and the
    # exchange integral is h[p,q,p,q] (which auto-vanishes for opposite spin).
    # The p == q term cancels exactly, so no spin/self guards are needed.
    e_two = 0.0
    for p in occ:
        for q in occ:
            e_two += eri_mo[p, q, q, p] - eri_mo[p, q, p, q]

    return e_one + 0.5 * e_two


def _compute_a_b(h1_mo, h2_mo, occ_indices, i, j, k, l):
    """
    Compute the parameters a and b for the two-level effective Hamiltonian
    defined by the HF determinant and a doubly excited determinant.

    Using the same index convention as the equations: the excitation removes
    the occupied spin-orbitals (i, j) and fills the virtual spin-orbitals
    (k, l) via the generator G = a^dag_k a^dag_l a_i a_j, so that
    |Phi> = G |HF>.

        a = ( <HF|H|HF> - <Phi|H|Phi> ) / 2
        b = <HF|H|Phi>

    The indices must be passed in the generator's operator order (the TCC /
    Qiskit-sorted order), since the sign of b is fixed by the fermionic
    ordering of G.

    Parameters
    ----------
    h1_mo : np.ndarray
        One-electron spin-orbital integrals h_{pq}.
    h2_mo : np.ndarray
        Two-electron spin-orbital integrals in PHYSICIST notation,
        h2_mo[p, q, r, s] = <pq|rs>.
    occ_indices : list[int]
        Occupied spin-orbital indices of the HF determinant.
    i, j : int
        Occupied spin-orbitals removed by the excitation (annihilation order).
    k, l : int
        Virtual spin-orbitals filled by the excitation (creation order).

    Returns
    -------
    a : float
    b : float
    """
    # HF determinant energy
    e_hf = _determinant_energy(h1_mo, h2_mo, occ_indices)

    # Excited determinant: remove (i, j), add (k, l)
    occ_exc = _apply_double_excitation(occ_indices, k, l, i, j)
    e_exc = _determinant_energy(h1_mo, h2_mo, occ_exc)

    a_val = 0.5 * (e_hf - e_exc)

    # b = <HF|H|Phi>, matching the derivation
    #     2b = Re(2 h_ijkl - [h_jikl + h_ijlk] delta_{mu nu})
    # i.e. b = Re( h_ijkl - 1/2 (h_jikl + h_ijlk) delta_{mu nu} ).
    # Indices must be passed so that the direct term is spin-allowed, i.e.
    # spin(i)=spin(l) and spin(j)=spin(k); then in the (sigma,tau,tau,sigma)
    # ordering the two exchange terms automatically vanish for an opposite-spin
    # excitation, which is exactly the delta_{mu nu}. The overall sign relative
    # to the TCC generator is applied by the caller (optimal_theta_max).
    b_val = h2_mo[i, j, k, l]
    same_spin = np.unique([index % 2 for index in (k, l, i, j)]).size == 1
    if same_spin:
        b_val -= 0.5 * (h2_mo[j, i, k, l] + h2_mo[i, j, l, k])
    b_val = float(np.real(b_val))

    return a_val, b_val


def _build_spin_orbital_integrals(h1, h2):
    """
    Expand spatial-orbital integrals into full spin-orbital integrals.
    ERIs are given in chemist notation and are transformed to physicist notation in the output.

    Parameters
    ----------
    h1 : (n, n) ndarray
        Spatial one-electron integrals h_pq
    h2 : (n, n, n, n) ndarray
        Spatial two-electron integrals in chemist notation (pq|rs).

    Returns
    -------
    h1_so : (2n, 2n) ndarray
        Spin-orbital one-electron integrals h_{pq}
    h2_so : (2n, 2n, 2n, 2n) ndarray
        Spin-orbital two-electron integrals in PHYSICIST notation,
        h2_so[p, q, r, s] = <pq|rs>, i.e. the integrals that multiply
        a^dag_p a^dag_q a_r a_s and match the indices used in the equations.
    """
    n = h1.shape[0]
    nso = 2 * n

    # Allocate spin-orbital integrals
    h1_so = np.zeros((nso, nso))
    h2_so = np.zeros((nso, nso, nso, nso))

    # One-electron integrals:
    # h_{pσ,qσ} = h_{pq}, h_{pσ,qτ} = 0 for σ ≠ τ
    for p in range(n):
        for q in range(n):
            for sigma in (0, 1):
                p_so = 2 * p + sigma
                q_so = 2 * q + sigma
                h1_so[p_so, q_so] = h1[p, q]

    # Two-electron integrals in PHYSICIST notation matching the operator
    # ordering a^dag_p a^dag_q a_r a_s used in the equations:
    #   h_{pqrs} = <pq|sr> = (ps|qr)_chem.
    # The spin pattern across the four index positions is (sigma, tau, tau,
    # sigma): electron 1 sits on positions p,s and electron 2 on positions q,r,
    # so the element is non-zero only when spin(p)=spin(s) and spin(q)=spin(r).
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    for sigma in (0, 1):
                        for tau in (0, 1):
                            p_so = 2 * p + sigma  # position 0, electron 1, spin sigma
                            q_so = 2 * q + tau  # position 1, electron 2, spin tau
                            r_so = 2 * r + tau  # position 2, electron 2, spin tau
                            s_so = 2 * s + sigma  # position 3, electron 1, spin sigma
                            h2_so[p_so, q_so, r_so, s_so] = h2[p, s, q, r]

    return h1_so, h2_so


def _reference_optimal_theta(h1: np.ndarray, eri: np.ndarray, occ_spatial: list[int], excitation_indices: list[int]) -> tuple[float, float]:
    """
    High-level convenience function to compute the optimal VQE angle theta_opt
    for a given double excitation in an RHF reference.

    Accepts TCC-format excitation indices after Qiskit-style sorting:
        (*sorted(virts), *sorted(occs)) → indices sorted by value within each pair,
        occ/virt identity determined by checking against HF occupation.

    Parameters
    ----------
    h1 : np.ndarray
        One-electron matrix elements.
    eri : np.ndarray
        Two-electron matrix elements in chemist order.
    occ_spatial : list[int]
        Spatial orbital indices of occupied spatial orbital in the referenc state (e.g. Hartree-Fock state)
    excitation_indices : list[int] or tuple[int, int, int, int]
        Four spin-orbital indices in TCC/Qiskit-sorted format.

    Returns
    -------
    theta_opt : float
        Exact optimal angle (in radians) minimising E(theta).
    delta_E : float
        Maximum energy impact a + sqrt(a^2 + b^2).
    """
    # Orbital counts (per spin) for the block <-> interleaved index mapping
    no = len(occ_spatial)
    nv = h1.shape[0] - no

    # Build spin-orbital integrals (interleaved 2*spatial+spin ordering)
    h1_so, h2_so = _build_spin_orbital_integrals(h1, eri)
    occ_so = {2 * occ for occ in occ_spatial} | {2 * occ + 1 for occ in occ_spatial}

    # Map the TCC block-ordered excitation indices to interleaved ordering,
    # preserving operator order: G = a^dag_v0 a^dag_v1 a_o0 a_o1, where the
    # Qiskit/TCC sorting makes (v0, v1) the virtual creations and (o0, o1) the
    # occupied annihilations.
    v0, v1, o0, o1 = (_block_to_interleaved(idx, no, nv) for idx in excitation_indices)

    assert o0 in occ_so and o1 in occ_so, "expected last two indices occupied"
    assert v0 not in occ_so and v1 not in occ_so, "expected first two indices virtual"

    # Assign equation labels so the direct integral h_ijkl is spin-allowed,
    # i.e. spin(i)=spin(l) and spin(j)=spin(k). The TCC sorting guarantees
    # spin(v0)=spin(o0) and spin(v1)=spin(o1), so pair i<->l (=o0,v0) and
    # j<->k (=o1,v1). This swaps which virtual is k vs l relative to the bare
    # operator order.
    i_occ, j_occ = o0, o1
    k_vir, l_vir = v1, v0

    a_val, b_val = _compute_a_b(h1_so, h2_so, list(occ_so), i_occ, j_occ, k_vir, l_vir)

    # theta maximizes the energy impact; the overall minus sign is the
    # exp(-i theta G) vs. TCC parameter-sign convention.
    # Pick the energy-MINIMIZING stationary point of the period-pi landscape
    # E(theta) = const + a(1 - cos 2theta) + b sin 2theta. Using arctan(b/a)
    # discards the sign of a and lands on the maximum whenever a > 0 (which
    # happens for excitations whose doubly-excited determinant falls below HF,
    # e.g. near dissociation). arctan2(-b, -a) keeps the quadrant and always
    # selects the minimum (correct mod pi, the physical period of the landscape).
    theta_opt = -0.5 * np.arctan2(-b_val, -a_val)

    delta_E = a_val + np.sqrt(a_val**2 + b_val**2)

    return theta_opt, delta_E



def _molecule(basis):
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22", basis=basis, verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    return mf


def _pool(mf):
    """All spin-conserving doubles in TCC/Qiskit-sorted block order."""
    no = int(np.sum(mf.mo_occ > 0.0))
    nv = mf.mo_coeff.shape[1] - no
    nso = 2 * (no + nv)
    occ_block = [i for i in range(nso) if i < no or no + nv <= i < 2 * no + nv]
    vir_block = [i for i in range(nso) if i not in occ_block]
    no_ = no
    nv_ = nv

    def spin(idx):
        return _block_to_interleaved(idx, no_, nv_) % 2

    return [(v0, v1, o0, o1)
            for o0, o1 in itertools.combinations(occ_block, 2)
            for v0, v1 in itertools.combinations(vir_block, 2)
            if spin(v0) == spin(o0) and spin(v1) == spin(o1)]


def _valid_pool(mf):
    """Excitations optimal_theta accepts (the pool also contains invalid occ/virt splits)."""
    h1, eri, occ_spatial = _transform_integrals_to_mo(mf)
    valid = []
    for indices in _pool(mf):
        try:
            _reference_optimal_theta(h1, eri, occ_spatial, indices)
        except AssertionError:
            continue
        valid.append(indices)
    return valid


@pytest.mark.parametrize("basis", ["sto-3g", "6-31g"])
def test_batched_matches_single_bitwise(basis):
    """The optimisation is only defensible if it changes nothing."""
    mf = _molecule(basis)
    h1, eri, occ_spatial = _transform_integrals_to_mo(mf)
    pool = _valid_pool(mf)
    assert pool, "no valid excitations to compare"

    batched = optimal_thetas(h1, eri, occ_spatial, pool)
    assert len(batched) == len(pool)
    for indices, (theta, delta_E) in zip(pool, batched):
        theta_ref, delta_E_ref = _reference_optimal_theta(h1, eri, occ_spatial, indices)
        assert theta == theta_ref
        assert delta_E == delta_E_ref


def test_results_follow_input_order():
    """Results are returned unsorted, aligned 1:1 with the excitations passed in."""
    mf = _molecule("sto-3g")
    h1, eri, occ_spatial = _transform_integrals_to_mo(mf)
    pool = _valid_pool(mf)

    forward = optimal_thetas(h1, eri, occ_spatial, pool)
    reversed_ = optimal_thetas(h1, eri, occ_spatial, list(reversed(pool)))
    assert reversed_ == list(reversed(forward))


def test_pyscf_wrapper_matches_integral_api():
    mf = _molecule("sto-3g")
    pool = _valid_pool(mf)
    h1, eri, occ_spatial = _transform_integrals_to_mo(mf)
    assert optimal_thetas_pyscf(mf, pool) == optimal_thetas(h1, eri, occ_spatial, pool)


def test_returns_plain_tuples():
    mf = _molecule("sto-3g")
    theta, delta_E = optimal_thetas_pyscf(mf, _valid_pool(mf))[0]
    assert isinstance(theta, float) and isinstance(delta_E, float)


# H4/STO-3G block layout (no=2, nv=2): 0,1 beta-occ | 2,3 beta-virt | 4,5 alpha-occ |
# 6,7 alpha-virt. (2, 6, 0, 4) is a well-formed spin-conserving excitation; the cases below
# each violate exactly one precondition.
@pytest.mark.parametrize("indices, reason", [
    ((2, 6, 4, 0), "spin-conserving"),      # v0 beta but o0 alpha
    ((2, 6, 3, 4), "occupied"),             # o0 is a virtual
    ((0, 6, 1, 4), "virtual"),              # v0 is occupied
])
def test_malformed_excitation_is_rejected(indices, reason):
    """Preconditions raise ValueError, not AssertionError.

    The inlined integral lookup drops the spin guard that the tensor construction applied
    implicitly, so a violation would otherwise return a plausible wrong number. `python -O`
    strips asserts, hence an explicit raise.
    """
    mf = _molecule("sto-3g")
    h1, eri, occ_spatial = _transform_integrals_to_mo(mf)
    with pytest.raises(ValueError, match=reason):
        optimal_thetas(h1, eri, occ_spatial, [indices])


def test_valid_excitation_is_accepted():
    """Guards the parametrisation above: the reference excitation must NOT raise."""
    mf = _molecule("sto-3g")
    h1, eri, occ_spatial = _transform_integrals_to_mo(mf)
    assert len(optimal_thetas(h1, eri, occ_spatial, [(2, 6, 0, 4)])) == 1


def test_empty_pool():
    assert optimal_thetas_pyscf(_molecule("sto-3g"), []) == []


def test_single_excitation_wrappers_match_reference():
    """The scalar entry points keep their original signature and return shape."""
    mf = _molecule("sto-3g")
    h1, eri, occ_spatial = _transform_integrals_to_mo(mf)
    for indices in _valid_pool(mf):
        expected = _reference_optimal_theta(h1, eri, occ_spatial, indices)
        assert optimal_theta(h1, eri, occ_spatial, indices) == expected
        assert optimal_theta_pyscf(mf, indices) == expected
