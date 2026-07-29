"""Closed-form prediction of the optimal VQE angle of a double excitation.

For a single double-excitation generator G = a^dag_k a^dag_l a_i a_j applied to
an RHF reference, the energy as a function of the rotation angle theta is the
period-pi curve

    E(theta) = const + a (1 - cos 2theta) + b sin 2theta,

with a = (E_HF - E_exc) / 2 and b = <HF|H|Phi>. The minimiser of this curve has
the closed form theta = -1/2 * arctan2(-b, -a). `optimal_theta` evaluates a
and b from the molecular integrals and returns this exact angle.

The module is self-contained: it only needs a converged PySCF RHF object and the
four excitation indices (in TCC/Qiskit-sorted spin-orbital order).

For more information, see https://arxiv.org/abs/2602.10776
"""

from typing import NamedTuple

import numpy as np


def _transform_integrals_to_mo(mf):
    """
    Transform one- and two-electron integrals from AO to MO basis.

    Returns
    -------
    h1_mo : (nmo, nmo)
    eri_mo : (nmo, nmo, nmo, nmo)
    occ_indices : list[int]
    """
    try:
        from pyscf import ao2mo
    except ImportError as e:
        raise ImportError("pyscf is required for _transform_integrals_to_mo().") from e
    mo = mf.mo_coeff
    hcore_ao = mf.get_hcore()
    h1_mo = mo.T @ hcore_ao @ mo

    eri_ao = mf._eri
    nmo = mo.shape[1]

    # IMPORTANT: compact=False → full 4-index tensor
    eri_mo_4 = ao2mo.full(eri_ao, mo, compact=False)
    eri_mo = eri_mo_4.reshape(nmo, nmo, nmo, nmo)

    occ_indices = np.where(mf.mo_occ > 0.0)[0].tolist()
    return h1_mo, eri_mo, occ_indices


def _apply_double_excitation(occ_indices, i, j, a, b):
    """
    Build the occupied orbital list for a doubly excited determinant
    from a reference determinant.

    Parameters
    ----------
    occ_indices : list[int]
        Occupied spatial MO indices of the reference (HF) determinant.
    a, b : int
        Indices of occupied orbitals to be excited (i, j in occ).
    i, j : int
        Indices of virtual orbitals to be occupied in the excited determinant.

    Returns
    -------
    list[int]
        Occupied spatial MO indices of the excited determinant.
    """
    occ_set = set(occ_indices)
    occ_set.discard(a)
    occ_set.discard(b)
    occ_set.add(i)
    occ_set.add(j)
    return sorted(occ_set)


def _block_to_interleaved(idx, no, nv):
    """Map a TCC block-ordered spin-orbital index to the interleaved
    (2*spatial + spin) convention used for the spin-orbital integrals.

    TCC orders spin-orbitals as [beta_occ, beta_virt, alpha_occ, alpha_virt].
    spin: 0 = alpha, 1 = beta.
    """
    if idx < no:  # beta occ
        spatial, spin = idx, 1
    elif idx < no + nv:  # beta virt
        spatial, spin = no + (idx - no), 1
    elif idx < 2 * no + nv:  # alpha occ
        spatial, spin = idx - (no + nv), 0
    else:  # alpha virt
        spatial, spin = no + (idx - (2 * no + nv)), 0
    return 2 * spatial + spin



# ---------------------------------------------------------------------------
# Batched prediction. Performing parameter prediction on a batch of operators instead
# of individually reduces the overhead of computing integrals and the HF state again and again.  
#
# The integrals are also indexed directly instead of being expanded into a full
# spin-orbital tensor. _build_spin_orbital_integrals allocates (2n)^4 entries via
# a 6-deep Python loop -- 4.3 GB and ~1.3e8 iterations at n=76 -- while every
# quantity needed here is a handful of specific elements.
#
# Conventions match _build_spin_orbital_integrals exactly:
#   spin-orbital index p -> spatial p // 2, spin p % 2
#   h2_so[p, q, r, s] = h2[p//2, s//2, q//2, r//2]
#                       if spin(p) == spin(s) and spin(q) == spin(r) else 0
# ---------------------------------------------------------------------------
class ExcitationPrediction(NamedTuple):
    """One scored excitation. Tuple-compatible, so it unpacks as
    ``excitation_indices, theta, delta_E``."""

    excitation_indices: tuple[int, ...]
    theta: float
    delta_E: float


def _h2_so_element(h2: np.ndarray, p: int, q: int, r: int, s: int) -> float:
    """Single spin-orbital integral <pq|rs> read from the spatial tensor."""
    if (p % 2) != (s % 2) or (q % 2) != (r % 2):
        return 0.0
    return h2[p // 2, s // 2, q // 2, r // 2]


def _determinant_energy_spatial(h1: np.ndarray, h2: np.ndarray, occ_so: list[int]) -> float:
    """_determinant_energy evaluated on the spatial integrals.

    The Coulomb term h2_so[p,q,q,p] is always spin-allowed; the exchange term
    h2_so[p,q,p,q] survives only for equal spin -- which is what makes the
    opposite-spin contribution vanish in the tensor formulation. The accumulation
    order matches _determinant_energy so results are bitwise identical (floating
    point addition is not associative; a vectorised sum agrees only to ~1 ulp).
    """
    occ = np.asarray(occ_so, dtype=int)
    spatial = occ // 2

    e_one = np.sum(h1[spatial, spatial])

    e_two = 0.0
    for p in occ:
        for q in occ:
            coulomb = h2[p // 2, p // 2, q // 2, q // 2]
            exchange = h2[p // 2, q // 2, q // 2, p // 2] if (p % 2) == (q % 2) else 0.0
            e_two += coulomb - exchange

    return e_one + 0.5 * e_two


def _compute_a_b_spatial(h1, h2, occ_so, i, j, k, l, e_reference):
    """_compute_a_b on the spatial integrals, reusing the reference energy."""
    occ_exc = _apply_double_excitation(occ_so, k, l, i, j)
    a_val = 0.5 * (e_reference - _determinant_energy_spatial(h1, h2, occ_exc))

    b_val = _h2_so_element(h2, i, j, k, l)
    same_spin = np.unique([index % 2 for index in (k, l, i, j)]).size == 1
    if same_spin:
        b_val -= 0.5 * (_h2_so_element(h2, j, i, k, l) + _h2_so_element(h2, i, j, l, k))

    return a_val, float(np.real(b_val))


def optimal_thetas(h1: np.ndarray, eri: np.ndarray, occ_spatial: list[int],
                   excitation_indices_list) -> list[ExcitationPrediction]:
    """Optimal angles for many double excitations at once.

    Parameters
    ----------
    h1 : np.ndarray
        One-electron matrix elements.
    eri : np.ndarray
        Two-electron matrix elements in chemist order.
    occ_spatial : list[int]
        Spatial orbital indices occupied in the reference state.
    excitation_indices_list : iterable of 4-index sequences
        One entry per excitation, each in TCC/Qiskit-sorted format.

    Returns
    -------
    list[ExcitationPrediction]
        ``(excitation_indices, theta, delta_E)`` sorted by decreasing ``delta_E``.

    Notes
    -----
    ``delta_E`` is compared at full machine precision -- deliberately not rounded, so no
    genuinely distinct excitations are merged. Exact ties (bitwise-equal ``delta_E``) fall
    back to the excitation indices, so the ordering is a pure function of the integrals
    passed in.

    That is NOT the same as being reproducible across runs. Integrals produced by a
    BLAS-threaded AO->MO transform (e.g. :func:`pyscf.ao2mo`) differ by ~1e-11 between
    thread counts, which is enough to swap the order of symmetry-degenerate excitations --
    their ``delta_E`` are equal analytically but not bitwise. A caller that truncates to the
    top-N and needs the SAME N every time (e.g. to keep a circuit structure, and any cache
    keyed on it, stable across processes) must therefore quantise on its own, for instance::

        predictions.sort(key=lambda p: (round(p.delta_E, 8), p.excitation_indices),
                         reverse=True)

    Rounding is left to the caller because the noise originates upstream, in the integrals,
    not in this function.
    """
    no = len(occ_spatial)
    nv = h1.shape[0] - no
    occ_so = sorted({2 * occ for occ in occ_spatial} | {2 * occ + 1 for occ in occ_spatial})

    # Excitation-independent: evaluate once for the whole pool.
    e_reference = _determinant_energy_spatial(h1, eri, occ_so)

    predictions = []
    for excitation_indices in excitation_indices_list:
        v0, v1, o0, o1 = (_block_to_interleaved(idx, no, nv) for idx in excitation_indices)

        assert o0 in occ_so and o1 in occ_so, "expected last two indices occupied"
        assert v0 not in occ_so and v1 not in occ_so, "expected first two indices virtual"

        # Same relabelling as optimal_theta: pair i<->l (=o0,v0) and j<->k (=o1,v1)
        # so the direct integral is spin-allowed.
        i_occ, j_occ = o0, o1
        k_vir, l_vir = v1, v0

        a_val, b_val = _compute_a_b_spatial(h1, eri, occ_so, i_occ, j_occ, k_vir, l_vir,
                                            e_reference)
        theta_opt = -0.5 * np.arctan2(-b_val, -a_val)
        delta_E = a_val + np.sqrt(a_val**2 + b_val**2)
        predictions.append(ExcitationPrediction(tuple(excitation_indices), theta_opt, delta_E))

    # Largest energy impact first. delta_E is compared at full machine precision --
    # deliberately not rounded, so no genuinely distinct excitations are merged. Exact
    # ties (bitwise-equal delta_E, e.g. symmetry-degenerate excitations) fall back to the
    # index tuple, keeping the order reproducible across runs, thread counts and BLAS
    # builds: a caller truncating to the top-N always gets the same N.
    predictions.sort(key=lambda prediction: (-prediction.delta_E, prediction.excitation_indices))
    return predictions


def optimal_thetas_pyscf(mf, excitation_indices_list) -> list[ExcitationPrediction]:
    """Batched :func:`optimal_theta_pyscf`: one AO->MO transform for the whole pool.

    Parameters
    ----------
    mf : pyscf.scf.hf.RHF
        Converged PySCF mean-field object defining the molecular Hamiltonian.
    excitation_indices_list : iterable of 4-index sequences
        One entry per excitation, each in TCC/Qiskit-sorted format.

    Returns
    -------
    list[ExcitationPrediction]
        ``(excitation_indices, theta, delta_E)`` sorted by decreasing ``delta_E``.
    """
    h1_mo, eri_mo, occ_spatial = _transform_integrals_to_mo(mf)
    return optimal_thetas(h1_mo, eri_mo, occ_spatial, excitation_indices_list)
