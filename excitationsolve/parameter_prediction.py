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

from collections.abc import Sequence

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


def _determinant_energy_spatial(h1: np.ndarray, h2: np.ndarray, occ_so: Sequence[int]) -> float:
    """_determinant_energy evaluated on the spatial integrals.

    The Coulomb term h2_so[p,q,q,p] is always spin-allowed; the exchange term
    h2_so[p,q,p,q] survives only for equal spin -- which is what makes the
    opposite-spin contribution vanish in the tensor formulation.
    """
    spatial = np.asarray(occ_so, dtype=int) // 2

    e_one = np.sum(h1[spatial, spatial])
    e_two = 0.0
    for p in occ_so:
        for q in occ_so:
            coulomb = h2[p // 2, p // 2, q // 2, q // 2]
            exchange = h2[p // 2, q // 2, q // 2, p // 2] if (p % 2) == (q % 2) else 0.0
            e_two += coulomb - exchange

    return e_one + 0.5 * e_two


def _compute_a_b_spatial(h1: np.ndarray, h2: np.ndarray, occ_so: Sequence[int],
                         i: int, j: int, k: int, l: int,
                         e_reference: float) -> tuple[float, float]:
    """_compute_a_b on the spatial integrals, reusing the reference energy."""
    occ_exc = _apply_double_excitation(occ_so, k, l, i, j)
    a_val = 0.5 * (e_reference - _determinant_energy_spatial(h1, h2, occ_exc))

    # b = <HF|H|Phi>, matching the derivation
    #     2b = Re(2 h_ijkl - [h_jikl + h_ijlk] delta_{mu nu})
    # i.e. b = Re( h_ijkl - 1/2 (h_jikl + h_ijlk) delta_{mu nu} ).
    # The spin-orbital elements are read straight from the spatial integrals via
    # h2_so[p,q,r,s] = h2[p//2, s//2, q//2, r//2]; the spin guard of the tensor form is not
    # needed because optimal_thetas validates that the excitation is spin-conserving and
    # relabelled so the direct term is allowed (spin(i)==spin(l), spin(j)==spin(k)). The two
    # exchange terms then only contribute when all four spin-orbitals share a spin, which is
    # exactly the delta_{mu nu}.
    b_val = h2[i // 2, l // 2, j // 2, k // 2]
    same_spin = k % 2 == l % 2 == i % 2 == j % 2
    if same_spin:
        b_val -= 0.5 * (h2[j // 2, l // 2, i // 2, k // 2] + h2[i // 2, k // 2, j // 2, l // 2])

    # np.real implements the Re() of the derivation above -- it is part of the formula, not a
    # defensive cast. It is a no-op for real MO integrals, but float() alone would silently
    # discard an imaginary part (with a ComplexWarning) if complex orbitals were ever used.
    return a_val, float(np.real(b_val))



# ---------------------------------------------------------------------------
# Batched prediction. Performing parameter prediction on a batch of operators instead
# of individually reduces the overhead of computing integrals and the HF state again and again.  
#
# The integrals are also indexed directly instead of being expanded into a full
# spin-orbital tensor.
#
# Index convention:
#   spin-orbital index p -> spatial p // 2, spin p % 2
#   h2_so[p, q, r, s] = h2[p//2, s//2, q//2, r//2]
#                       if spin(p) == spin(s) and spin(q) == spin(r) else 0
# ---------------------------------------------------------------------------
def optimal_thetas(h1: np.ndarray, eri: np.ndarray, occ_spatial: Sequence[int],
                   excitation_indices_list: Sequence[Sequence[int]]) -> list[tuple[float, float]]:
    """Optimal angles for many double excitations at once.

    Parameters
    ----------
    h1 : np.ndarray
        One-electron matrix elements.
    eri : np.ndarray
        Two-electron matrix elements in chemist order.
    occ_spatial : Sequence[int]
        Spatial orbital indices occupied in the reference state.
    excitation_indices_list : Sequence[Sequence[int]]
        One entry per excitation, each four spin-orbital indices in TCC/Qiskit-sorted format.
        Tuples, lists and 2-D arrays are all accepted.

    Returns
    -------
    list[tuple[float, float]]
        ``(theta, delta_E)`` per excitation, in the SAME ORDER as the input. Each entry is
        what the previous per-excitation implementation returned for that excitation.

    Notes
    -----
    Results are returned unsorted, in input order, and ``delta_E`` is not rounded. Callers
    that rank the pool and truncate to the top-N should be aware that integrals from a
    BLAS-threaded AO->MO transform (e.g. :func:`pyscf.ao2mo`) differ by ~1e-11 between thread
    counts -- enough to swap symmetry-degenerate excitations, whose ``delta_E`` are equal
    analytically but not bitwise. If the selected set must be identical across processes (for
    instance to keep a circuit structure, and any cache keyed on it, stable), quantise when
    sorting::

        order = sorted(range(len(dE)), key=lambda n: (round(dE[n], 8), pool[n]), reverse=True)

    Excitations must be spin-conserving; violations raise ``ValueError``.
    """
    no = len(occ_spatial)
    nv = h1.shape[0] - no
    occ_so = sorted(2 * occ + spin for occ in occ_spatial for spin in (0, 1))
    occ_so_set = set(occ_so)   # membership only; the sorted list keeps the summation order

    # Excitation-independent: evaluate once for the whole pool.
    e_reference = _determinant_energy_spatial(h1, eri, occ_so)

    predictions = []
    for excitation_indices in excitation_indices_list:
        # TCC/Qiskit order is (virt, virt, occ, occ), sorted within each pair, which guarantees
        # spin(virt_0) == spin(occ_0) and spin(virt_1) == spin(occ_1). The equation labels
        # (i, j, k, l) must pair i<->l and j<->k on the SAME spin for the direct integral
        # h_ijkl to be spin-allowed, so the first virtual is l and the second is k -- the swap
        # is applied here in the unpacking.
        l_vir, k_vir, i_occ, j_occ = (_block_to_interleaved(idx, no, nv)
                                      for idx in excitation_indices)

        if i_occ not in occ_so_set or j_occ not in occ_so_set:
            raise ValueError(f"expected last two indices occupied: {excitation_indices}")
        if l_vir in occ_so_set or k_vir in occ_so_set:
            raise ValueError(f"expected first two indices virtual: {excitation_indices}")
        if (l_vir % 2) != (i_occ % 2) or (k_vir % 2) != (j_occ % 2):
            raise ValueError("expected a spin-conserving excitation (TCC sorting pairs the "
                             f"first virtual with the first occupied): {excitation_indices}")

        a_val, b_val = _compute_a_b_spatial(h1, eri, occ_so, i_occ, j_occ, k_vir, l_vir,
                                            e_reference)
        theta_opt = -0.5 * np.arctan2(-b_val, -a_val)
        delta_E = a_val + np.sqrt(a_val**2 + b_val**2)
        predictions.append((theta_opt, delta_E))

    return predictions


def optimal_thetas_pyscf(mf,
                         excitation_indices_list: Sequence[Sequence[int]]
                         ) -> list[tuple[float, float]]:
    """Batched :func:`optimal_theta_pyscf`: one AO->MO transform for the whole pool.

    Parameters
    ----------
    mf : pyscf.scf.hf.RHF
        Converged PySCF mean-field object defining the molecular Hamiltonian.
    excitation_indices_list : Sequence[Sequence[int]]
        One entry per excitation, each four spin-orbital indices in TCC/Qiskit-sorted format.
        Tuples, lists and 2-D arrays are all accepted.

    Returns
    -------
    list[tuple[float, float]]
        ``(theta, delta_E)`` per excitation, in the SAME ORDER as the input.
    """
    h1_mo, eri_mo, occ_spatial = _transform_integrals_to_mo(mf)
    return optimal_thetas(h1_mo, eri_mo, occ_spatial, excitation_indices_list)


def optimal_theta(h1: np.ndarray, eri: np.ndarray, occ_spatial: Sequence[int],
                  excitation_indices: Sequence[int]) -> tuple[float, float]:
    """Optimal angle and energy impact for a SINGLE double excitation.

    Thin wrapper around :func:`optimal_thetas`. Prefer the batched function when scoring more
    than one excitation: it shares the reference-determinant energy across the pool, so calling
    this in a loop repeats that work per excitation.

    Parameters
    ----------
    h1 : np.ndarray
        One-electron matrix elements.
    eri : np.ndarray
        Two-electron matrix elements in chemist order.
    occ_spatial : Sequence[int]
        Spatial orbital indices occupied in the reference state.
    excitation_indices : Sequence[int]
        Four spin-orbital indices in TCC/Qiskit-sorted format.

    Returns
    -------
    theta_opt : float
        Exact optimal angle (in radians) minimising E(theta).
    delta_E : float
        Maximum energy impact a + sqrt(a^2 + b^2).
    """
    return optimal_thetas(h1, eri, occ_spatial, [excitation_indices])[0]


def optimal_theta_pyscf(mf, excitation_indices: Sequence[int]) -> tuple[float, float]:
    """Optimal angle and energy impact for a SINGLE double excitation, from a PySCF object.

    Thin wrapper around :func:`optimal_thetas_pyscf`. Prefer the batched function when scoring
    more than one excitation: it performs the AO->MO transform once for the whole pool, so
    calling this in a loop repeats that transform per excitation.

    Parameters
    ----------
    mf : pyscf.scf.hf.RHF
        Converged PySCF mean-field object defining the molecular Hamiltonian.
    excitation_indices : Sequence[int]
        Four spin-orbital indices in TCC/Qiskit-sorted format.

    Returns
    -------
    theta_opt : float
        Exact optimal angle (in radians) minimising E(theta).
    delta_E : float
        Maximum energy impact a + sqrt(a^2 + b^2).
    """
    return optimal_thetas_pyscf(mf, [excitation_indices])[0]
