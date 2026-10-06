"""Closed-form prediction of the optimal VQE angle of a single or double excitation.

For a single- or double-excitation generator G = a^dag_k a_i (single) or
G = a^dag_k a^dag_l a_i a_j (double) applied to an RHF/ROHF reference, the energy as a
function of the rotation angle theta is the period-pi curve

    E(theta) = const + a (1 - cos 2theta) + b sin 2theta,

with a = (E_HF - E_exc) / 2 and b = <HF|H|Phi>. The minimiser of this curve has
the closed form theta = -1/2 * arctan2(-b, -a).

For more information, see https://arxiv.org/abs/2602.10776
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _transform_integrals_to_mo(mf: Any) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """
    Transform one- and two-electron integrals from AO to MO basis.

    Parameters
    ----------
    mf : pyscf.scf.hf.RHF or pyscf.scf.rohf.ROHF
        Converged PySCF mean-field object defining the molecular Hamiltonian.

    Returns
    -------
    h1_mo : np.ndarray, shape (nmo, nmo)
        One-electron matrix elements over spatial MOs.
    eri_mo : np.ndarray, shape (nmo, nmo, nmo, nmo)
        Two-electron matrix elements over spatial MOs, in chemist order.
    occ_indices : list[int]
        Alpha-spin spatial MO indices occupied in the reference state.
    occ_indices_beta : list[int]
        Beta-spin spatial MO indices occupied in the reference state. For RHF this equals
        occ_indices; for ROHF, PySCF puts singly occupied orbitals in the alpha spin, so
        occ_indices_beta excludes them (mo_occ > 1 rather than mo_occ > 0).
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
    occ_indices_beta = np.where(mf.mo_occ > 1.0)[0].tolist()
    return h1_mo, eri_mo, occ_indices, occ_indices_beta


def _excited_energy(h_diag: np.ndarray, G: np.ndarray, S: np.ndarray, e_reference: float, i: int, j: int, k: int, l: int) -> float:
    """Calculates E_excited(Phi) = <Phi|H|Phi>, Phi being the excited Slater determinant, from
    the reference energy E(HF) = <HF|H|HF>.

    Spin-orbital p occupies spatial orbital p_ = p // 2 with spin p % 2

    E_excited(Phi) calculated using Slater-Condon rules:

        E_excited(Phi) - E(HF)=
      h[k] + h[l] - h[i] - h[j]              (one-electron part)
      + S[k] + S[l] - S[i] - S[j]            (mean field; together with the line above this is
                                              just eps_k + eps_l - eps_i - eps_j, the
                                              orbital-energy difference)
      - G[i,k] - G[i,l] - G[j,k] - G[j,l]    (S counted i and j, which are no longer occupied:
                                              remove the k,l interactions with them ...)
      + G[i,j] + G[k,l]                      (... and fix the two pairs among the moved
                                              orbitals themselves. These G terms are the
                                              correction for the mean field not being frozen.)

    h[p] is the one-electron energy of spin-orbital p:     h[p]   = h1[p_, p_]

    G[p,q] is the Coulomb minus (same-spin) exchange interaction of the pair (p, q):
        G[p,q] = h2[p_, p_, q_, q_] - h2[p_, q_, q_, p_] if spin(p) == spin(q)
                 else h2[p_, p_, q_, q_]

    S[p] = sum over the OCCUPIED q of G[p,q] is the mean field

    Parameters
    ----------
    h_diag : np.ndarray, shape (nso,)
        One-electron energy h[p] of every spin-orbital.
    G : np.ndarray, shape (nso, nso)
        Pair interaction G[p,q].
    S : np.ndarray, shape (nso,)
        Mean field S[p].
    e_reference : float
        Reference determinant energy E(HF), evaluated once per pool by :func:`optimal_thetas`.
    i, j : int
        Occupied spin-orbitals removed by the excitation.
    k, l : int
        Virtual spin-orbitals filled by the excitation.

    Returns
    -------
    e_excited : float
        Excited state determinant energy.
    """
    return e_reference + (
        h_diag[k] + h_diag[l] - h_diag[i] - h_diag[j] + S[k] + S[l] - S[i] - S[j] - G[i, k] - G[i, l] - G[j, k] - G[j, l] + G[i, j] + G[k, l]
    )


def _excited_energy_single(h_diag: np.ndarray, G: np.ndarray, S: np.ndarray, e_reference: float, i: int, k: int) -> float:
    """Calculates E_excited(Phi) = <Phi|H|Phi>, Phi being the excited Slater determinant, from
    the reference energy E(HF) = <HF|H|HF>.

    Spin-orbital p occupies spatial orbital p_ = p // 2 with spin p % 2

    E_excited(Phi) calculated using Slater-Condon rules:

        E_excited(Phi) - E(HF) =
      h[k] - h[i]                            (one-electron part)
      + S[k] - S[i]                          (mean field; together with the line above this is
                                              just eps_k - eps_i, the orbital-energy difference)
      - G[i,k]                               (S counted i, which is no longer occupied: remove
                                              the k interaction with it. There is no second pair
                                              to fix, unlike the double excitation, since only
                                              one orbital moved.)

    h[p] is the one-electron energy of spin-orbital p:     h[p]   = h1[p_, p_]

    G[p,q] is the Coulomb minus (same-spin) exchange interaction of the pair (p, q):
        G[p,q] = h2[p_, p_, q_, q_] - h2[p_, q_, q_, p_] if spin(p) == spin(q)
                 else h2[p_, p_, q_, q_]

    S[p] = sum over the OCCUPIED q of G[p,q] is the mean field

    Parameters
    ----------
    h_diag : np.ndarray, shape (nso,)
        One-electron energy h[p] of every spin-orbital.
    G : np.ndarray, shape (nso, nso)
        Pair interaction G[p,q].
    S : np.ndarray, shape (nso,)
        Mean field S[p].
    e_reference : float
        Reference determinant energy E(HF), evaluated once per pool by :func:`optimal_thetas`.
    i : int
        Occupied spin-orbital emptied by the excitation.
    k : int
        Virtual spin-orbital filled by the excitation.

    Returns
    -------
    e_excited : float
        Excited state determinant energy.
    """
    return e_reference + (h_diag[k] - h_diag[i] + S[k] - S[i] - G[i, k])


def _compute_a_b(
    h2: np.ndarray, h_diag: np.ndarray, G: np.ndarray, S: np.ndarray, i: int, j: int, k: int, l: int, e_reference: float
) -> tuple[float, float]:
    """Two-level parameters a and b of one double excitation, from the spatial integrals.

    The excitation empties the occupied spin-orbitals (i, j) and fills the virtual ones (k, l)
    via G = a^dag_k a^dag_l a_i a_j, so that |Phi> = G|HF> and

        a = ( <HF|H|HF> - <Phi|H|Phi> ) / 2,
        b = <HF|H|Phi> = Re( h_ijkl - 1/2 (h_jikl + h_ijlk) delta_{mu nu} )

    Parameters
    ----------
    h2 : np.ndarray
        Two-electron matrix elements over SPATIAL orbitals, in chemist order.
    h_diag, G, S : np.ndarray
        One-electron energies, pair interactions and mean field, as built once per pool by
        :func:`optimal_thetas`.
    i, j : int
        Occupied spin-orbitals removed by the excitation.
    k, l : int
        Virtual spin-orbitals filled by the excitation.
    e_reference : float
        <HF|H|HF>, evaluated once per pool by :func:`optimal_thetas`.

    Returns
    -------
    a_val : float
    b_val : float
    """

    e_excited = _excited_energy(h_diag, G, S, e_reference, i, j, k, l)

    a_val = 0.5 * (e_reference - e_excited)
    b_val = h2[i // 2, l // 2, j // 2, k // 2]
    same_spin = k % 2 == l % 2 == i % 2 == j % 2
    if same_spin:
        b_val -= 0.5 * (h2[j // 2, l // 2, i // 2, k // 2] + h2[i // 2, k // 2, j // 2, l // 2])

    return a_val, float(np.real(b_val))


def _compute_a_b_single(
    h1: np.ndarray, h2: np.ndarray, h_diag: np.ndarray, G: np.ndarray, S: np.ndarray, occ: np.ndarray, i: int, k: int, e_reference: float
) -> tuple[float, float]:
    """Two-level parameters a and b of one single excitation, from the spatial integrals.

    The excitation empties the occupied spin-orbital i and fills the virtual one k via
    G = a^dag_k a_i, so that |Phi> = G|HF> and

        a = ( <HF|H|HF> - <Phi|H|Phi> ) / 2,
        b = -<HF|H|Phi> = -F_ik, with F_ik the (i, k) element of the spin-orbital Fock matrix

    The minus sign matches the TCC convention: the TCC single a^dag_k a_i carries the opposite
    sign relative to the double, whose virtual indices are swapped in :func:`optimal_thetas`.

    Parameters
    ----------
    h1 : np.ndarray
        One-electron matrix elements over SPATIAL orbitals.
    h2 : np.ndarray
        Two-electron matrix elements over SPATIAL orbitals, in chemist order.
    h_diag, G, S : np.ndarray
        One-electron energies, pair interactions and mean field, as built once per pool by
        :func:`optimal_thetas`.
    occ : np.ndarray
        Occupied spin-orbitals, as built once per pool by :func:`optimal_thetas`.
    i : int
        Occupied spin-orbital emptied by the excitation.
    k : int
        Virtual spin-orbital filled by the excitation.
    e_reference : float
        <HF|H|HF>, evaluated once per pool by :func:`optimal_thetas`.

    Returns
    -------
    a_val : float
    b_val : float
    """

    e_excited = _excited_energy_single(h_diag, G, S, e_reference, i, k)

    a_val = 0.5 * (e_reference - e_excited)

    i_, k_ = i // 2, k // 2
    p = occ[occ != i]
    p_ = p // 2
    same_spin = (p % 2) == (i % 2)
    b_val = -(h1[i_, k_] + h2[p_, p_, i_, k_].sum() - np.where(same_spin, h2[p_, k_, i_, p_], 0.0).sum())

    return a_val, float(np.real(b_val))


def _block_to_interleaved(idx: int, no: int, nv: int) -> int:
    """Map a TCC block-ordered spin-orbital index to the interleaved
    (2*spatial + spin) convention used for the spin-orbital integrals.

    TCC orders spin-orbitals as [beta_occ, beta_virt, alpha_occ, alpha_virt].
    spin: 0 = alpha, 1 = beta.

    Parameters
    ----------
    idx : int
        Spin-orbital index in TCC block order.
    no, nv : int
        Number of occupied and of virtual SPATIAL orbitals.

    Returns
    -------
    int
        The same spin-orbital as 2 * spatial + spin.
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


def optimal_thetas(
    h1: np.ndarray,
    eri: np.ndarray,
    occ_spatial: Sequence[int],
    excitation_indices_list: Sequence[Sequence[int]],
    occ_spatial_beta: Sequence[int] | None = None,
) -> list[tuple[float, float]]:
    """Optimal angles for many single or double excitations at once.

    Parameters
    ----------
    h1 : np.ndarray
        One-electron matrix elements.
    eri : np.ndarray
        Two-electron matrix elements in chemist order.
    occ_spatial : Sequence[int]
        Alpha-spin spatial orbital indices occupied in the reference state.
    excitation_indices_list : Sequence[Sequence[int]]
        One entry per excitation, each two (single) or four (double) spin-orbital indices in
        TCC/Qiskit-sorted format (virt, occ) or (virt, virt, occ, occ), sorted within each pair.
        Tuples, lists and 2-D arrays are all accepted.
    occ_spatial_beta : Sequence[int] | None, optional
        Beta-spin spatial orbital indices occupied in the reference state. Defaults to
        ``occ_spatial`` (closed-shell reference); pass a different list for an ROHF open-shell
        reference, where PySCF's singly occupied orbitals are alpha-only.

    Returns
    -------
    list[tuple[float, float]]
        ``(theta, delta_E)`` per excitation, in the SAME ORDER as the input. Each entry is
        what the previous per-excitation implementation returned for that excitation.

    Notes
    -----
    Callers that rank the pool and truncate to the top-N should be aware that integrals from
    :func:`pyscf.ao2mo` differ within numerical accuracy between calls, possibly swapping
    excitations order.

    Excitations must be spin-conserving; violations raise ``ValueError``.
    """
    if occ_spatial_beta is None:
        occ_spatial_beta = occ_spatial
    no = len(occ_spatial)
    nv = h1.shape[0] - no
    occ_so = sorted([2 * occ for occ in occ_spatial] + [2 * occ + 1 for occ in occ_spatial_beta])
    occ_so_set = set(occ_so)  # membership only; the sorted list keeps the summation order

    # Excitation-independent: build the one- and two-body tables once for the whole pool.
    nso = 2 * h1.shape[0]
    spatial = np.arange(nso) // 2
    spin = np.arange(nso) % 2

    h_diag = h1[spatial, spatial]
    rows, cols = spatial[:, None], spatial[None, :]
    coulomb = eri[rows, rows, cols, cols]
    exchange = np.where(spin[:, None] == spin[None, :], eri[rows, cols, cols, rows], 0.0)
    G = coulomb - exchange
    occ = np.asarray(occ_so, dtype=int)
    S = G[:, occ].sum(axis=1)
    e_reference = float(h_diag[occ].sum() + 0.5 * S[occ].sum())

    predictions = []
    for excitation_indices in excitation_indices_list:
        if len(excitation_indices) == 2:
            # TCC order is (virt, occ), giving G = a^dag_k a_i directly, no swap needed.
            k_vir, i_occ = (_block_to_interleaved(idx, no, nv) for idx in excitation_indices)

            if i_occ not in occ_so_set:
                raise ValueError(f"expected second index occupied: {excitation_indices}")
            if k_vir in occ_so_set:
                raise ValueError(f"expected first index virtual: {excitation_indices}")
            if (k_vir % 2) != (i_occ % 2):
                raise ValueError(f"expected a spin-conserving excitation: {excitation_indices}")

            a_val, b_val = _compute_a_b_single(h1, eri, h_diag, G, S, occ, i_occ, k_vir, e_reference)
        else:
            # TCC/Qiskit order is (virt, virt, occ, occ), sorted within each pair, which guarantees
            # spin(virt_0) == spin(occ_0) and spin(virt_1) == spin(occ_1). The equation labels
            # (i, j, k, l) must pair i<->l and j<->k on the SAME spin for the direct integral
            # h_ijkl to be spin-allowed, so the first virtual is l and the second is k -- the swap
            # is applied here in the unpacking.
            l_vir, k_vir, i_occ, j_occ = (_block_to_interleaved(idx, no, nv) for idx in excitation_indices)

            if i_occ not in occ_so_set or j_occ not in occ_so_set:
                raise ValueError(f"expected last two indices occupied: {excitation_indices}")
            if l_vir in occ_so_set or k_vir in occ_so_set:
                raise ValueError(f"expected first two indices virtual: {excitation_indices}")
            if (l_vir % 2) != (i_occ % 2) or (k_vir % 2) != (j_occ % 2):
                raise ValueError(
                    f"expected a spin-conserving excitation (TCC sorting pairs the first virtual with the first occupied): {excitation_indices}"
                )

            a_val, b_val = _compute_a_b(eri, h_diag, G, S, i_occ, j_occ, k_vir, l_vir, e_reference)

        theta_opt = -0.5 * np.arctan2(-b_val, -a_val)
        delta_E = a_val + np.sqrt(a_val**2 + b_val**2)
        predictions.append((theta_opt, delta_E))

    return predictions


def optimal_theta(
    h1: np.ndarray,
    eri: np.ndarray,
    occ_spatial: Sequence[int],
    excitation_indices: Sequence[int],
    occ_spatial_beta: Sequence[int] | None = None,
) -> tuple[float, float]:
    """Optimal angle and energy impact for ONE single or double excitation.

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
        Alpha-spin spatial orbital indices occupied in the reference state.
    excitation_indices : Sequence[int]
        Two (single) or four (double) spin-orbital indices in TCC/Qiskit-sorted format.
    occ_spatial_beta : Sequence[int] | None, optional
        Beta-spin spatial orbital indices occupied in the reference state. Defaults to
        ``occ_spatial`` (closed-shell reference); pass a different list for an ROHF open-shell
        reference, where PySCF's singly occupied orbitals are alpha-only.

    Returns
    -------
    theta_opt : float
        Exact optimal angle (in radians) minimising E(theta).
    delta_E : float
        Maximum energy impact a + sqrt(a^2 + b^2).
    """
    return optimal_thetas(h1, eri, occ_spatial, [excitation_indices], occ_spatial_beta)[0]


def optimal_thetas_pyscf(mf: Any, excitation_indices_list: Sequence[Sequence[int]]) -> list[tuple[float, float]]:
    """Batched :func:`optimal_theta_pyscf`: one AO->MO transform for the whole pool.

    Parameters
    ----------
    mf : pyscf.scf.hf.RHF or pyscf.scf.rohf.ROHF
        Converged PySCF mean-field object defining the molecular Hamiltonian.
    excitation_indices_list : Sequence[Sequence[int]]
        One entry per excitation, each two (single) or four (double) spin-orbital indices in
        TCC/Qiskit-sorted format. Tuples, lists and 2-D arrays are all accepted.

    Returns
    -------
    list[tuple[float, float]]
        ``(theta, delta_E)`` per excitation, in the SAME ORDER as the input.
    """
    h1_mo, eri_mo, occ_spatial, occ_spatial_beta = _transform_integrals_to_mo(mf)
    return optimal_thetas(h1_mo, eri_mo, occ_spatial, excitation_indices_list, occ_spatial_beta)


def optimal_theta_pyscf(mf: Any, excitation_indices: Sequence[int]) -> tuple[float, float]:
    """Optimal angle and energy impact for ONE single or double excitation, from a PySCF object.

    Thin wrapper around :func:`optimal_thetas_pyscf`. Prefer the batched function when scoring
    more than one excitation: it performs the AO->MO transform once for the whole pool, so
    calling this in a loop repeats that transform per excitation.

    Parameters
    ----------
    mf : pyscf.scf.hf.RHF or pyscf.scf.rohf.ROHF
        Converged PySCF mean-field object defining the molecular Hamiltonian.
    excitation_indices : Sequence[int]
        Two (single) or four (double) spin-orbital indices in TCC/Qiskit-sorted format.

    Returns
    -------
    theta_opt : float
        Exact optimal angle (in radians) minimising E(theta).
    delta_E : float
        Maximum energy impact a + sqrt(a^2 + b^2).
    """
    return optimal_thetas_pyscf(mf, [excitation_indices])[0]
