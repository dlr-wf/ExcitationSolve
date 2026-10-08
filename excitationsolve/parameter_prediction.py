r"""Closed-form prediction of the optimal VQE angle of a single or double excitation.

For a single- or double-excitation generator G = a^dag_j a_i (single) or
G = a^dag_k a^dag_l a_i a_j (double) applied to an RHF/ROHF reference, the energy as a
function of the rotation angle theta is the period-pi curve

    E(theta) = const + a (1 - cos 2theta) + b sin 2theta,

with a = (E_exc - E_HF) / 2 and b = <HF|H|Phi>. The minimiser of this curve has
the closed form theta = 1/2 * arctan2(-b, a).

We implement the formulas from https://arxiv.org/abs/2602.10776 but use auxiliary matrix elements.
Using the index order h_ijkl = ∬ φ_i*(r1) φ*_j(r2) φ_k(r2) φ_l(r1) / |r1 - r2| dr1 dr2,
we define
    G_{ij}^{\sigma\mu} = h_{ijji}^{\sigma\mu} - h_{ijij}^{\sigma\mu} \delta_{\sigma\mu}
with G_{ii}^{\sigma\sigma} = 0 and G_{ij}^{\sigma\mu} = G_{ji}^{\mu\sigma},
and
    S_q^\sigma = \sum_{p^\mu \in o} G_{qp}^{\sigma\mu} = \sum_{p^\mu \in \bm o \cap \bm o'} G_{qp}^{\sigma\mu} + G_{qi}^{\sigma\mu} + G_{qj}^{\sigma\nu}
for a double excitation from i^\mu, j^\nu to k^\nu, l^\mu.
With this we can write
    \sum_{p^\mu \in \bm o \cap \bm o'} G_{qp}^{\sigma\mu} = S_q^\sigma - G_{qi}^{\sigma\mu} - G_{qj}^{\sigma\nu}

For double excitations acting on the Hartree-Fock state we obtain for the difference between excited energy and HF-energy:
    \braket{\psi_{i^\mu j^\nu}^{k^\nu l^\mu} | H | \psi_{i^\mu j^\nu}^{k^\nu l^\mu}} - \braket{\psi| H | \psi} =
        h_{kk} + h_{ll} - h_{ii} - h_{jj} + G_{kl} - G_{ij}
            +  S_{k} - G_{ki} - G_{kj}
            +  S_{l} - G_{li} - G_{lj}
            -  S_{i} + G_{ii} + G_{ij}
            -  S_{j} + G_{ji} + G_{jj}
        = h_{kk} + h_{ll} - h_{ii} - h_{jj} + S_{k} + S_{l} - S_{i} - S_{j} - G_{ik} - G_{il} - G_{jk} - G_{jl} + G_{ij} + G_{kl}
where we omitted the spin symbols and used G_{ii} = 0

For single excitations from i^\mu to j^\mu acting on a single Slater determinant we obtain for the difference between excited energy and HF-energy:
    \braket{\psi_{i^\mu}^{j^\mu} | H | \psi_{i^\mu}^{j^\mu}} - \braket{\psi| H | \psi} =
        h_{jj} - h_{ii} + S_{j} - S_{i} - G_{ij}
where \sum_{p^\sigma \in \bm o \cap \bm o'} G_{qp}^{\mu\sigma} = S_q^\mu - G_{qi}^{\mu\mu}

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


def _compute_a_b(
    h2: np.ndarray, h_diag: np.ndarray, G: np.ndarray, S: np.ndarray, i: int, j: int, k: int, l: int, e_reference: float
) -> tuple[float, float]:
    """Compute the parameters a and b of one double excitation, from the spatial integrals.

    The excitation empties the occupied spin-orbitals (i, j) and fills the virtual ones (k, l)
    via G = a^dag_k a^dag_l a_i a_j, so that |Phi> = G|HF> and

        a = ( <Phi|H|Phi> - <HF|H|HF> ) / 2,
        b = <HF|H|Phi> = Re( h_ijkl - 1/2 (h_jikl + h_ijlk) delta_{mu nu} )

    Parameters
    ----------
    h2 : np.ndarray
        Two-electron matrix elements in chemist order: h_ijkl = ∬ φ_i*(r1) φ_j(r1) φ_k*(r2) φ_l(r2) / |r1 - r2| dr1 dr2
        Shape (N, N, N, N), where N is the number of spatial orbitals.
    h_diag : np.ndarray
        Diagonal one-electron matrix-elements h_diag[i] = h1[i,i] where h1 is the one-electron matrix with shape (2N, 2N) and i is a spin-orbital index.
        If h1, as defined in optimal_theta... functions, is the one-electron matrix with shape (N, N), then h_diag[i] = h1[i//2, i//2].
        Shape (2N,).
    G : np.ndarray
        Coulomb minus exchange G[i,j] = coulomb[i,j] - exchange[i,j] = h2[i//2, i//2, j//2, j//2] - h2[i//2, j//2, j//2, i//2] if spin(i) == spin(j)
            else h2[i//2, i//2, j//2, j//2].
        Shape (2N, 2N).
    S : np.ndarray
        G summed over occupied orbitals.
        S[i] =  ∑_{j ∈ occ} G[i,j] = ∑_{j ∈ occ} ( h2[i//2, i//2, j//2, j//2] - h2[i//2, j//2, j//2, i//2] ) if spin(i) == spin(j)
            else ∑_{j ∈ occ} h2[i//2, i//2, j//2, j//2].
        Shape (2N,).
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
    # E_excited(Phi) - E(HF) =
    #   h[k] + h[l] - h[i] - h[j]
    #   + S[k] + S[l] - S[i] - S[j]
    #   - G[i,k] - G[i,l] - G[j,k] - G[j,l]
    #   + G[i,j] + G[k,l]
    # See the docstring at the beginning of this file for a derivation how this formula relates to the formulas in https://arxiv.org/abs/2602.10776
    a_val = 0.5 * (
        h_diag[k] + h_diag[l] - h_diag[i] - h_diag[j] + S[k] + S[l] - S[i] - S[j] - G[i, k] - G[i, l] - G[j, k] - G[j, l] + G[i, j] + G[k, l]
    )

    b_val = h2[i // 2, l // 2, j // 2, k // 2]
    same_spin = k % 2 == l % 2 == i % 2 == j % 2
    if same_spin:
        b_val -= 0.5 * (h2[j // 2, l // 2, i // 2, k // 2] + h2[i // 2, k // 2, j // 2, l // 2])

    return a_val, float(np.real(b_val))


def _compute_a_b_single(
    h1: np.ndarray, h2: np.ndarray, h_diag: np.ndarray, G: np.ndarray, S: np.ndarray, occ: np.ndarray, i: int, j: int, e_reference: float
) -> tuple[float, float]:
    r"""Compute the parameters a and b of one single excitation, from the spatial integrals.

    The excitation empties the occupied spin-orbital i and fills the virtual orbital j
    via G = a^dag_j a_i, so that |Phi> = G|HF> and

        a = ( <Phi|H|Phi> - <HF|H|HF> ) / 2,
        b = <HF|H|Phi> = Re( h_ij +  \sum_{p in o \cap o'} h^\sigma\mu_pijp - h^\sigma\mu_pipj \delta_\sigma\mu )

    Parameters
    ----------
    h1 : np.ndarray
        One-electron matrix elements. Shape (N, N) where N is the number of spatial orbitals.
    h2 : np.ndarray
        Two-electron matrix elements in chemist order: h_ijkl = ∬ φ_i*(r1) φ_j(r1) φ_k*(r2) φ_l(r2) / |r1 - r2| dr1 dr2
        Shape (N, N, N, N), where N is the number of spatial orbitals.
    h_diag : np.ndarray
        Diagonal one-electron matrix-elements h_diag[i] = h1[i,i] where h1 is the one-electron matrix with shape (2N, 2N) and i is a spin-orbital index.
        If h1, as defined in optimal_theta... functions, is the one-electron matrix with shape (N, N), then h_diag[i] = h1[i//2, i//2].
        Shape (2N,).
    G : np.ndarray
        Coulomb minus exchange G[i,j] = coulomb[i,j] - exchange[i,j] = h2[i//2, i//2, j//2, j//2] - h2[i//2, j//2, j//2, i//2] if spin(i) == spin(j)
            else h2[i//2, i//2, j//2, j//2].
        Shape (2N, 2N).
    S : np.ndarray
        G summed over occupied orbitals.
        S[i] =  ∑_{j ∈ occ} G[i,j] = ∑_{j ∈ occ} ( h2[i//2, i//2, j//2, j//2] - h2[i//2, j//2, j//2, i//2] ) if spin(i) == spin(j)
            else ∑_{j ∈ occ} h2[i//2, i//2, j//2, j//2].
        Shape (2N,).
    occ : np.ndarray
        Occupied spin-orbitals.
    i : int
        Occupied spin-orbital removed by the excitation.
    j : int
        Virtual spin-orbital filled by the excitation.
    e_reference : float
        <HF|H|HF>, evaluated once per pool by :func:`optimal_thetas`.

    Returns
    -------
    a_val : float
    b_val : float
    """

    # E_excited(Phi) - E(HF) = h_{jj} - h_{ii} + S_{j} - S_{i} - G_{ij}
    # See the docstring at the beginning of this file for a derivation how this formula relates to the formulas in https://arxiv.org/abs/2602.10776
    a_val = 0.5 * (h_diag[j] - h_diag[i] + S[j] - S[i] - G[i, j])

    p = occ[occ != i]
    same_spin = (p % 2) == (i % 2)
    b_val = h1[i // 2, j // 2] + h2[p // 2, p // 2, i // 2, j // 2].sum() - np.where(same_spin, h2[p // 2, j // 2, i // 2, p // 2], 0.0).sum()

    return a_val, float(np.real(b_val))


def _block_to_interleaved(idx: int, no: int, nv: int) -> int:
    """Map a TCC block-ordered spin-orbital index to the interleaved
    (2*spatial + spin) convention used for the spin-orbital integrals.

    TCC orders spin-orbitals as [beta_occ, beta_virt, alpha_occ, alpha_virt].
    spin: 0 = alpha, 1 = beta.

    This function transforms this TCC order into
        [beta_occ_1, alpha_occ_1, beta_occ_2, alpha_occ_2, ..., beta_virt_1, alpha_virt_1, ...]

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
    r"""Optimal angles for many single or double excitations at once.

    Parameters
    ----------
    h1 : np.ndarray
        One-electron matrix elements. Shape (N, N) where N is the number of spatial orbitals.
    eri : np.ndarray
        Two-electron matrix elements in chemist order: h_ijkl = ∬ φ_i*(r1) φ_j(r1) φ_k*(r2) φ_l(r2) / |r1 - r2| dr1 dr2
        Shape (N, N, N, N) where N is the number of spatial orbitals.
    occ_spatial : Sequence[int]
        Alpha-spin spatial orbital indices occupied in the reference state.
    excitation_indices_list : Sequence[Sequence[int]]
        One entry per excitation:
        (k, l, i, j) corresponds to a^\dagger_k a^\dagger_l a_i a_j - a^\dagger_j a^\dagger_i a_l a_k
        and
        (j, i) corresponds to a^\dagger_j a_i - a^\dagger_i a_j
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
    occ_so_set = set(occ_so)
    occ_so_array = np.asarray(occ_so, dtype=int)

    # Excitation-independent: build the one- and two-body tables once for the whole pool.
    n_spin_orbitals = 2 * h1.shape[0]
    spatial_orbital_idc = np.arange(n_spin_orbitals) // 2
    spin = np.arange(n_spin_orbitals) % 2

    # The //2 used in the comments below only comes from the fact that h1 and eri are defined over spatial orbitals instead of spin orbitals

    h_diag = h1[spatial_orbital_idc, spatial_orbital_idc]  # Shape (2N,), h_diag[i] = h1[i//2, i//2]
    rows, cols = spatial_orbital_idc[:, None], spatial_orbital_idc[None, :]
    coulomb = eri[rows, rows, cols, cols]  # Shape (2N, 2N), coulomb[i,j] = eri[i//2 ,i//2, j//2, j//2]
    # Shape (2N, 2N), exchange[i,j] = eri[i//2, j//2, j//2, i//2] if spin(i) == spin(j) else 0.0
    exchange = np.where(spin[:, None] == spin[None, :], eri[rows, cols, cols, rows], 0.0)
    # Shape (2N, 2N), G[i,j] = coulomb[i,j] - exchange[i,j] = eri[i//2, i//2, j//2, j//2] - eri[i//2, j//2, j//2, i//2] if spin(i) == spin(j) else eri[i//2, i//2, j//2, j//2]
    # Using spin-resolved ERIs in chemist order h_ijkl^στ: G[i,j] = h_iijj^στ - h_ijji^στ δ_μν
    G = coulomb - exchange
    # Shape (2N,), S[i] =  ∑_{j ∈ occ} G[i,j] = ∑_{j ∈ occ} ( eri[i//2, i//2, j//2, j//2] - eri[i//2, j//2, j//2, i//2] ) if spin(i) == spin(j) else ∑_{j ∈ occ} eri[i//2, i//2, j//2, j//2]
    S = G[:, occ_so_array].sum(axis=1)
    e_reference = float(h_diag[occ_so_array].sum() + 0.5 * S[occ_so_array].sum())

    predictions = []
    for excitation_indices in excitation_indices_list:
        sign_prefactor = 1.0
        if len(excitation_indices) == 2:
            # TCC order is (virt, occ), giving G = a^dag_j a_i directly, no swap needed.
            j_vir, i_occ = (_block_to_interleaved(idx, no, nv) for idx in excitation_indices)

            if i_occ not in occ_so_set:
                raise ValueError(f"Expected second index occupied in excitation {excitation_indices}")
            if j_vir in occ_so_set:
                raise ValueError(f"Expected first index virtual in excitation {excitation_indices}")
            if (j_vir % 2) != (i_occ % 2):
                raise ValueError(f"Expected a spin-conserving excitation but {excitation_indices} is not! (j, i) corresponds to a†_j a_i - a†_i a_j.")

            a_val, b_val = _compute_a_b_single(h1, eri, h_diag, G, S, occ_so_array, i_occ, j_vir, e_reference)
        else:
            k_vir, l_vir, i_occ, j_occ = (_block_to_interleaved(idx, no, nv) for idx in excitation_indices)
            # Check if exactly one of the following is true:
            #   1. the spin of i_occ is not the same as l_vir
            #   2. the spin of j_occ is not the same as k_vir
            # If yes, the excitation does not preserve spin which is not allowed
            if ((l_vir % 2) != (i_occ % 2)) != ((k_vir % 2) != (j_occ % 2)):
                raise ValueError(
                    f"Expected a spin-conserving excitation but {excitation_indices} is not! (k, l, i, j) corresponds to a†_k a†_l a_i a_j - a†_j a†_i a_l a_k."
                )
            # We expect for the following that the spins of i_occ and l_vir are the same and that the spins of j_occ and k_vir are the same
            #   If they are not the same we have to imagine that we swap the creation operator in the excitation.
            #   This swapping would introduce a minus sign which is handled by swapping indices and introducing a minus-sign via a prefactor.
            #   This then affects the optimal angle.
            if (l_vir % 2) != (i_occ % 2) or (k_vir % 2) != (j_occ % 2):
                sign_prefactor *= -1.0
                l_vir, k_vir = k_vir, l_vir

            if i_occ not in occ_so_set or j_occ not in occ_so_set:
                raise ValueError(f"Expected last two indices occupied: {excitation_indices}")
            if l_vir in occ_so_set or k_vir in occ_so_set:
                raise ValueError(f"Expected first two indices virtual: {excitation_indices}")

            a_val, b_val = _compute_a_b(eri, h_diag, G, S, i_occ, j_occ, k_vir, l_vir, e_reference)

        theta_opt = 0.5 * sign_prefactor * np.arctan2(-b_val, a_val)
        delta_E = -a_val + np.sqrt(a_val**2 + b_val**2)  # Energy impact, >= 0
        predictions.append((theta_opt, delta_E))

    return predictions


def optimal_theta(
    h1: np.ndarray,
    eri: np.ndarray,
    occ_spatial: Sequence[int],
    excitation_indices: Sequence[int],
    occ_spatial_beta: Sequence[int] | None = None,
) -> tuple[float, float]:
    r"""Optimal angle and energy impact for ONE single or double excitation.

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
        (k, l, i, j) corresponds to a double-excitation a^\dagger_k a^\dagger_l a_i a_j - a^\dagger_j a^\dagger_i a_l a_k
        and
        (j, i) corresponds to a single-excitation a^\dagger_j a_i - a^\dagger_i a_j
    occ_spatial_beta : Sequence[int] | None, optional
        Beta-spin spatial orbital indices occupied in the reference state. Defaults to
        ``occ_spatial`` (closed-shell reference); pass a different list for an ROHF open-shell
        reference, where PySCF's singly occupied orbitals are alpha-only.

    Returns
    -------
    theta_opt : float
        Exact optimal angle (in radians) minimising E(theta).
    delta_E : float
        Maximum energy impact -a + sqrt(a^2 + b^2) >= 0.
    """
    return optimal_thetas(h1, eri, occ_spatial, [excitation_indices], occ_spatial_beta)[0]


def optimal_thetas_pyscf(mf: Any, excitation_indices_list: Sequence[Sequence[int]]) -> list[tuple[float, float]]:
    r"""Batched :func:`optimal_theta_pyscf`: one AO->MO transform for the whole pool.

    Parameters
    ----------
    mf : pyscf.scf.hf.RHF or pyscf.scf.rohf.ROHF
        Converged PySCF mean-field object defining the molecular Hamiltonian.
    excitation_indices_list : Sequence[Sequence[int]]
        One entry per excitation:
        (k, l, i, j) corresponds to a^\dagger_k a^\dagger_l a_i a_j - a^\dagger_j a^\dagger_i a_l a_k
        and
        (j, i) corresponds to a^\dagger_j a_i - a^\dagger_i a_j
        Tuples, lists and 2-D arrays are all accepted.

    Returns
    -------
    list[tuple[float, float]]
        ``(theta, delta_E)`` per excitation, in the SAME ORDER as the input.
    """
    h1_mo, eri_mo, occ_spatial, occ_spatial_beta = _transform_integrals_to_mo(mf)
    return optimal_thetas(h1_mo, eri_mo, occ_spatial, excitation_indices_list, occ_spatial_beta)


def optimal_theta_pyscf(mf: Any, excitation_indices: Sequence[int]) -> tuple[float, float]:
    r"""Optimal angle and energy impact for ONE single or double excitation, from a PySCF object.

    Thin wrapper around :func:`optimal_thetas_pyscf`. Prefer the batched function when scoring
    more than one excitation: it performs the AO->MO transform once for the whole pool, so
    calling this in a loop repeats that transform per excitation.

    Parameters
    ----------
    mf : pyscf.scf.hf.RHF or pyscf.scf.rohf.ROHF
        Converged PySCF mean-field object defining the molecular Hamiltonian.
    excitation_indices : Sequence[int]
        (k, l, i, j) corresponds to a double-excitation a^\dagger_k a^\dagger_l a_i a_j - a^\dagger_j a^\dagger_i a_l a_k
        and
        (j, i) corresponds to a single-excitation a^\dagger_j a_i - a^\dagger_i a_j

    Returns
    -------
    theta_opt : float
        Exact optimal angle (in radians) minimising E(theta).
    delta_E : float
        Maximum energy impact -a + sqrt(a^2 + b^2) >= 0.
    """
    return optimal_thetas_pyscf(mf, [excitation_indices])[0]
