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

import numpy as np
try:
    import pyscf
    from pyscf import ao2mo
except ImportError:
    pass

def _transform_integrals_to_mo(mf):
    """
    Transform one- and two-electron integrals from AO to MO basis.

    Returns
    -------
    h1_mo : (nmo, nmo)
    eri_mo : (nmo, nmo, nmo, nmo)
    occ_indices : list[int]
    """
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


def optimal_theta_pyscf(mf: pyscf.scf.hf.RHF, excitation_indices: list[int]) -> tuple[float, float]:
    """
    High-level convenience function to compute the optimal VQE angle theta_opt
    for a given double excitation in an RHF reference.

    Accepts TCC-format excitation indices after Qiskit-style sorting:
        (*sorted(virts), *sorted(occs)) → indices sorted by value within each pair,
        occ/virt identity determined by checking against HF occupation.

    Parameters
    ----------
    mf : pyscf.scf.hf.RHF
        Converged PySCF mean-field object defining the molecular Hamiltonian.
    excitation_indices : list[int] or tuple[int, int, int, int]
        Four spin-orbital indices in TCC/Qiskit-sorted format.

    Returns
    -------
    theta_opt : float
        Exact optimal angle (in radians) minimising E(theta).
    delta_E : float
        Maximum energy impact a + sqrt(a^2 + b^2).
    """
    # Transform integrals to MO basis
    h1_mo, eri_mo, occ_spatial = _transform_integrals_to_mo(mf)

    return optimal_theta(h1_mo, eri_mo, occ_spatial, excitation_indices)


def optimal_theta(h1: np.ndarray, eri: np.ndarray, occ_spatial: list[int], excitation_indices: list[int]) -> tuple[float, float]:
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
