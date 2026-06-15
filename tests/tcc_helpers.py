import numpy as np


# Adjusted from tencirchem.static.ucc.UCC.get_ex1_ops
def get_ex1_ops(norb: int, nelec: tuple[int, int]) -> list[tuple[int, int]]:
    assert nelec[0] == nelec[1], (
        f"Number of up- and down-spin electrons are different. "
        + "TenCirChem does not seem to support that since it only differentiates "
        + "between occupied and virtual orbital independent of the spin."
    )
    nelec = nelec[0]

    no = nelec  # Number of occupied orbitals
    nv = norb - nelec  # Number of virtual orbitals

    ex1_ops = []
    for i in range(no):
        for a in range(nv):
            # alpha to alpha
            ex_op_a = (2 * no + nv + a, no + nv + i)
            # beta to beta
            ex_op_b = (no + a, i)
            ex1_ops.extend([ex_op_a, ex_op_b])

    return ex1_ops


# Adjusted from tencirchem.static.ucc.UCC.get_ex2_ops
def get_ex2_ops(norb: int, nelec: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    assert nelec[0] == nelec[1], (
        f"Number of up- and down-spin electrons are different. "
        + "TenCirChem does not seem to support that since it only differentiates "
        + "between occupied and virtual orbital independent of the spin."
    )
    nelec = nelec[0]

    no = nelec  # Number of occupied orbitals
    nv = norb - nelec  # Number of virtual orbitals

    def alpha_o(_i):
        return no + nv + _i

    def alpha_v(_i):
        return 2 * no + nv + _i

    def beta_o(_i):
        return _i

    def beta_v(_i):
        return no + _i

    # double excitations
    ex_ops = []
    # 2 alphas or 2 betas
    for i in range(no):
        for j in range(i):
            for a in range(nv):
                for b in range(a):
                    # i correspond to a and j correspond to b, as in PySCF convention
                    # otherwise the t2 amplitude has incorrect phase
                    # 2 alphas
                    ex_op_aa = (alpha_v(b), alpha_v(a), alpha_o(i), alpha_o(j))
                    # 2 betas
                    ex_op_bb = (beta_v(b), beta_v(a), beta_o(i), beta_o(j))
                    ex_ops.extend([ex_op_aa, ex_op_bb])
    assert len(ex_ops) == 2 * (no * (no - 1) / 2) * (nv * (nv - 1) / 2)
    # 1 alpha + 1 beta
    for i in range(no):
        for j in range(i + 1):
            for a in range(nv):
                for b in range(a + 1):
                    # i correspond to a and j correspond to b, as in PySCF convention
                    # otherwise the t2 amplitude has incorrect phase
                    if i == j and a == b:
                        # paired
                        ex_op_ab = (beta_v(a), alpha_v(a), alpha_o(i), beta_o(i))
                        ex_ops.append(ex_op_ab)
                        continue
                    # simple reflection
                    ex_op_ab1 = (beta_v(b), alpha_v(a), alpha_o(i), beta_o(j))
                    ex_op_ab2 = (alpha_v(b), beta_v(a), beta_o(i), alpha_o(j))
                    ex_ops.extend([ex_op_ab1, ex_op_ab2])
                    if (i != j) and (a != b):
                        # exchange alpha and beta
                        ex_op_ab3 = (beta_v(a), alpha_v(b), alpha_o(i), beta_o(j))
                        ex_op_ab4 = (alpha_v(a), beta_v(b), beta_o(i), alpha_o(j))
                        ex_ops.extend([ex_op_ab3, ex_op_ab4])

    return ex_ops


if __name__ == "__main__":
    import tencirchem

    nelec_spin = 1
    norb = 3
    nelec = (nelec_spin, nelec_spin)
    n_electrons = np.sum(nelec)
    n_spin_orbitals = 2 * norb
    nocc = nelec_spin
    ###########################################

    occ_spin_lst = np.concatenate([np.ones(sum(nelec) // 2), np.zeros(norb - sum(nelec) // 2)]).tolist()
    occ_lst = occ_spin_lst + occ_spin_lst
    hf_qiskit_str = " ".join([str(int(x)) for x in occ_lst])

    hf_lst = np.concatenate([np.ones(sum(nelec)), np.zeros(n_spin_orbitals - sum(nelec))]).tolist()
    hf_str = " ".join([str(int(x)) for x in hf_lst])
    connection_str = " ".join(["|" for _ in range(len(hf_lst))])
    hf_indices = " ".join([str(int(x)) for x in range(len(hf_lst))])

    occ_spin_lst_tcc = np.concatenate([np.zeros(norb - sum(nelec) // 2), np.ones(sum(nelec) // 2)]).tolist()
    occ_lst_tcc = occ_spin_lst_tcc + occ_spin_lst_tcc
    hf_tcc_str = " ".join([str(int(x)) for x in occ_lst_tcc])
    hf_lst = np.concatenate([np.ones(sum(nelec)), np.zeros(n_spin_orbitals - sum(nelec))]).tolist()
    hf_indices_tcc = " ".join([str(int(x)) for x in reversed(range(len(hf_lst)))])

    print(f"HF state (TCC):         | {hf_tcc_str} >")
    print(f"indices (TCC):            {hf_indices_tcc}")
    print(50 * "-")
    print(f"indices (qiskit):         {hf_indices}")
    print(f"HF state (qiskit):      | {hf_qiskit_str} >")

    print("\nBuilding Hamiltonian...")

    # nocc = np.sum(h_ks.occupations)
    np.random.seed(42)
    tcc_uccsd = tencirchem.UCCSD.from_integral(np.random.randn(norb, norb), np.random.randn(norb, norb, norb, norb), nelec)
    # See https://tensorcircuit.github.io/TenCirChem-NG/faq.html#why-are-the-number-of-excitation-operators-and-the-number-of-parameters-different-what-is-param-ids
    tcc_uccsd.param_ids = None
    nparams = tcc_uccsd.n_params

    n_singles_tcc = len([x for x in tcc_uccsd.ex_ops if len(x) == 2])
    n_doubles_tcc = len([x for x in tcc_uccsd.ex_ops if len(x) == 4])
    print(f"TCC exc. ops. ({n_singles_tcc} singles, {n_doubles_tcc} doubles):\n{tcc_uccsd.ex_ops}")
    # print([f"{bin(i)[2:].zfill(norb*2)}" for i in tcc_uccsd.get_ci_strings()])
    print(f"TCC n_params:        {tcc_uccsd.n_params}")
    print(f"TCC param_ids:       {tcc_uccsd.param_ids}")
    print(f"TCC param_to_ex_ops: {dict(tcc_uccsd.param_to_ex_ops)}")

    singles_tcc = get_ex1_ops(norb, nelec)
    doubles_tcc = get_ex2_ops(norb, nelec)

    print()
    print(f"{len(singles_tcc)} singles TCC: {singles_tcc}")
    print(f"{len(doubles_tcc)} doubles TCC: {doubles_tcc}")

    exc_tcc = singles_tcc + doubles_tcc
    assert exc_tcc == tcc_uccsd.ex_ops
