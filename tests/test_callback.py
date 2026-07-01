import time
import numpy as np
import scipy
import matplotlib.pyplot as plt
from pyscf import scf, gto, fci, ao2mo
from tencirchem import UCCSD
import tcc_helpers
import pyscf_molecules
from excitationsolve import ExcitationSolveScipy
import logging


def test_tcc_h3plus_callback_xk():
    logging.basicConfig(level=logging.INFO)

    # molecule = pyscf_molecules.H_chain.build_hydrogen_chain(2)
    molecule = pyscf_molecules.H3plus
    symbols = molecule.symbols
    basis = molecule.basis
    geometry = molecule.geometry
    charge = molecule.charge

    atom = "; ".join([f"{a} {', '.join([str(x) for x in p.tolist()])}" for a, p in zip(symbols, geometry)])
    symbols_unique, unique_counts = np.unique(symbols, return_counts=True)
    molname = "".join([f"{a}{n}" for a, n in zip(symbols_unique, unique_counts)])
    if charge > 0:
        molname += "+" * np.abs(charge)
    elif charge < 0:
        molname += "-" * np.abs(charge)

    unit = "Angstrom"  # Angstrom or Bohr
    mol_pyscf = gto.M(atom=atom, basis=basis, charge=charge, unit=unit)
    _electrons = mol_pyscf.nelectron
    rhf = scf.RHF(mol_pyscf)
    _hf_energy = rhf.kernel()

    _mo_occ = rhf.mo_occ

    print(f"Building TCC Hamiltonian for {molname} (early build to use TCC canonicalized MO-coefficients) ...")
    tcc_uccsd = UCCSD(rhf, init_method="zeros", run_hf=False, run_mp2=False, run_ccsd=False, run_fci=True)  # TCC params = -params
    print(f"{tcc_uccsd.engine=}")
    rhf.mo_coeff = tcc_uccsd.hf.mo_coeff

    fci_calc = fci.FCI(mol_pyscf, rhf.mo_coeff)
    fci_energy, ci_vector = fci_calc.kernel()
    print(f"FCI Energy: {fci_energy} Ha")

    reference_energy = fci_energy

    # one_ao = mol_pyscf.intor_symmetric("int1e_kin") + mol_pyscf.intor_symmetric("int1e_nuc")
    # two_ao = mol_pyscf.intor("int2e_sph")
    # one_mo = np.einsum("pi,pq,qj->ij", rhf.mo_coeff, one_ao, rhf.mo_coeff) * 1.0
    # two_mo = ao2mo.incore.full(two_ao, rhf.mo_coeff) * 1.0
    # two_mo = two_mo.swapaxes(1, 2).swapaxes(2, 3)  # to physicist order
    # core_constant = rhf.energy_nuc()

    nelec = mol_pyscf.nelec
    norb = mol_pyscf.nao

    # energy_offset = core_constant

    #####################################################################################
    ##                                 Define Ansatz                                   ##
    #####################################################################################
    singles_pool = tcc_helpers.get_ex1_ops(norb, nelec)
    doubles_pool = tcc_helpers.get_ex2_ops(norb, nelec)
    singles_pool_sorted = sorted(singles_pool)
    doubles_pool_sorted = sorted([(*sorted(x[:2]), *sorted(x[2:])) for x in doubles_pool])
    # complete_pool = doubles_pool_sorted + singles_pool_sorted
    complete_pool = singles_pool_sorted + doubles_pool_sorted

    tcc_uccsd.ex_ops = complete_pool
    # tcc_uccsd.param_ids = None
    # tcc_uccsd.param_ids = [0, 0, 1]
    print(f"{tcc_uccsd.param_ids=}")

    n_params = len(complete_pool)
    _params = np.zeros(n_params)
    _ex_ops = [(x[: len(x) // 2], x[len(x) // 2 :]) for x in complete_pool]

    #######################
    # Manual optimization
    times_tcc = []
    eval_count_tcc = 0
    counts_tcc = []
    values_tcc = []
    tcc_vqe_params_lst = []

    def cost(x):
        nonlocal times_tcc
        nonlocal eval_count_tcc
        nonlocal counts_tcc
        nonlocal values_tcc

        tcc_energy = tcc_uccsd.energy(x)
        times_tcc.append(time.perf_counter())

        eval_count_tcc += 1
        print(
            f"Optimizer evaluation #{eval_count_tcc}, Diff. to ref.: {np.abs(tcc_energy - reference_energy)}",
            end="\r",
            flush=True,
        )
        counts_tcc.append(eval_count_tcc)
        values_tcc.append(tcc_energy)
        tcc_vqe_params_lst.append(x)

        return tcc_energy

    callback_works = False

    def callback(xk):
        nonlocal callback_works
        callback_works = True
        print(f" ========= From callback: {xk=} =========")

    maxiter = 100
    excsolve_obj = ExcitationSolveScipy(maxiter=maxiter, tol=1e-10, save_parameters=True)
    optimizer_func = excsolve_obj.minimize
    _, parameter_occ = np.unique(tcc_uccsd.param_ids, return_counts=True)
    options = dict(reference_energy=reference_energy, parameter_occ=parameter_occ)

    n_params_tmp = tcc_uccsd.n_params
    initial_params = np.zeros(n_params_tmp)
    res_tcc = scipy.optimize.minimize(cost, initial_params, method=optimizer_func, callback=callback, options=options)
    _params_tcc = res_tcc.x

    print(f"\nFinal energy difference: {(res_tcc.fun - reference_energy):.2e}")

    # plt.plot(excsolve_obj.nfevs, np.abs(excsolve_obj.energies - reference_energy))
    # plt.plot(excsolve_obj.nfevs_after_it, np.abs(excsolve_obj.energies_after_it - reference_energy), linestyle="", marker="x")
    # for x in excsolve_obj.nfevs_after_it:
    #     plt.axvline(x, linestyle="--", color="black", alpha=0.5)
    # plt.yscale("log")
    # plt.grid()
    # plt.show()

    assert callback_works, "Callback does not work!"


def test_tcc_h3plus_callback_intermediateresult():
    logging.basicConfig(level=logging.INFO)

    # molecule = pyscf_molecules.H_chain.build_hydrogen_chain(2)
    molecule = pyscf_molecules.H3plus
    symbols = molecule.symbols
    basis = molecule.basis
    geometry = molecule.geometry
    charge = molecule.charge

    atom = "; ".join([f"{a} {', '.join([str(x) for x in p.tolist()])}" for a, p in zip(symbols, geometry)])
    symbols_unique, unique_counts = np.unique(symbols, return_counts=True)
    molname = "".join([f"{a}{n}" for a, n in zip(symbols_unique, unique_counts)])
    if charge > 0:
        molname += "+" * np.abs(charge)
    elif charge < 0:
        molname += "-" * np.abs(charge)

    unit = "Angstrom"  # Angstrom or Bohr
    mol_pyscf = gto.M(atom=atom, basis=basis, charge=charge, unit=unit)
    _electrons = mol_pyscf.nelectron
    rhf = scf.RHF(mol_pyscf)
    _hf_energy = rhf.kernel()

    _mo_occ = rhf.mo_occ

    print(f"Building TCC Hamiltonian for {molname} (early build to use TCC canonicalized MO-coefficients) ...")
    tcc_uccsd = UCCSD(rhf, init_method="zeros", run_hf=False, run_mp2=False, run_ccsd=False, run_fci=True)  # TCC params = -params
    print(f"{tcc_uccsd.engine=}")
    rhf.mo_coeff = tcc_uccsd.hf.mo_coeff

    fci_calc = fci.FCI(mol_pyscf, rhf.mo_coeff)
    fci_energy, ci_vector = fci_calc.kernel()
    print(f"FCI Energy: {fci_energy} Ha")

    reference_energy = fci_energy

    # one_ao = mol_pyscf.intor_symmetric("int1e_kin") + mol_pyscf.intor_symmetric("int1e_nuc")
    # two_ao = mol_pyscf.intor("int2e_sph")
    # one_mo = np.einsum("pi,pq,qj->ij", rhf.mo_coeff, one_ao, rhf.mo_coeff) * 1.0
    # two_mo = ao2mo.incore.full(two_ao, rhf.mo_coeff) * 1.0
    # two_mo = two_mo.swapaxes(1, 2).swapaxes(2, 3)  # to physicist order
    # core_constant = rhf.energy_nuc()

    nelec = mol_pyscf.nelec
    norb = mol_pyscf.nao

    # energy_offset = core_constant

    #####################################################################################
    ##                                 Define Ansatz                                   ##
    #####################################################################################
    singles_pool = tcc_helpers.get_ex1_ops(norb, nelec)
    doubles_pool = tcc_helpers.get_ex2_ops(norb, nelec)
    singles_pool_sorted = sorted(singles_pool)
    doubles_pool_sorted = sorted([(*sorted(x[:2]), *sorted(x[2:])) for x in doubles_pool])
    # complete_pool = doubles_pool_sorted + singles_pool_sorted
    complete_pool = singles_pool_sorted + doubles_pool_sorted

    tcc_uccsd.ex_ops = complete_pool
    # tcc_uccsd.param_ids = None
    # tcc_uccsd.param_ids = [0, 0, 1]
    print(f"{tcc_uccsd.param_ids=}")

    n_params = len(complete_pool)
    _params = np.zeros(n_params)
    _ex_ops = [(x[: len(x) // 2], x[len(x) // 2 :]) for x in complete_pool]

    #######################
    # Manual optimization
    times_tcc = []
    eval_count_tcc = 0
    counts_tcc = []
    values_tcc = []
    tcc_vqe_params_lst = []

    def cost(x):
        nonlocal times_tcc
        nonlocal eval_count_tcc
        nonlocal counts_tcc
        nonlocal values_tcc

        tcc_energy = tcc_uccsd.energy(x)
        times_tcc.append(time.perf_counter())

        eval_count_tcc += 1
        print(
            f"Optimizer evaluation #{eval_count_tcc}, Diff. to ref.: {np.abs(tcc_energy - reference_energy)}",
            end="\r",
            flush=True,
        )
        counts_tcc.append(eval_count_tcc)
        values_tcc.append(tcc_energy)
        tcc_vqe_params_lst.append(x)

        return tcc_energy

    callback_works = False

    def callback(intermediate_result):
        nonlocal callback_works
        callback_works = True
        print(f" ========= From callback: {intermediate_result.x=} | {intermediate_result.fun=} =========")

    maxiter = 100
    excsolve_obj = ExcitationSolveScipy(maxiter=maxiter, tol=1e-10, save_parameters=True)
    optimizer_func = excsolve_obj.minimize
    _, parameter_occ = np.unique(tcc_uccsd.param_ids, return_counts=True)
    options = dict(reference_energy=reference_energy, parameter_occ=parameter_occ)

    n_params_tmp = tcc_uccsd.n_params
    initial_params = np.zeros(n_params_tmp)
    res_tcc = scipy.optimize.minimize(cost, initial_params, method=optimizer_func, callback=callback, options=options)
    _params_tcc = res_tcc.x

    print(f"\nFinal energy difference: {(res_tcc.fun - reference_energy):.2e}")

    # plt.plot(excsolve_obj.nfevs, np.abs(excsolve_obj.energies - reference_energy))
    # plt.plot(excsolve_obj.nfevs_after_it, np.abs(excsolve_obj.energies_after_it - reference_energy), linestyle="", marker="x")
    # for x in excsolve_obj.nfevs_after_it:
    #     plt.axvline(x, linestyle="--", color="black", alpha=0.5)
    # plt.yscale("log")
    # plt.grid()
    # plt.show()

    assert callback_works, "Callback does not work!"
