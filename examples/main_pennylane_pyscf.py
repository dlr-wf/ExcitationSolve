import time
from pyscf import gto, ao2mo, scf, fci
import pennylane as qml
from pennylane import numpy as np
import scipy
import matplotlib.pyplot as plt
from excitationsolve.excitation_solve_pennylane import excitationsolve_pennylane

if __name__ == "__main__":
    symbols = ["H", "H"]
    # from mol.coordinates
    bondlength = 0.742
    basis = "STO-3G"
    geometry = np.array(
        [
            [-bondlength / 2, 0.0, 0.0],
            [+bondlength / 2, 0.0, 0.0],
        ]
    )
    charge = 0

    atom = "; ".join([f"{a} {', '.join([str(x) for x in p.tolist()])}" for a, p in zip(symbols, geometry)])
    symbols_unique, unique_counts = np.unique(symbols, return_counts=True)
    molname = "".join([f"{a}{n}" for a, n in zip(symbols_unique, unique_counts)])
    if charge > 0:
        molname += "+" * np.abs(charge)
    elif charge < 0:
        molname += "-" * np.abs(charge)

    unit = "Angstrom"  # Angstrom or Bohr
    mol_pyscf = gto.M(atom=atom, basis=basis, charge=charge, unit=unit)
    electrons = mol_pyscf.nelectron
    rhf = scf.RHF(mol_pyscf)
    energy = rhf.kernel()

    fci_calc = fci.FCI(mol_pyscf, rhf.mo_coeff)
    fci_energy, ci_vector = fci_calc.kernel()
    print(f"FCI Energy: {fci_energy} Ha")

    one_ao = mol_pyscf.intor_symmetric("int1e_kin") + mol_pyscf.intor_symmetric("int1e_nuc")
    two_ao = mol_pyscf.intor("int2e_sph")
    one_mo = np.einsum("pi,pq,qj->ij", rhf.mo_coeff, one_ao, rhf.mo_coeff)
    two_mo = ao2mo.incore.full(two_ao, rhf.mo_coeff)
    two_mo = np.swapaxes(two_mo, 1, 3)  # to physicist order
    core_constant = np.array([rhf.energy_nuc()])
    H_fermionic = qml.qchem.fermionic_observable(core_constant, one_mo, two_mo)

    H = qml.jordan_wigner(H_fermionic)
    qubits = H.num_wires
    wires = range(qubits)

    # Define the HF state
    hf_state = qml.qchem.hf_state(electrons, qubits)

    # Generate single and double excitations
    singles, doubles = qml.qchem.excitations(electrons, qubits)

    # Map excitations to the wires the UCCSD circuit will act on
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

    # Define the initial values of the circuit parameters
    num_params = len(singles) + len(doubles)
    # Define the device
    dev_hf = qml.device("lightning.qubit", wires=qubits)
    dev_excsolve = qml.device("lightning.qubit", wires=qubits)

    @qml.qnode(dev_hf)
    def circuit_hf():
        qml.BasisState(hf_state, wires=range(qubits))
        return qml.expval(H)

    hf_energy = circuit_hf()

    # Define the qnode
    @qml.qnode(dev_excsolve, interface=None)
    def circuit_excsolve(params):
        global singles
        # Fix UCCSD parameter order: first doubles, then singles
        params = np.concatenate([params[len(singles) :], params[: len(singles)]])
        qml.UCCSD(params, wires, s_wires, d_wires, hf_state)
        return qml.expval(H)

    # Define the initial values of the circuit parameters
    params_excsolve = np.zeros(num_params)

    num_samples = 5
    num_vqe_steps = 10

    energies_excsolve = []
    energy_evaluations = (np.arange(num_vqe_steps * num_params) + 1) * 4 - 4

    # Optimize the circuit parameters and compute the energy
    print("Starting ExcitationSolve VQE optimization (starting timer)")
    start_time = time.time()

    current_energy_excsolve = hf_energy
    for n in range(num_vqe_steps):
        params_excsolve, current_energy_excsolve = excitationsolve_pennylane(circuit_excsolve, params_excsolve, num_samples=num_samples)
        energies_excsolve.extend(current_energy_excsolve)

        print(
            f"step = {n}, E_excsolve = {energies_excsolve[-1]:.8f} Ha, Diff. to FCI: {np.abs(energies_excsolve[-1]-fci_energy)}",
            end="\r",
        )

    energies_excsolve = np.array(energies_excsolve)

    end_time = time.time()
    time_s = end_time - start_time
    time_min = time_s / 60
    time_h = time_min / 60
    print(f"Took {time_s:.2f} s/{time_min:.2f} min/{time_h:.2f} h!")
    print(f"E_excsolve = {energies_excsolve[-1]:.8f} Ha, Diff. to FCI: {np.abs(energies_excsolve[-1]-fci_energy)}")

    dev_cobyla = qml.device("lightning.qubit", wires=qubits)

    @qml.qnode(dev_cobyla, interface=None)
    def circuit_cobyla(params):
        qml.UCCSD(params, wires, s_wires, d_wires, hf_state)
        return qml.expval(H)

    params_cobyla = np.zeros(num_params)
    energies_cobyla = []
    count_cobyla = 0

    def cost(weights):
        global count_cobyla
        count_cobyla = count_cobyla + 1
        energy_cobyla = circuit_cobyla(weights)
        energies_cobyla.append(energy_cobyla)
        print(
            f"Iteration {len(energies_cobyla)}, Diff. to FCI: {np.abs(energy_cobyla-fci_energy)}",
            end="\r",
        )
        return energy_cobyla

    print("Starting COBYLA optimization")
    maxiter_cobyla = 1_000_000
    cobyla_res = scipy.optimize.minimize(
        cost,
        x0=params_cobyla,
        method="COBYLA",
        options={"maxiter": maxiter_cobyla},
        tol=1e-13,
    )

    params_cobyla = cobyla_res.x
    cobyla_evaluations = np.arange(len(energies_cobyla))
    print(cobyla_res)

    # Options
    params = {
        "text.usetex": True,
        "font.size": 12,
        "font.family": "lmodern",
        "text.latex.preamble": r"\usepackage{amsmath}",
    }

    plt.rcParams.update(params)

    colors = plt.cm.tab20.colors

    fig, ax = plt.subplots(figsize=(6.7 / 2, 4))
    ax.set_title(f"{molname} $\\vert$ {bondlength} $\\vert$ {basis}")
    ax.set_xlabel(r"\#Energy evaluations [-]")
    ax.set_ylabel("Energy difference to FCI [Ha]")

    label = r"$E_\mathrm{Exc.Solve}$"
    ax.plot(
        energy_evaluations,
        np.abs(fci_energy - energies_excsolve),
        linestyle="-",
        marker=".",
        color=colors[6],
        label=label,
    )

    ax.plot(
        cobyla_evaluations,
        np.abs(fci_energy - energies_cobyla),
        linestyle="-",
        marker=".",
        markersize=4,
        color=colors[8],
        label=r"$E_\mathrm{COBYLA}$",
    )

    ax.set_yscale("log")
    ax.set_yscale("log")

    start_x, end_x = ax.get_xlim()
    start_y, end_y = ax.get_ylim()

    chem_acc = 1e-3  # "chemical accuracy"
    chem_acc_color = plt.cm.tab10.colors[-1]
    chem_acc_alpha = 0.3
    ax.axhspan(
        ymin=0.0,
        ymax=chem_acc,
        color=chem_acc_color,
        alpha=chem_acc_alpha,
        label=r"$\leq$~chem.~acc.",
    )

    ax.set_yscale("log")
    ax.legend(loc="upper right")
    ax.grid(axis="y")

    ax.yaxis.set_minor_locator(plt.NullLocator())

    num_vqe_steps = 100
    lines_1d = np.arange(0, end_x, (num_samples - 1) * num_params)[1 : num_vqe_steps + 1]

    # shaded area in the background every second vqe_step
    shaded_color = plt.cm.tab10.colors[-3]
    shaded_color = "black"
    for x_pos in np.arange(0, end_x, (num_samples - 1) * num_params)[: num_vqe_steps + 1 : 2]:
        plt.axvspan(x_pos, x_pos + (num_samples - 1) * num_params, color=shaded_color, alpha=0.1)
