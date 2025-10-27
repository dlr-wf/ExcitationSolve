# ExcitationSolve [![arXiv](https://img.shields.io/badge/arXiv-2409.05939-blue.svg?logo=arxiv&logoColor=white.svg)](https://arxiv.org/abs/2409.05939) [![DOI](https://zenodo.org/badge/958004975.svg)](https://doi.org/10.5281/zenodo.17457122)

An optimization algorithm for ansätze consisting of excitation operators in variational quantum eigensolvers (VQEs).

## Abstract
We introduce ExcitationSolve, a fast globally-informed gradient-free optimizer for physically-motivated ansätze constructed of excitation operators, a common choice in variational quantum eigensolvers. ExcitationSolve is to be classified as an extension of quantum-aware and hyperparameter-free optimizers such as Rotosolve, from parameterized unitaries with generators $G$ of the form $G^2=I$, e.g., rotations, to the more general class of $G^3=G$ exhibited by the physically-inspired excitation operators such as in the unitary coupled cluster approach. ExcitationSolve is capable of finding the global optimum along each variational parameter using the same quantum resources that gradient-based optimizers require for a single update step. We provide optimization strategies for both fixed- and adaptive variational ansätze, as well as a multi-parameter generalization for the simultaneous selection and optimization of multiple excitation operators. Finally, we demonstrate the utility of ExcitationSolve by conducting electronic ground state energy calculations of molecular systems and thereby outperforming state-of-the-art optimizers commonly employed in variational quantum algorithms. Across all tested molecules in their equilibrium geometry, ExcitationSolve remarkably reaches chemical accuracy in a single sweep over the parameters of a fixed ansatz. This sweep requires only the quantum circuit executions of one gradient descent step. In addition, ExcitationSolve achieves adaptive ansätze consisting of fewer operators than in the gradient-based adaptive approach, hence decreasing the circuit execution time.

## Examples
VQE convergence for different molecules and optimizers, including ExcitationSolve in red, using a UCCSD ansatz.
<h1>
<p align="center">
    <img src="assets/convergence_plots.png" alt="Convergence" width="500"/>
</p>
</h1>

## Installation
```
pip install git+https://github.com/dlr-wf/ExcitationSolve.git
```

## Usage
See the [examples folder](./examples/) for usage examples for qiskit and pennylane. The core (1D) algorithm is implemented in [excitation_solve.py](./excitationsolve/excitation_solve.py) and can be used independently of any quantum computing SDK like qiskit or pennylane. The 2D ExcitationSolve algorithm is implemented in [excitation_solve_2d.py](./excitationsolve/excitation_solve_2d.py). An implementation for using ExcitationSolve in combination with ADAPT-VQE in PennyLane can be found in [excitation_solve_adapt.py](./excitationsolve/excitation_solve_adapt.py).
### Qiskit 
```python
from excitationsolve.excitation_solve_qiskit import ExcitationSolveQiskit
```
Then use the `ExcitationSolveQiskit` optimizer instead of other qiskit optimizers, like COBYLA or L-BFGS, as demonstrated in the [qiskit example](./examples/main_qiskit.py).
#### Optimal energies and parameters
The qiskit VQE callback function can only track the energies and parameters of executed circuit. ExcitationSolve does not need to execute the circuit at optimal parameter configurations to perform the optimization. Therefore, the callback function never recieves optimal energies and parameters. Plotting these, would show false optimization progress, although the final optimized energy and paramters are saved in the `VQEResult` object, returned from the `VQE.compute_minimum_eigenvalue` function. To better track the optimization progress, the `ExcitationSolveQiskit` optimizer object internally saves all energies in its `energies` member variable, corresponding number of energy evaluations `nfevs` and optionally the parameters `params` (if `save_parameters` is set to `True`). These are saved after each optimization step, i.e. after optimizing each single parameter. For plotting the optimization progress use the data stored in `counts` and `values` from the following code snippet:
```python
optimizer = ExcitationSolveQiskit(maxiter=100, save_parameters=True)
# Perfrom optimization here...
counts = optimizer.nfevs
values = optimizer.energies
params = optimizer.params
```

### Pennylane
Use the `excitationsolve.excitation_solve_pennylane.excitationsolve_pennylane` function in a VQE loop as used in the [pennylane (dataset) example](./examples/main_pennylane.py) or the [pennylane (pyscf) example](./examples/main_pennylane_pyscf.py.py).
Alternatively, one can use the `excitationsolve.excitation_solve_step` function to optimize a single parameter. The `excitationsolve.excitation_solve_pennylane.excitationsolve_pennylane` optimizes all parameters, or a subset of them, in the quantum circuit using `excitationsolve.excitation_solve_step`.

### SciPy
Use the `excitationsolve.ExcitationSolveScipy` class in combination with the `scipy.optimize.minimize` function:
```python
excsolve_obj = ExcitationSolveScipy(maxiter=100, tol=1e-10, save_parameters=True)
optimizer = excsolve_obj.minimize
res = scipy.optimize.minimize(cost, params, method=optimizer)
energies = excsolve_obj.energies
counts = excsolve_obj.nfevs
```


## Authors
- Jonas Jäger
- Thierry N. Kaldenbach
- Max Haas
- Erik Schultheis

## Contact
Feel free to contact [David Melching](mailto:David.Melching@dlr.de) if you have any questions.

## Citation
If you use portions of this code please cite our [paper](https://arxiv.org/abs/2409.05939):
```bibtex
@misc{Jaeger2024Fast,
      title={Fast gradient-free optimization of excitations in variational quantum eigensolvers}, 
      author={Jonas Jäger and Thierry Nicolas Kaldenbach and Max Haas and Erik Schultheis},
      year={2024},
      eprint={2409.05939},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2409.05939}, 
      doi={10.48550/arXiv.2409.05939}
}
```

## Acknowledgment
This project was made possible by the DLR Quantum Computing Initiative and the Federal Ministry for Economic Affairs and Climate Action; https://qci.dlr.de/quanticom.
