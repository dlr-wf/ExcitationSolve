from __future__ import annotations
from collections.abc import Callable
import numpy as np

try:
    from qiskit_algorithms.optimizers import (
        Optimizer,
        OptimizerSupportLevel,
        OptimizerResult,
    )
    from qiskit_algorithms.optimizers.optimizer import POINT
except ImportError as e:
    raise ImportError(f"Install qiskit for using the qiskit_algorithms implementation of ExcitationSolve. {e}")
from excitationsolve import excitation_solve_step


class ExcitationSolveQiskit(Optimizer):
    def __init__(self, maxiter, tol=1e-12, num_samples=5, hf_energy=None, save_parameters=False):
        """The ExcitationSolve optimizer.
        Note that this optimizer never needs to evaluate the ansatz circuit
        at the (current) optimal parameters, unless the optimal parameters fall onto
        the sample points used to reconstruct the energy function.
        Therefore, when used with a qiskit VQE object, the energies transmitted
        to a VQE callback function, do not seem to improve or converge. Nevertheless,
        the determined optimal energy and parameters are still returned.

        Args:
            maxiter (int): Maximum number of VQE iterations (maximum number of times to optimize all parameters)
            tol: Threshold of energy difference after subsequent VQE iterations defining convergence
            num_samples (int, optional): Number of different parameter values at which to sample
                the energy to reconstruct the energy function in one parameter.
                Must be greater or equal to 5. Defaults to 5.
            hf_energy (float | None, optional): The Hartree-Fock energy, i.e. the energy of the
                system where all parameters in the circuit are zero. If none, this will be
                calculated by evaluating the energy of the ansatz with all parameters set to zero.
                If this energy is known from a prior classical calculation, e.g. a Hartree-Fock
                calculation, one energy evaluation is saved. Defaults to None.
            save_parameters (bool, optional) If True, params member variable contains
                all optimal parameter values after each optimization step,
                i.e. after optimizing each single parameter. Defaults to False.
        """
        super().__init__()

        self.maxiter = maxiter
        self.tol = tol
        assert num_samples >= 5, f"Number of samples needs to be greater or equal to 5 but is {num_samples}!"
        self.num_samples = num_samples
        self.hf_energy = hf_energy
        self.save_parameters = save_parameters

        self.energies = []
        self.nfevs = []
        self.params = []

    def get_support_level(self):
        """Return support level dictionary"""
        return {
            "gradient": OptimizerSupportLevel.ignored,
            "bounds": OptimizerSupportLevel.ignored,
            "initial_point": OptimizerSupportLevel.required,
        }

    def minimize(
        self,
        fun: Callable[[POINT], float],
        x0: POINT,
        jac: Callable[[POINT], POINT] | None = None,
        bounds: list[tuple[float, float]] | None = None,
    ) -> OptimizerResult:
        self.energies = []
        self.nfevs = []
        self.params = []

        param_dist = 4 * np.pi / self.num_samples
        shifts = (np.arange(self.num_samples) - self.num_samples // 2) * param_dist

        params_excsolve = np.array(x0.copy())
        num_params = len(params_excsolve)

        # qiskit excitation parameters result in excitation operators being \pi periodic
        # ExcitationSolve expects them to be 2\pi periodic. Therefore, we rescale
        # the parameters
        param_scaling = 1 / 2

        energies_once = [np.inf]
        energy_at_zero = None
        if self.hf_energy is not None:
            energy_at_zero = self.hf_energy
            self.energies = [self.hf_energy]
            self.nfevs = [0]

        nfev = 0
        for n_iter in range(self.maxiter):
            for param_to_vary in range(num_params):
                e_shifted = []
                for shift in shifts:
                    if shift == 0.0 and energy_at_zero is not None:
                        e_shifted.append(energy_at_zero)
                        continue
                    energy_tmp = fun(
                        np.array(
                            [
                                ((shift + params_excsolve[i]) * param_scaling if i == param_to_vary else params_excsolve[i] * param_scaling)
                                for i in range(num_params)
                            ]
                        )
                    )
                    e_shifted.append(energy_tmp)
                    if shift == 0.0 and energy_at_zero is None:
                        # Save Hartree-Fock (HF) energy as first energy with 0 as corresponding number of circuit evaluations needed
                        # since the HF energy can be known before any circuit execution by performing a classical HF calculation.
                        self.energies = [energy_tmp]
                        self.nfevs = [0]
                    nfev += 1

                params_excsolve[param_to_vary], current_energy_excsolve = excitation_solve_step(
                    shifts + params_excsolve[param_to_vary],
                    e_shifted,
                )
                energy_at_zero = current_energy_excsolve

                # Energy at current optimal parameters (not necessary to evaluate,
                #   since we already know it from the excitation_solve_step function)
                # Nevertheless we could (unnecessarily) execute the circuit at the
                # current optimal parameters so that a VQE callback function gets information
                # about the optimization progress/convergence
                # fun(params_excsolve * param_scaling)

                # The optimal energies at each step are saved in the self.energies list
                self.energies.append(current_energy_excsolve)
                self.nfevs.append(nfev)
                if self.save_parameters:
                    self.params.append(params_excsolve * param_scaling)

            energies_once.append(current_energy_excsolve)
            if np.abs(energies_once[-1] - energies_once[-2]) <= self.tol:
                break

        result = OptimizerResult()
        result.x = params_excsolve * param_scaling
        result.fun = current_energy_excsolve
        result.nfev = nfev
        result.njev = None
        result.nit = n_iter + 1

        self.energies = np.array(self.energies)
        self.nfevs = np.array(self.nfevs)
        self.params = np.array(self.params)

        return result
