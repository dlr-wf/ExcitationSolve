from __future__ import annotations
import logging
from collections.abc import Callable
import numpy as np
from scipy.optimize import OptimizeResult
from excitationsolve import excitation_solve_step


class ExcitationSolveScipy:
    def __init__(self, maxiter, tol=1e-12, num_samples=5, hf_energy=None, save_parameters=False, param_scaling=0.5):
        """The ExcitationSolve optimizer as a SciPy optimizer that can be given to the scipy.optimize.minimize function.

        Usage:
        ```python
            excsolve_obj = ExcitationSolveScipy(maxiter=100, tol=1e-10, save_parameters=True)
            optimizer = excsolve_obj.minimize
            res = scipy.optimize.minimize(cost, params, method=optimizer)
            energies = excsolve_obj.energies
            counts = excsolve_obj.nfevs
        ```

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
            save_parameters (bool, optional): If True, params member variable contains
                all optimal parameter values after each optimization step,
                i.e. after optimizing each single parameter. Defaults to False.
            param_scaling (float, optional): Factor used for rescaling the parameters. This ExcitationSolve optimizer
                                             expects the parameters to be 2\pi periodic. For example, in Qiskit
                                             the excitation parameters result in excitation operators being \pi periodic.
                                             Therefore, we use a factor of 0.5 for qiskit, resulting in a Period of 2\pi.
        """
        super().__init__()

        self.maxiter = int(maxiter)
        self.tol = tol
        assert num_samples >= 5, f"Number of samples needs to be greater or equal to 5 but is {num_samples}!"
        self.num_samples = num_samples
        self.hf_energy = hf_energy
        self.save_parameters = save_parameters
        self.param_scaling = param_scaling

        self.energies = []
        self.nfevs = []
        self.params = []

    def minimize(self, fun: Callable[[np.ndarray], float], x0: np.ndarray, args=(), **kwargs) -> OptimizeResult:
        param_dist = 4 * np.pi / self.num_samples
        shifts = (np.arange(self.num_samples) - self.num_samples // 2) * param_dist

        params_excsolve = np.array(x0.copy())
        num_params = len(params_excsolve)

        energy_at_zero = None
        if self.hf_energy is not None:
            energy_at_zero = self.hf_energy
        energies_once = [np.inf]

        nfev = 0
        for n_iter in range(self.maxiter):
            for param_to_vary in range(num_params):
                e_shifted = []
                for shift in shifts:
                    if shift == 0.0 and energy_at_zero is not None:
                        e_shifted.append(energy_at_zero)
                        continue
                    e_shifted.append(
                        fun(
                            np.array(
                                [
                                    ((shift * self.param_scaling + params_excsolve[i]) if i == param_to_vary else params_excsolve[i])
                                    for i in range(num_params)
                                ]
                            )
                        )
                    )
                    nfev += 1

                params_excsolve[param_to_vary], current_energy_excsolve = excitation_solve_step(
                    (shifts * self.param_scaling + params_excsolve[param_to_vary]) / self.param_scaling,
                    e_shifted,
                )
                params_excsolve[param_to_vary] *= self.param_scaling
                energy_at_zero = current_energy_excsolve

                # Energy at current optimal parameters (not necessary to evaluate,
                #   since we already know it from the excitation_solve_step function)
                # Nevertheless we can (unnecessarily) execute the circuit at the
                # current optimal parameters so that a VQE callback function gets information
                # about the optimization progress/convergence
                # fun(params_excsolve * param_scaling)

                self.energies.append(current_energy_excsolve)
                self.nfevs.append(nfev)
                if self.save_parameters:
                    self.params.append(params_excsolve)
                logging.debug("Current ExcitationSolve optimum energy: %s", current_energy_excsolve)

            energies_once.append(current_energy_excsolve)
            if n_iter > 0:
                msg = "Current ExcitationSolve optimum energy after %s iterations: %s | Diff. to prev.: %s" % (
                    n_iter + 1,
                    energies_once[-1],
                    np.abs(energies_once[-1] - energies_once[-2]),
                )
            else:
                msg = "Current ExcitationSolve optimum energy after %s iteration: %s" % (
                    n_iter + 1,
                    energies_once[-1],
                )
            logging.info(msg)
            if np.abs(energies_once[-1] - energies_once[-2]) <= self.tol:
                break

        result = OptimizeResult()
        result.x = params_excsolve
        result.fun = current_energy_excsolve
        result.nfev = nfev
        result.njev = None
        result.nit = n_iter + 1

        self.energies = np.array(self.energies)
        self.nfevs = np.array(self.nfevs)
        self.params = np.array(self.params)

        return result
