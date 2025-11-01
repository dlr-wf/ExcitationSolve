from excitationsolve import excitation_solve_step
import numpy as np

try:
    import pennylane as qml
except ImportError as e:
    raise ImportError(f"Install pennylane for using the pennylane implementation of ExcitationSolve. {e}")


def excitationsolve_pennylane(
    circuit: qml.QNode,
    params_excsolve: np.ndarray,
    params_to_vary: list | None = None,
    num_samples=5,
) -> tuple[np.ndarray, list]:
    """Optimizes all parameters once in the given pennylane quantum circuit.

    Args:
        circuit (QNode): Quantum circuit that takes the parameters params_excsolve
                         and returns a scalar value which should be optimized
        params_excsolve (pennylane.numpy.tensor): Tensor holding the parameters to be optimized.
        params_to_vary (list | None): List of indices or None. If list then only
                                      parameters with indices in that list are optimized.
                                      If None all parameters are optimized. Defaults to None.
        num_samples (int, optional): Number of different parameter shifts used to reconstruct
                                     the energy function. Has to be equal or greater than 5.
                                     If no noise is present values greater than five do not
                                     improve the optimization. Defaults to 5.

    Returns:
        tuple(pennylane.numpy.tensor, list): Optimized parameters and energies
                                             after optimizing each parameter
    """
    if params_to_vary is None:
        params_to_vary = range(len(params_excsolve))

    param_dist = 4 * np.pi / num_samples
    shifts = (np.arange(num_samples) - num_samples // 2) * param_dist

    e_excsolve_list = []

    for param_to_vary in params_to_vary:
        e_shifted = [
            circuit(np.array([(shift + param if i == param_to_vary else param) for i, param in enumerate(params_excsolve)])) for shift in shifts
        ]
        param_variations = shifts + params_excsolve[param_to_vary]
        params_excsolve[param_to_vary], e_excsolve = excitation_solve_step(param_variations, e_shifted)

        e_excsolve_list.append(e_excsolve)

    return params_excsolve, e_excsolve_list
