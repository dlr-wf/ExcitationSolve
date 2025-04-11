import logging
import numpy as np
from excitationsolve.trig_poly_utils import fourier_series_minimum


def excitation_solve_step(
    parameter_variations: list | np.ndarray,
    energy_samples: list | np.ndarray,
    return_coeffs=False,
    non_exact=False,
) -> tuple[float, float] | tuple[float, float, np.ndarray]:
    """
    Optimizes a single excitation parameter globally given the energies for five shifts of this parameters.
    Recommended to use in a loop for all excitation parameters and perform multiple sweeps through the parameters (potentially shuffled) until convergence.

    Args:
        parameter_variations: list or np.ndarray
            Five variations/shifts of a single excitation parameter.
        energy_samples: list or np.ndarray
            Five energy samples for a single excitation parameter varied.
    Returns:
        float, float
            The optimized excitation parameter and the corresponding energy.
    """

    parameter_variations = np.array(parameter_variations).flatten()
    energy_samples = np.array(energy_samples).flatten()
    assert (
        len(parameter_variations) >= 5
    ), f"The number of parameter variations must be at least 5. Got {len(parameter_variations)}."
    assert (
        len(energy_samples) >= 5
    ), f"The number of energy samples must be at least 5. Got {len(energy_samples)}."
    assert len(parameter_variations) == len(
        energy_samples
    ), f"The number of parameter variations and energy samples must match. Got {len(parameter_variations)} and {len(energy_samples)}."
    assert (
        parameter_variations.max() - parameter_variations.min() <= 4 * np.pi
    ), "The parameter variations must be within one period of the excitation operator (4*pi)."

    # ### Fit second-order Fourier series / trigonometric polynomial to the energy samples: ###
    # Determine linear equation system with N equations and 5 unknowns (N number of samples):
    c1 = np.cos(parameter_variations / 2)
    s1 = np.sin(parameter_variations / 2)
    c2 = np.cos(parameter_variations)
    s2 = np.sin(parameter_variations)
    A = np.array([np.ones(len(energy_samples)), c1, s1, c2, s2]).T
    b = np.array(energy_samples)
    logging.debug("Linear equation system: \nA: \n%s \n b: %s", np.around(A, 3), b)

    if len(parameter_variations) == 5:
        # ## Exact reconstruction: ##
        # solve the equation system:
        coeffs = np.linalg.solve(A, b)
    else:
        # ## Fitted reconstruction (least squares fit) for potentially noisy energy values: ##
        logging.debug(
            "More than 5 parameter variations or energy samples provided. A least squares fit will be performed instead of an exact reconstruction."
        )
        coeffs, residuals, rank_A, s = np.linalg.lstsq(A, b)
        logging.debug(
            "Least-squares fit info:\n\tResiduals: %s \n\tRank of A: %i \n\tSingular values of A: %s",
            residuals,
            rank_A,
            s,
        )

    logging.debug("Solved coefficients: %s", coeffs)

    if non_exact:
        # reconstructed energy function:
        def energy_function(xs):
            c1 = np.cos(xs / 2)
            s1 = np.sin(xs / 2)
            c2 = np.cos(xs)
            s2 = np.sin(xs)
            return (
                coeffs[0]
                + coeffs[1] * c1
                + coeffs[2] * s1
                + coeffs[3] * c2
                + coeffs[4] * s2
            )

        ### Brute-force optimization of reconstruction
        n_samples = int(1e3)
        x_min, x_max = -2 * np.pi, 2 * np.pi
        x = np.linspace(x_min, x_max, n_samples)
        y = energy_function(x)  # Note, vectorized evaluation must be supported!
        min_idx = np.argmin(y)
        return x[min_idx], y[min_idx]

    # ### Optimize parameter globally in reconstruction (companion matrix method)
    a = [coeffs[0], *coeffs[1::2]]  # cosine coefficients
    b = coeffs[2::2]  # sine coefficients
    assert (
        len(a) == len(b) + 1
    ), f"The number of cosine coefficients must be one more than the number of sine coefficients. Got {len(a)} and {len(b)}."
    min_x, min_y = fourier_series_minimum(a, b, return_y=True)
    # rescale to original parameter range since the optimization was performed on the interval [pi, pi] with doubled frequencies instead of [-2*pi, 2*pi]:
    min_x *= 2

    if return_coeffs:
        return min_x, min_y, coeffs
    return min_x, min_y


### Demo: ###
if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility and because 42 is the answer to everything
    logging.basicConfig(level=logging.INFO)  # Enable logging
    # Example data:
    parameter_variations_test = np.array([0, np.pi / 2, -np.pi / 2, np.pi, -np.pi])
    energy_samples_test = np.random.rand(5)  # random energies for demonstration
    # Optimize excitation parameter:
    optimized_parameter, optimized_energy = excitation_solve_step(
        parameter_variations_test, energy_samples_test
    )
    print(
        f"Optimized excitation parameter: {optimized_parameter} \nOptimized energy: {optimized_energy}"
    )
