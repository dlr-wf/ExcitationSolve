import numpy as np
import scipy
import logging


def excitation_solve_2d_step(
    parameter_variations: list | np.ndarray,
    energy_samples: list | np.ndarray,
    return_coeffs=False,
    non_exact=False,
) -> tuple[float, float] | tuple[float, float, np.ndarray]:
    """
    Optimizes two excitation parameters globally given the energies for 25 shifts of this parameters.
    Recommended to use in a loop for all excitation parameters and perform multiple sweeps through the parameters (potentially shuffled) until convergence.

    Args:
        parameter_variations: list or np.ndarray
            25 variations/shifts of two excitation parameters. Shape: (25, 2)
        energy_samples: list or np.ndarray
            25 energy samples for two excitation parameters varied.
    Returns:
        float, float
            The optimized excitation parameter and the corresponding energy.
    """

    parameter_variations = np.array(parameter_variations)
    energy_samples = np.array(energy_samples).flatten()
    assert parameter_variations.ndim == 2, f"The parameter variations must be a 2D array. Got {parameter_variations.ndim} dimensions."
    assert len(parameter_variations) >= 5**2, f"The number of parameter variations must be at least 5. Got {len(parameter_variations)}."
    assert parameter_variations.shape[1] == 2, f"The parameter variations must have two columns. Got {parameter_variations.shape[1]}."
    assert len(energy_samples) >= 5**2, f"The number of energy samples must be at least 5. Got {len(energy_samples)}."
    assert len(parameter_variations) == len(energy_samples), (
        f"The number of parameter variations and energy samples must match. Got {len(parameter_variations)} and {len(energy_samples)}."
    )
    assert parameter_variations[:, 0].max() - parameter_variations[:, 0].min() <= 4 * np.pi, (
        "The variations of the first parameter must be within one period of the excitation operator (4*pi)."
    )
    assert parameter_variations[:, 1].max() - parameter_variations[:, 1].min() <= 4 * np.pi, (
        "The variations of the second parameter must be within one period of the excitation operator (4*pi)."
    )

    # ### Fit second-order Fourier series / trigonometric polynomial to the energy samples: ###
    # Determine linear equation system with N equations and 5**2 unknowns (N number of samples):
    param_1_terms = [
        np.ones(len(parameter_variations)),
        np.cos(parameter_variations[:, 0] / 2),
        np.sin(parameter_variations[:, 0] / 2),
        np.cos(parameter_variations[:, 0]),
        np.sin(parameter_variations[:, 0]),
    ]
    param_2_terms = [
        np.ones(len(parameter_variations)),
        np.cos(parameter_variations[:, 1] / 2),
        np.sin(parameter_variations[:, 1] / 2),
        np.cos(parameter_variations[:, 1]),
        np.sin(parameter_variations[:, 1]),
    ]
    A = np.array([p1_term * p2_term for p1_term in param_1_terms for p2_term in param_2_terms]).T
    b = np.array(energy_samples)
    logging.debug("Linear equation system: \nA: \n%s \n b: %s", np.around(A, 3), b)

    if len(parameter_variations) == 5**2:
        # ## Exact reconstruction: ##
        # solve the equation system:
        coeffs = np.linalg.solve(A, b)
    else:
        raise NotImplementedError("Least squares fit not implemented for 2D parameter variations yet.")

    logging.debug("Solved coefficients: %s", coeffs)

    def energy_function(xs):
        if xs.ndim == 1:
            xs = np.array([xs])
        xs_1_terms = [
            np.ones(len(xs)),
            np.cos(xs[:, 0] / 2),
            np.sin(xs[:, 0] / 2),
            np.cos(xs[:, 0]),
            np.sin(xs[:, 0]),
        ]
        xs_2_terms = [
            np.ones(len(xs)),
            np.cos(xs[:, 1] / 2),
            np.sin(xs[:, 1] / 2),
            np.cos(xs[:, 1]),
            np.sin(xs[:, 1]),
        ]
        A = np.array([xs1_term * xs2_term for xs1_term in xs_1_terms for xs2_term in xs_2_terms]).T
        ys = A @ coeffs
        return ys

    if non_exact:
        # ### Optimize parameter globally in reconstruction (brute-force sampling method) ###
        # sample in both directions of the parameter space:
        n_samples_per_dim = int(1e3)
        samples_one_dim = np.linspace(-2 * np.pi, 2 * np.pi, n_samples_per_dim)
        samples = np.array([[x, y] for x in samples_one_dim for y in samples_one_dim])
        sampled_energies = energy_function(samples)
        min_idx = np.argmin(sampled_energies)
        min_x, min_y = samples[min_idx], sampled_energies[min_idx]
    else:
        # ### Optimize parameter globally in reconstruction (Nyquist + gradient descent)
        n_nyquist_samples_per_dim = 5
        init_points = np.linspace(-2 * np.pi, 2 * np.pi, n_nyquist_samples_per_dim + 1)[:-1]

        min_x = None
        min_y = np.infty
        for x1 in init_points:
            for x2 in init_points:
                res = scipy.optimize.minimize(energy_function, x0=np.array([x1, x2]), method="BFGS", options={"maxiter": 1_000_000})
                if res.success:
                    if res.fun < min_y:
                        min_x = res.x
                        min_x = (min_x + 2 * np.pi) % (4 * np.pi) - 2 * np.pi
                        min_y = res.fun

                    logging.debug(
                        "Gradient descent optimization converged after %i iterations to point %s.",
                        res.nit,
                        res.x,
                    )
                else:
                    logging.warning(
                        "Gradient descent optimization did not converge for initial point %s, %s.",
                        x1,
                        x2,
                    )

    logging.debug("Returning point x=%s, y=%s.", min_x, min_y)

    if return_coeffs:
        return min_x, min_y, coeffs
    return min_x, min_y


### Demo: ###
if __name__ == "__main__":
    # np.random.seed(42)  # For reproducibility and because 42 is the answer to everything
    logging.basicConfig(level=logging.INFO)  # Enable logging
    # Example data:
    parameter_variations_test = np.linspace(-2 * np.pi, 2 * np.pi, 6)[:-1]  # 5 variations of the excitation parameter (for demonstration)
    parameter_2d_variations_test = np.array([[x, y] for x in parameter_variations_test for y in parameter_variations_test])
    energy_samples_test = np.random.rand(len(parameter_2d_variations_test))  # random energies for demonstration
    # print(parameter_2d_variations_test)
    # print(energy_samples_test)

    # Optimize excitation parameter:
    print("Performing optimization via scipy optimization...")
    optimized_parameter, optimized_energy = excitation_solve_2d_step(parameter_2d_variations_test, energy_samples_test)
    # Double-check with brute-force optimization:
    print("Performing optimization via brute-force optimization...")
    optimized_parameter_check, optimized_energy_check = excitation_solve_2d_step(parameter_2d_variations_test, energy_samples_test, non_exact=True)
    print(
        f"Optimized excitation parameter: {optimized_parameter} \nOptimized energy: {optimized_energy}\n\n"
        f"Optimized excitation parameter (brute-force): {optimized_parameter_check} \n"
        f"Optimized energy (brute-force): {optimized_energy_check}\n"
        f"Optimized energy difference: {np.abs(optimized_energy_check - optimized_energy)}"
    )

    assert optimized_energy <= optimized_energy_check, "Optimized energy is not minimal."
