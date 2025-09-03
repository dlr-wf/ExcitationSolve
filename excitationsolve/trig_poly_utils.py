"""
This utils script implements the direct optimization of a trigonometric polynomial as described in

    Boyd, John. “Computing the Zeros, Maxima and Inflection Points of Chebyshev, Legendre and Fourier Series: 
    Solving Transcendental Equations by Spectral Interpolation and Polynomial Rootfinding.” 
    Journal of Engineering Mathematics 56 (November 2006): 203–19. https://doi.org/10.1007/s10665-006-9087-5.

This is realized via a conversion
- derive the trigonometric polynomial (aka finite Fourier series), conserves order N
- into a complex polynomial of order 2N
- root finding of this polynomial via matrix companion method (i.e., an eigenvalue problem) implemented in numpy

:Authors:
    Jonas Jaeger <jojaeger@cs.ubc.ca>
:Date: 
    October 2023
"""

import numpy as np
import matplotlib.pyplot as plt


def fourier_series_from_coeffs(a, b):
    """Returns the Fourier series corresponding to the given coefficients."""
    degree = len(b)

    def f(x):
        return a[0] + sum(
            [
                a[i] * np.cos(i * x) + b[i - 1] * np.sin(i * x)
                for i in range(1, degree + 1)
            ]
        )

    return f


def fourier_series_derivative(a, b):
    """Returns the derivative of the given Fourier series."""
    degree = len(b)
    a_prime = np.empty(degree + 1)
    b_prime = np.empty(degree)
    for i in range(degree + 1):
        a_prime[i] = i * b[i - 1] if i > 0 else 0
    for i in range(degree):
        b_prime[i] = -(i + 1) * a[i + 1]
    return a_prime, b_prime


def fourier_series_to_polynomial(a, b):
    """Returns the polynomial associated with the given Fourier series."""
    degree_trig = len(b)
    degree_poly = 2 * degree_trig
    hs = np.empty(degree_poly + 1, dtype=np.complex128)
    for j in range(len(hs)):
        if j == degree_trig:
            hs[j] = 2 * a[0]
        elif j < degree_trig:
            hs[j] = a[degree_trig - j] + 1j * b[degree_trig - j - 1]
        else:
            hs[j] = a[j - degree_trig] - 1j * b[j - degree_trig - 1]
    hs /= 2
    return np.poly1d(list(reversed(hs)))


def fourier_series_zeros(a, b, only_real=True):
    """Returns the zeros of the polynomial associated with the given Fourier series."""
    polynomial_zeros = np.roots(fourier_series_to_polynomial(a, b))
    if only_real:
        zeros = [
            np.angle(p_zero)
            for p_zero in polynomial_zeros
            if np.isclose(np.abs(p_zero), 1)
        ]  # returns real part only
    else:
        zeros = np.angle(polynomial_zeros) - 1j * np.log(np.abs(polynomial_zeros))
    return zeros


def fourier_series_stationary_points(a, b, return_y=False):
    """Returns the stationary points of the given Fourier series."""
    a_prime, b_prime = fourier_series_derivative(a, b)
    if np.allclose(a_prime, 0.0) and np.allclose(b_prime, 0.0):
        stat_x = [0.0]
    else:
        stat_x = fourier_series_zeros(a_prime, b_prime, only_real=True)
    if return_y:
        return stat_x, [fourier_series_from_coeffs(a, b)(x) for x in stat_x]
    else:
        return stat_x


def fourier_series_minimum(a, b, return_y=False):
    """Returns the minimum of the given Fourier series."""
    stat_x, stat_y = fourier_series_stationary_points(a, b, return_y=True)
    stat_x = np.array(stat_x)
    stat_y = np.array(stat_y)
    y_min = stat_y.min()
    mask = np.isclose(stat_y, y_min)
    x_close = stat_x[mask]
    y_close = stat_y[mask]
    idx_smallest = np.abs(x_close).argmin()  # Choose minimum closest to zero

    if return_y:
        return x_close[idx_smallest], y_close[idx_smallest]
    else:
        return x_close[idx_smallest]


### Visualization helpers: ###


def print_fourier_series(a, b):
    """Returns a latex string for the given Fourier series."""
    terms = [f"{a[0]:.2f}"]  # init with constant term
    for i, (a_i, b_i) in enumerate(zip(a[1:], b)):
        terms.append(f"{a_i:.2f} \\cos({i + 1}x)")
        terms.append(f"{b_i:.2f} \\sin({i + 1}x)")
    return " + ".join(terms)


def plot_fourier_series(a, b, stat_x=None, stat_y=None, ax=None):
    """Plots the stationary points of the given Fourier series."""
    if stat_x is None:
        stat_x, stat_y = fourier_series_stationary_points(a, b, return_y=True)
    if stat_y is None:
        stat_y = [fourier_series_from_coeffs(a, b)(x) for x in stat_x]
    if ax is None:
        _, ax = plt.subplots()
    xs = np.linspace(-2 * np.pi / 1.5, 2 * np.pi / 1.5, 10_000)

    # vertical lines at period boundaries
    ax.axvline(x=-np.pi, color="red", linestyle="--", alpha=0.5)
    ax.axvline(x=+np.pi, color="red", linestyle="--", alpha=0.5)
    # fine grid in plot:
    ax.grid(True, which="both")

    ax.plot(xs, [fourier_series_from_coeffs(a, b)(x) for x in xs])
    ax.plot(stat_x, stat_y, "x", markersize=7, markeredgewidth=2)
    global_min_idx = np.argmin(stat_y)
    # plot as red open circle:
    ax.plot(
        stat_x[global_min_idx],
        stat_y[global_min_idx],
        "o",
        color="red",
        fillstyle="none",
        markersize=10,
        markeredgewidth=2,
    )
