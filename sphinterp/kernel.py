"""SPH kernels.

Derived from Splash: https://github.com/danieljprice/splash.
"""

import numba
import numpy as np


@numba.njit  # type: ignore
def cubic(q):
    """Cubic kernel function.

    The form of this function includes the "C_norm" factor. I.e.
    C_norm * f(q).

    Parameters
    ----------
    q
        The particle separation in units of smoothing length, i.e. r/h.

    Returns
    -------
    float
        C_norm * f(q) for the cubic kernel.
    """
    if q < 1:
        return 0.75 * q ** 3 - 1.5 * q ** 2 + 1
    elif q < 2:
        return -0.25 * (q - 2) ** 3
    else:
        return 0.0


@numba.njit  # type: ignore
def cubic_gradient(q):
    """Cubic kernel gradient function.

    The form of this function includes the "C_norm" factor. I.e.
    C_norm * f'(q).

    Parameters
    ----------
    q
        The particle separation in units of smoothing length, i.e. r/h.

    Returns
    -------
    float
        C_norm * f'(q) for the cubic kernel.
    """
    if q < 1:
        return q * (2.25 * q - 3.0)
    elif q < 2:
        return -0.75 * (q - 2.0) ** 2
    else:
        return 0.0


@numba.njit  # type: ignore
def quintic(q):
    """Quintic kernel function.

    The form of this function includes the "C_norm" factor. I.e.
    C_norm * f(q).

    Parameters
    ----------
    q
        The particle separation in units of smoothing length, i.e. r/h.

    Returns
    -------
    float
        C_norm * f(q) for the quintic kernel.
    """
    if q < 1:
        return -10 * q ** 5 + 30 * q ** 4 - 60 * q ** 2 + 66
    elif q < 2:
        return -((q - 3) ** 5) + 6 * (q - 2) ** 5
    elif q < 3:
        return -((q - 3) ** 5)
    else:
        return 0.0


@numba.njit  # type: ignore
def quintic_gradient(q):
    """Quintic kernel gradient function.

    The form of this function includes the "C_norm" factor. I.e.
    C_norm * f'(q).

    Parameters
    ----------
    q
        The particle separation in units of smoothing length, i.e. r/h.

    Returns
    -------
    float
        C_norm * f'(q) for the quintic kernel.
    """
    if q < 1:
        return q * (-50 * q ** 3 + 120 * q ** 2 - 120)
    elif q < 2:
        return -5 * (q - 3) ** 4 + 30 * (q - 2.0) ** 4
    elif q < 3:
        return -5 * (q - 3) ** 4
    else:
        return 0.0


@numba.njit  # type: ignore
def wendland_c4(q):
    """Wendland C4 kernel function.

    The form of this function includes the "C_norm" factor. I.e.
    C_norm * f(q).

    Parameters
    ----------
    q
        The particle separation in units of smoothing length, i.e. r/h.

    Returns
    -------
    float
        C_norm * f(q) for the Wendland C4 kernel.
    """
    if q < 2:
        return (-q / 2 + 1) ** 6 * (35 * q ** 2 / 12 + 3 * q + 1)
    else:
        return 0.0


@numba.njit  # type: ignore
def wendland_c4_gradient(q):
    """Wendland C4 kernel gradient function.

    The form of this function includes the "C_norm" factor. I.e.
    C_norm * f'(q).

    Parameters
    ----------
    q
        The particle separation in units of smoothing length, i.e. r/h.

    Returns
    -------
    float
        C_norm * f'(q) for the Wendland C4 kernel.
    """
    if q < 2:
        return (
            11.6666666666667 * q ** 2 * (0.5 * q - 1) ** 5
            + 4.66666666666667 * q * (0.5 * q - 1) ** 5
        )
    else:
        return 0.0


NAME = (
    'cubic',
    'quintic',
    'Wendland C4',
)

RADIUS = {
    'cubic': 2.0,
    'quintic': 3.0,
    'Wendland C4': 2.0,
}

NORM = {
    'cubic': 1 / np.pi,
    'quintic': 1 / (120 * np.pi),
    'Wendland C4': 495 / (256 * np.pi),
}

FUNCTION = {
    'cubic': cubic,
    'quintic': quintic,
    'Wendland C4': wendland_c4,
}

GRADIENT = {
    'cubic': cubic_gradient,
    'quintic': quintic_gradient,
    'Wendland C4': wendland_c4_gradient,
}
