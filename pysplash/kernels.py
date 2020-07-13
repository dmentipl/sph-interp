"""SPH kernels.

Derived from Splash: https://github.com/danieljprice/splash.
"""

import numba
import numpy as np

RADKERNEL = 2.0
RADKERNEL2 = 4.0
CNORMK3D = 1.0 / np.pi


@numba.njit
def w_cubic(q2: float) -> float:
    """Cubic spline kernel.

    Parameters
    ----------
    q2

    Returns
    -------
    w
    """
    w = 0.0
    if q2 < 1:
        q = np.sqrt(q2)
        w = 1.0 - 1.5 * q2 + 0.75 * q2 * q
    elif q2 < 4:
        q = np.sqrt(q2)
        w = 0.25 * (2.0 - q) ** 3
    return w
