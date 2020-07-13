"""SPH kernels.

Derived from Splash: https://github.com/danieljprice/splash.
"""

import numba
import numpy as np
from numpy import ndarray

RADKERNEL = 2.0
RADKERNEL2 = 4.0
CNORMK3D = 1.0 / np.pi

NPTS = 100
MAXCOLTABLE = 1000
DQ2TABLE = RADKERNEL2 / MAXCOLTABLE
DDQ2TABLE = 1.0 / DQ2TABLE


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


@numba.njit
def setup_integratedkernel() -> ndarray:
    """Set up integrated kernel.

    Tabulates the integral through the cubic spline kernel tabulated in
    (r/h)**2 so that sqrt is not necessary.

    Returns
    -------
    coltable
    """
    coltable = np.zeros(MAXCOLTABLE + 1)

    for idx in range(MAXCOLTABLE):
        # Tabulate for (cylindrical) r**2 between 0 and RADKERNEL**2
        rxy2 = idx * DQ2TABLE

        # Integrate z between 0 and sqrt(RADKERNEL^2 - rxy^2)
        deltaz = np.sqrt(RADKERNEL2 - rxy2)
        dz = deltaz / (NPTS - 1)
        coldens = 0
        for j in range(NPTS):
            z = j * dz
            q2 = rxy2 + z * z
            wkern = w_cubic(q2)
            if j == 0 or j == NPTS - 1:
                coldens = coldens + 0.5 * wkern * dz
            else:
                coldens = coldens + wkern * dz
        coltable[idx] = 2.0 * coldens * CNORMK3D
    coltable[MAXCOLTABLE] = 0.0

    return coltable


@numba.njit
def wfromtable(q2: float, coltable: ndarray) -> float:
    """Interpolate from integrated kernel table values to give w(q).

    Parameters
    ----------
    q2
    coltable

    Returns
    -------
    w
    """
    # Find nearest index in table
    index = int(q2 * DDQ2TABLE)
    index1 = min(index, MAXCOLTABLE)

    # Find increment along from this index
    dxx = q2 - index * DQ2TABLE

    # Find gradient
    dwdx = (coltable[index1] - coltable[index]) * DDQ2TABLE

    # Compute value of integrated kernel
    wfromtable = coltable[index] + dwdx * dxx

    return wfromtable
