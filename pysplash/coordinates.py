"""Coordinate utilities.

Derived from Splash: https://github.com/danieljprice/splash.
"""

# TODO: add docstrings

from typing import Tuple

import numba
import numpy as np
from numpy import ndarray

ICYLINDRICAL = 2
ISPHERICAL = 3


@numba.njit
def get_coord_info(
    iplotx: int, iploty: int, iplotz: int, igeom: int
) -> Tuple[int, int, int, bool, bool, bool]:
    ixcoord = iplotx
    iycoord = iploty
    izcoord = iplotz

    if ixcoord < 0 or ixcoord > 2:
        raise ValueError('Cannot find x coordinate offset')
    if iycoord < 0 or iycoord > 2:
        raise ValueError('Cannot find y coordinate offset')
    if izcoord < 0 or izcoord > 2:
        raise ValueError('Cannot find z coordinate offset')

    islengthx = coord_is_length(ixcoord, igeom)
    islengthy = coord_is_length(iycoord, igeom)
    islengthz = coord_is_length(izcoord, igeom)

    return ixcoord, iycoord, izcoord, islengthx, islengthy, islengthz


@numba.njit
def get_pixel_limits(
    xci: ndarray,
    radkern: float,
    igeom: int,
    npixx: int,
    npixy: int,
    pixwidthx: float,
    pixwidthy: float,
    xmin: float,
    ymin: float,
    ixcoord: int,
    iycoord: int,
) -> Tuple[ndarray, int, int, int, int, int]:
    ierr = 0

    # Get limits of rendering in new coordinate system
    xi, xpixmin, xpixmax = get_coord_limits(radkern, xci, igeom)

    # Now work out contributions to pixels in the the transformed space
    ipixmax = int((xpixmax[ixcoord] - xmin) / pixwidthx) + 1
    if ipixmax < 0:
        ierr = 1
    jpixmax = int((xpixmax[iycoord] - ymin) / pixwidthy) + 1
    if jpixmax < 0:
        ierr = 2

    ipixmin = int((xpixmin[ixcoord] - xmin) / pixwidthx)
    if ipixmin > npixx:
        ierr = 3
    jpixmin = int((xpixmin[iycoord] - ymin) / pixwidthy)
    if jpixmin > npixy:
        ierr = 4

    if not coord_is_periodic(ixcoord, igeom):
        # Make sure they only contribute to pixels in the image
        if ipixmin < 0:
            ipixmin = 0
        if ipixmax > npixx:
            ipixmax = npixx

    if not coord_is_periodic(iycoord, igeom):
        # Note that this optimises much better than using min/max
        if jpixmin < 0:
            jpixmin = 0
        if jpixmax > npixy:
            jpixmax = npixy

    return xi, ipixmin, ipixmax, jpixmin, jpixmax, ierr


@numba.njit
def coord_is_periodic(ix: int, igeom: int) -> bool:
    coord_is_periodic = False
    if igeom == ICYLINDRICAL and ix == 1:
        coord_is_periodic = True
    elif igeom == ISPHERICAL and ix == 1:
        coord_is_periodic = True
    return coord_is_periodic


@numba.njit
def coord_is_length(ix: int, igeom: int) -> bool:
    if igeom == ICYLINDRICAL:
        if ix in (0, 2):
            return True
        elif ix == 1:
            return False
    elif igeom == ISPHERICAL:
        if ix == 0:
            return True
        elif ix in (1, 2):
            return False
    raise ValueError('Cannot determine if coordinate is a length or angle')


@numba.njit
def get_coord_limits(
    rad: float, xin: ndarray, itypein: int
) -> Tuple[ndarray, ndarray, ndarray]:

    xout = np.zeros(3)
    xmin = np.zeros(3)
    xmax = np.zeros(3)

    if itypein == ICYLINDRICAL:
        r = np.sqrt(xin[0] ** 2 + xin[1] ** 2)
        xout[0] = r
        xout[1] = np.arctan2(xin[1], xin[0])
        xout[2] = xin[2]
        xmin[0] = np.max(r - rad, 0.0)
        xmax[0] = r + rad
        if r > 0 and xmin[0] > 0:
            dphi = np.arctan(rad / r)
            xmin[1] = xout[1] - dphi
            xmax[1] = xout[1] + dphi
        else:
            xmin[1] = -np.pi
            xmax[1] = np.pi
        xmin[2] = xout[2] - rad
        xmax[2] = xout[2] + rad

    elif itypein == ISPHERICAL:
        r = np.linalg.norm(xin)
        xout[0] = r
        xout[1] = np.arctan2(xin[1], xin[0])
        if r > 0:
            xout[2] = np.arccos(xin[2] / r)
        else:
            xout[2] = 0
        xmin[0] = np.max(r - rad, 0)
        xmax[0] = r + rad
        if r > 0 and xmin[0] > 0:
            rcyl = np.sqrt(xin[0] ** 2 + xin[1] ** 2)
            if rcyl > rad:
                dphi = np.arcsin(rad / rcyl)
                xmin[1] = xout[1] - dphi
                xmax[1] = xout[1] + dphi
            else:
                xmin[1] = -np.pi
                xmax[1] = np.pi
            dtheta = np.arcsin(rad / r)
            xmin[2] = xout[2] - dtheta
            xmax[2] = xout[2] + dtheta
            xmin[2] = np.max(xmin[2], 0)
            xmax[2] = np.min(xmax[2], np.pi)
        else:
            xmin[1] = -np.pi
            xmax[1] = np.pi
            xmin[2] = 0
            xmax[2] = np.pi

    else:
        raise ValueError('Cannot determine coordinate limits')

    return xout, xmin, xmax


@numba.njit
def coord_transform(xin: ndarray, itypein: int) -> ndarray:
    xout = np.zeros(3)
    if itypein == ICYLINDRICAL:
        # Input is cylindrical polars, output is cartesian
        xout[0] = xin[0] * np.cos(xin[1])
        xout[1] = xin[0] * np.sin(xin[1])
        xout[2] = xin[2]
    elif itypein == ISPHERICAL:
        xout[0] = np.linalg.norm(xin)
        xout[1] = np.arctan2(xin[1], xin[0])
        if xout[0] > 0:
            xout[2] = np.arccos(xin[2] / xout[0])
        else:
            xout[2] = 0
    else:
        raise ValueError('Cannot determine coordinate transform')
    return xout
