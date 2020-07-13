"""Cross-section interpolation functions.

Derived from Splash: https://github.com/danieljprice/splash.
"""

import numba
import numpy as np
from numpy import ndarray

from .kernels import w_cubic, RADKERNEL, RADKERNEL2, CNORMK3D
from .coordinates import coord_transform, get_coord_info, get_pixel_limits


@numba.njit
def interpolate_cross_section(
    x: ndarray,
    y: ndarray,
    z: ndarray,
    hh: ndarray,
    weight: ndarray,
    dat: ndarray,
    itype: ndarray,
    npart: int,
    xmin: float,
    ymin: float,
    zslice: float,
    npixx: int,
    npixy: int,
    pixwidthx: float,
    pixwidthy: float,
    normalise: bool,
) -> ndarray:
    """Interpolate particles to grid via cross section.

    Parameters
    ----------
    x
        The particle x positions.
    y
        The particle y positions.
    z
        The particle z positions.
    hh
        The particle smoothing length.
    weight
        The particle weight.
    dat
        The scalar data to interpolate.
    itype
        The particle type.
    npart
        The number of particles.
    xmin
        The minimum x position.
    ymin
        The minimum y position.
    zslice
        Cross section location.
    npixx
        The number of pixels in the x direction.
    npixy
        The number of pixels in the y direction.
    normalise
        Whether to normalize.

    Return
    ------
    datsmooth
        The data smoothed to a pixel grid.
    """
    datsmooth = np.zeros((npixx, npixy))
    datnorm = np.zeros((npixx, npixy))
    dx2i = np.zeros(npixx)
    const = CNORMK3D

    # Loop over particles
    for idx in range(npart):

        # Skip particles with itype < 0
        if itype[idx] < 0:
            continue

        # Set h related quantities
        hi = hh[idx]
        if not hi > 0.0:
            continue
        hi1 = 1.0 / hi
        hi21 = hi1 * hi1
        radkern = RADKERNEL * hi

        # For each particle, work out distance from the cross section slice
        dz = zslice - z[idx]
        dz2 = dz ** 2 * hi21

        # If this is < 2h then add the particle's contribution to the pixels
        # otherwise skip all this and start on the next particle
        if dz2 < RADKERNEL2:

            xi = x[idx]
            yi = y[idx]
            termnorm = const * weight[idx]
            term = termnorm * dat[idx]

            # Loop over pixels, adding the contribution from this particle
            # copy by quarters if all pixels within domain
            ipixmin = int((xi - radkern - xmin) / pixwidthx)
            ipixmax = int((xi + radkern - xmin) / pixwidthx) + 1
            jpixmin = int((yi - radkern - ymin) / pixwidthy)
            jpixmax = int((yi + radkern - ymin) / pixwidthy) + 1

            # Make sure they only contribute to pixels in the image
            # (note that this optimises much better than using min/max)
            if ipixmin < 0:
                ipixmin = 0
            if jpixmin < 0:
                jpixmin = 0
            if ipixmax > npixx:
                ipixmax = npixx
            if jpixmax > npixy:
                jpixmax = npixy

            # Precalculate an array of dx2 for this particle (optimisation)
            for ipix in range(ipixmin, ipixmax):
                dx2i[ipix] = ((xmin + (ipix - 0.5) * pixwidthx - xi) ** 2) * hi21 + dz2

            # Loop over pixels, adding the contribution from this particle
            for jpix in range(jpixmin, jpixmax):
                ypix = ymin + (jpix - 0.5) * pixwidthy
                dy = ypix - yi
                dy2 = dy * dy * hi21
                for ipix in range(ipixmin, ipixmax):
                    q2 = dx2i[ipix] + dy2
                    # SPH kernel - cubic spline
                    if q2 < RADKERNEL2:
                        wab = w_cubic(q2)
                        # Calculate data value at this pixel using the summation
                        # interpolant
                        datsmooth[ipix, jpix] = datsmooth[ipix, jpix] + term * wab
                        if normalise:
                            datnorm[ipix, jpix] = datnorm[ipix, jpix] + termnorm * wab

    # Normalise dat array
    if normalise:
        # Normalise everywhere (required if not using SPH weighting)
        for idxi in range(npixx):
            for idxj in range(npixy):
                if datnorm[idxi, idxj] > 0.0:
                    datsmooth[idxi, idxj] /= datnorm[idxi, idxj]

    # Return datsmooth
    return datsmooth.T


@numba.njit
def interpolate_cross_section_non_cartesian(
    x: ndarray,
    y: ndarray,
    z: ndarray,
    hh: ndarray,
    weight: ndarray,
    dat: ndarray,
    itype: ndarray,
    npart: int,
    xmin: float,
    ymin: float,
    zslice: float,
    npixx: int,
    npixy: int,
    pixwidthx: float,
    pixwidthy: float,
    normalise: bool,
    igeom: int,
    iplotx: int,
    iploty: int,
    iplotz: int,
    xorigin: ndarray,
) -> ndarray:
    """Interpolate particles to non-Cartesian grid via projection.

    Parameters
    ----------
    x
        The particle x positions in Cartesian coordinates.
    y
        The particle y positions in Cartesian coordinates.
    z
        The particle z positions in Cartesian coordinates.
    hh
        The particle smoothing length.
    weight
        The particle weight.
    dat
        The scalar data to interpolate.
    itype
        The particle type.
    npart
        The number of particles.
    xmin
        The minimum x position in non-Cartesian coordinates.
    ymin
        The minimum y position in non-Cartesian coordinates.
    zslice
        Cross section location.
    npixx
        The number of pixels in the x direction.
    npixy
        The number of pixels in the y direction.
    normalise
        Whether to normalize.
    igeom
        Integer representing the coordinate geometry:

        - 2: cylindrical
    iplotx
        Integer representing the x-coordinate for interpolation. 0, 1,
        or 2 for first, second, or third coordinate of chosen
        coordinate system. E.g. in cylindrical coords, 0 is 'R', 1 is
        'phi', and 2 is 'z'.
    iploty
        Integer representing the y-coordinate for interpolation. 0, 1,
        or 2 for first, second, or third coordinate of chosen
        coordinate system. E.g. in cylindrical coords, 0 is 'R', 1 is
        'phi', and 2 is 'z'.
    iplotz
        Integer representing the z-coordinate for interpolation. 0, 1,
        or 2 for first, second, or third coordinate of chosen
        coordinate system. E.g. in cylindrical coords, 0 is 'R', 1 is
        'phi', and 2 is 'z'.
    xorigin
        The coordinates of the origin as ndarray like (x, y, z).

    Return
    ------
    datsmooth
        The data smoothed to a pixel grid.
    """
    datsmooth = np.zeros((npixx, npixy))
    datnorm = np.zeros((npixx, npixy))

    const = CNORMK3D

    ixcoord, iycoord, izcoord, islengthx, islengthy, islengthz = get_coord_info(
        iplotx, iploty, iplotz, igeom
    )

    if not islengthz:
        raise ValueError('cross section not implemented when z is an angle')

    xminpix = xmin - 0.5 * pixwidthx
    yminpix = ymin - 0.5 * pixwidthy

    xpix = np.zeros(npixx)
    xci = np.zeros(3)
    xcoord = np.zeros(3)

    # Loop over particles
    for idx in range(npart):

        # Skip particles with itype < 0
        if itype[idx] < 0:
            continue

        # Set h related quantities
        hi = hh[idx]
        if not hi > 0.0:
            continue

        # Radius of the smoothing kernel
        radkern = RADKERNEL * hi

        # Set kernel related quantities
        hi1 = 1 / hi
        hi21 = hi1 * hi1

        # For each particle, work out distance from the cross section slice.
        dz = zslice - z[idx]
        dz2 = dz ** 2
        xcoord[izcoord] = 1

        # If this is < 2h then add the particle's contribution to the pixels
        # otherwise skip all this and start on the next particle
        if dz2 < RADKERNEL2:

            # Get limits of contribution from particle in cartesian space
            # xci is position in cartesian coordinates
            xci[0] = x[idx] + xorigin[0]
            xci[1] = y[idx] + xorigin[1]
            xci[2] = z[idx] + xorigin[2]
            xi, ipixmin, ipixmax, jpixmin, jpixmax, ierr = get_pixel_limits(
                xci,
                radkern,
                igeom,
                npixx,
                npixy,
                pixwidthx,
                pixwidthy,
                xmin,
                ymin,
                ixcoord,
                iycoord,
            )
            if ierr != 0:
                continue

            termnorm = const * weight[idx]
            term = termnorm * dat[idx]

            # Loop over pixels, adding the contribution from this particle
            for jpix in range(jpixmin, jpixmax):
                jp = np.mod(jpix, npixy)
                xcoord[iycoord] = yminpix + jp * pixwidthy

                for ipix in range(ipixmin, ipixmax):
                    ip = np.mod(ipix, npixx)
                    xcoord[ixcoord] = xminpix + ip * pixwidthx

                    # Transform to get location of pixel in cartesians
                    xpix = coord_transform(xcoord, igeom)

                    # Find distances using cartesians and perform interpolation
                    dy = xpix[iycoord] - xci[iycoord]
                    dx = xpix[ixcoord] - xci[ixcoord]

                    dx2 = dx * dx
                    dy2 = dy * dy
                    q2 = (dx2 + dy2 + dz * dz) * hi21

                    # SPH kernel - integral through cubic spline
                    # interpolate from a pre-calculated table
                    if q2 < RADKERNEL2:
                        wab = w_cubic(q2)
                        # Calculate data value at this pixel using the summation
                        # interpolant
                        datsmooth[ip, jp] = datsmooth[ip, jp] + term * wab
                        if normalise:
                            datnorm[ip, jp] = datnorm[ip, jp] + termnorm * wab

    # Normalise dat array
    if normalise:
        # Normalise everywhere (required if not using SPH weighting)
        for idxi in range(npixx):
            for idxj in range(npixy):
                if datnorm[idxi, idxj] > 0.0:
                    datsmooth[idxi, idxj] /= datnorm[idxi, idxj]

    # Return datsmooth
    return datsmooth.T
