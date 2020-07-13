"""Projection interpolation functions.

Derived from Splash: https://github.com/danieljprice/splash.
"""

import numba
import numpy as np
from numpy import ndarray

from ._coordinates import coord_transform, get_coord_info, get_pixel_limits
from ._kernels import RADKERNEL, RADKERNEL2, setup_integratedkernel, wfromtable


@numba.njit
def projection(
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
    npixx: int,
    npixy: int,
    pixwidthx: float,
    pixwidthy: float,
    normalise: bool,
    dscreen: float,
    zobserver: float,
) -> ndarray:
    """Interpolate particles to grid via projection.

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
    npixx
        The number of pixels in the x direction.
    npixy
        The number of pixels in the y direction.
    normalise
        Whether to normalize.
    dscreen
        The distance from observer to screen (for perspective
        rendering). If 0.0, then do not use perspective rendering.
    zobserver
        The z-position of the observer (for perspective rendering).

    Return
    ------
    datsmooth
        The data smoothed to a pixel grid.
    """
    coltable = setup_integratedkernel()

    datsmooth = np.zeros((npixx, npixy))
    datnorm = np.zeros((npixx, npixy))
    dx2i = np.zeros(npixx)
    xpix = np.zeros(npixx)
    term = 0.0

    use_perspective = np.abs(dscreen) > 0

    xminpix = xmin - 0.5 * pixwidthx
    yminpix = ymin - 0.5 * pixwidthy
    xmax = xmin + npixx * pixwidthx
    ymax = ymin + npixy * pixwidthy

    # Use a minimum smoothing length on the grid to make sure that particles
    # contribute to at least one pixel
    hmin = 0.5 * max(pixwidthx, pixwidthy)

    xpix = xminpix + np.arange(1, npixx + 1) * pixwidthx
    nsubgrid = 0
    nok = 0
    hminall = 1e10

    # Loop over particles
    for idx in range(npart):

        # Skip particles with itype < 0
        if itype[idx] < 0:
            continue

        # Set h related quantities
        hi = hh[idx]
        horigi = hi
        if not hi > 0.0:
            continue

        if use_perspective:
            # Skip particles outside of perspective
            if z[idx] > zobserver:
                continue
            zfrac = np.abs(dscreen / (z[idx] - zobserver))
            hi = hi * zfrac

        # Radius of the smoothing kernel
        radkern = RADKERNEL * hi

        # Cycle as soon as we know the particle does not contribute
        xi = x[idx]
        xpixmin = xi - radkern
        if xpixmin > xmax:
            continue
        xpixmax = xi + radkern
        if xpixmax < xmin:
            continue

        yi = y[idx]
        ypixmin = yi - radkern
        if ypixmin > ymax:
            continue
        ypixmax = yi + radkern
        if ypixmax < ymin:
            continue

        # Take resolution length as max of h and 1/2 pixel width
        if hi < hmin:
            hminall = min(hi, hminall)
            nsubgrid = nsubgrid + 1
            hsmooth = hmin
        else:
            hsmooth = hi
            nok = nok + 1
        radkern = RADKERNEL * hsmooth

        # Set kernel related quantities
        hi1 = 1.0 / hsmooth
        hi21 = hi1 * hi1
        termnorm = weight[idx] * horigi
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
            dx2i[ipix] = ((xpix[ipix] - xi) ** 2) * hi21

        for jpix in range(jpixmin, jpixmax):
            ypix = yminpix + jpix * pixwidthy
            dy = ypix - yi
            dy2 = dy * dy * hi21
            for ipix in range(ipixmin, ipixmax):
                # dx2 pre-calculated; dy2 pre-multiplied by hi21
                q2 = dx2i[ipix] + dy2
                # SPH kernel - integral through cubic spline
                # interpolate from a pre-calculated table
                if q2 < RADKERNEL2:
                    wab = wfromtable(q2, coltable)
                    # Calculate data value at this pixel using the summation interpolant
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

    # Warn about subgrid interpolation
    if nsubgrid > 1:
        nfull = int((xmax - xmin) / (hminall)) + 1
        if nsubgrid > 0.1 * nok:
            print('Warning: pixel size > 2h for ', nsubgrid, ' particles')
            print('need ', nfull, ' pixels for full resolution')

    # Return datsmooth
    return datsmooth.T


@numba.njit
def projection_non_cartesian(
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
    term = 0.0

    ixcoord, iycoord, izcoord, islengthx, islengthy, islengthz = get_coord_info(
        iplotx, iploty, iplotz, igeom
    )

    coltable = setup_integratedkernel()

    xpix = np.zeros(npixx)
    xci = np.zeros(3)
    xcoord = np.zeros(3)

    xminpix = xmin - 0.5 * pixwidthx
    yminpix = ymin - 0.5 * pixwidthy

    # Use a minimum smoothing length on the grid to make sure that particles
    # contribute to at least one pixel
    hmin = 0.0
    if islengthx:
        hmin = 0.5 * pixwidthx
    if islengthy:
        hmin = max(hmin, 0.5 * pixwidthy)

    # Loop over particles
    for idx in range(npart):

        # Skip particles with itype < 0
        if itype[idx] < 0:
            continue

        # Set h related quantities
        horigi = hh[idx]
        if not horigi > 0.0:
            continue
        hi = max(horigi, hmin)

        # Radius of the smoothing kernel
        radkern = RADKERNEL * hi

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

        # Set kernel related quantities
        hi1 = 1 / hi
        hi21 = hi1 * hi1

        # h gives the z length scale (NB: no perspective)
        if islengthz:
            termnorm = weight[idx] * horigi
        else:
            termnorm = weight[idx]
        term = termnorm * dat[idx]

        if islengthz:
            # Assume all pixels at same r as particlefor theta-phi
            xcoord[izcoord] = xi[izcoord]
        else:
            # Use phi=0 so get x = r cos(phi) = r
            xcoord[izcoord] = 0.0

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
                # z direction important if surface not flat (e.g. r slice)
                dz = xpix[izcoord] - xci[izcoord]

                dx2 = dx * dx
                dy2 = dy * dy
                q2 = (dx2 + dy2 + dz * dz) * hi21

                # SPH kernel - integral through cubic spline
                # interpolate from a pre-calculated table
                if q2 < RADKERNEL2:
                    wab = wfromtable(q2, coltable)
                    # Calculate data value at this pixel using the summation interpolant
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
