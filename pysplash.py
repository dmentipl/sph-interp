"""Splash interpolation functions.

Derived from Splash: https://github.com/danieljprice/splash.
"""

from typing import Tuple

import numba
import numpy as np
from numpy import ndarray

__version__ = '0.0.1'

RADKERNEL = 2.0
RADKERNEL2 = 4.0
CNORMK3D = 1.0 / np.pi

NPTS = 100
MAXCOLTABLE = 1000
DQ2TABLE = RADKERNEL2 / MAXCOLTABLE
DDQ2TABLE = 1.0 / DQ2TABLE

IVERBOSE = -1

ICYLINDRICAL = 2
ISPHERICAL = 3


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


@numba.njit
def interpolate_projection(
    x: ndarray,
    y: ndarray,
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
) -> ndarray:
    """Interpolate particles to grid via projection.

    Parameters
    ----------
    x
        The particle x positions.
    y
        The particle y positions.
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
        if nsubgrid > 0.1 * nok and IVERBOSE > -1:
            print('Warning: pixel size > 2h for ', nsubgrid, ' particles')
            print('need ', nfull, ' pixels for full resolution')

    # Return datsmooth
    return datsmooth.T


@numba.njit
def interpolate_projection_non_cartesian(
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

    ixcoord, iycoord, izcoord, islengthx, islengthy, islengthz = _get_coord_info(
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
        xi, ipixmin, ipixmax, jpixmin, jpixmax, ierr = _get_pixel_limits(
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
                xpix = _coord_transform(xcoord, igeom)

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

    ixcoord, iycoord, izcoord, islengthx, islengthy, islengthz = _get_coord_info(
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
            xi, ipixmin, ipixmax, jpixmin, jpixmax, ierr = _get_pixel_limits(
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
                    xpix = _coord_transform(xcoord, igeom)

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


@numba.njit
def _get_coord_info(
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

    islengthx = _coord_is_length(ixcoord, igeom)
    islengthy = _coord_is_length(iycoord, igeom)
    islengthz = _coord_is_length(izcoord, igeom)

    return ixcoord, iycoord, izcoord, islengthx, islengthy, islengthz


@numba.njit
def _get_pixel_limits(
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
    xi, xpixmin, xpixmax = _get_coord_limits(radkern, xci, igeom)

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

    if not _coord_is_periodic(ixcoord, igeom):
        # Make sure they only contribute to pixels in the image
        if ipixmin < 0:
            ipixmin = 0
        if ipixmax > npixx:
            ipixmax = npixx

    if not _coord_is_periodic(iycoord, igeom):
        # Note that this optimises much better than using min/max
        if jpixmin < 0:
            jpixmin = 0
        if jpixmax > npixy:
            jpixmax = npixy

    return xi, ipixmin, ipixmax, jpixmin, jpixmax, ierr


@numba.njit
def _coord_is_periodic(ix: int, igeom: int) -> bool:
    coord_is_periodic = False
    if igeom == ICYLINDRICAL and ix == 1:
        coord_is_periodic = True
    elif igeom == ISPHERICAL and ix == 1:
        coord_is_periodic = True
    return coord_is_periodic


@numba.njit
def _coord_is_length(ix: int, igeom: int) -> bool:
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
def _get_coord_limits(
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
def _coord_transform(xin: ndarray, itypein: int) -> ndarray:
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
