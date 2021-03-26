"""Interpolation to a pixel grid.

There are two functions: one for interpolation of scalar fields, and one
for interpolation of vector fields.
"""

from typing import Tuple

import numpy as np
from numpy import ndarray

from .projection import projection as interp_proj
from .slice import slice as interp_slice

Extent = Tuple[float, float, float, float]

NUM_PIXELS = (512, 512)


def interpolate(
    *,
    quantity: ndarray,
    x_coordinate: ndarray,
    y_coordinate: ndarray,
    dist_from_slice: ndarray,
    interp: str = 'projection',
    extent: Extent,
    smoothing_length: ndarray,
    particle_mass: ndarray,
    hfact: float,
    weighted: bool = None,
    num_pixels: Tuple[float, float] = NUM_PIXELS,
) -> ndarray:
    """Interpolate scalar quantity to a Cartesian pixel grid.

    Parameters
    ----------
    quantity
        A scalar quantity on the particles to interpolate.
    x_coordinate
        Particle coordinate for x-axis in interpolation.
    y_coordinate
        Particle coordinate for y-axis in interpolation.
    dist_from_slice
        The distance from the screen or cross section slice.
    interp
        The interpolation type. Default is 'projection'.

        - 'projection' : 2d interpolation via projection to xy-plane
        - 'slice' : 3d interpolation via cross-section slice.
    extent
        The range in the x- and y-direction as (xmin, xmax, ymin, ymax).
    smoothing_length
        The smoothing length on each particle.
    particle_mass
        The particle mass on each particle.
    hfact
        The smoothing length factor.
    weighted
        Use density weighted interpolation. Default is off.
    num_pixels
        The pixel grid to interpolate the scalar quantity to, as
        (npixx, npixy). Default is (512, 512).

    Returns
    -------
    ndarray
        An array of scalar quantities interpolated to a pixel grid with
        shape (npixx, npixy).
    """
    if not quantity.ndim == 1:
        raise ValueError('quantity.ndim > 1: can only interpolate scalar quantity')

    if interp == 'projection':
        do_slice = False
    elif interp == 'slice':
        do_slice = True
    else:
        raise ValueError('interp must be "projection" or "slice"')

    normalise = False
    if weighted is None:
        weighted = False
    if weighted:
        normalise = True

    npixx, npixy = num_pixels
    xmin, ymin = extent[0], extent[2]
    pixwidthx = (extent[1] - extent[0]) / npixx
    pixwidthy = (extent[3] - extent[2]) / npixy
    npart = len(smoothing_length)

    itype = np.ones(smoothing_length.shape)
    if weighted:
        weight = particle_mass / smoothing_length ** 3
    else:
        weight = hfact ** -3 * np.ones(smoothing_length.shape)

    if do_slice:
        interpolated_data = interp_slice(
            x=x_coordinate,
            y=y_coordinate,
            dslice=dist_from_slice,
            hh=smoothing_length,
            weight=weight,
            dat=quantity,
            itype=itype,
            npart=npart,
            xmin=xmin,
            ymin=ymin,
            npixx=npixx,
            npixy=npixy,
            pixwidthx=pixwidthx,
            pixwidthy=pixwidthy,
            normalise=normalise,
        )
    else:
        # Note that dscreen and zobserver are set to zero.
        # This turns perspective rendering off.
        # TODO: turn perspective rendering on.
        interpolated_data = interp_proj(
            x=x_coordinate,
            y=y_coordinate,
            z=dist_from_slice,
            hh=smoothing_length,
            weight=weight,
            dat=quantity,
            itype=itype,
            npart=npart,
            xmin=xmin,
            ymin=ymin,
            npixx=npixx,
            npixy=npixy,
            pixwidthx=pixwidthx,
            pixwidthy=pixwidthy,
            normalise=normalise,
            dscreen=0.0,
            zobserver=0.0,
        )

    return interpolated_data
