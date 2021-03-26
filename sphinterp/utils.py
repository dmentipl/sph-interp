"""Utils."""

import numpy as np
from numpy import ndarray


def distance_from_plane(
    x: ndarray, y: ndarray, z: ndarray, normal: ndarray, height: float = 0
) -> ndarray:
    """Calculate distance from a plane.

    Parameters
    ----------
    x
        The x-positions.
    y
        The y-positions.
    z
        The z-positions.
    normal
        The normal vector describing the plane (x, y, z).
    height
        The height of the plane above the origin.

    Return
    ------
    The distance from the plane of each point.
    """
    a, b, c = normal
    d = height
    return np.abs((a * x + b * y + c * z + d) / np.sqrt(a ** 2 + b ** 2 + c ** 2))
