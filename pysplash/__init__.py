"""
Pysplash
========

Splash in pure Python with numba. Specifically:

> Smoothed particle hydrodynamics interpolation to a grid.

The core interpolation functions are derived from the Splash Fortran
code rewritten in Python with numba for performance.

All algorithms and numerical methods are derived from Splash. See
<https://github.com/danieljprice/splash>.
"""

# Canonical version number
__version__ = '0.0.1'

from .cross_section import (interpolate_cross_section,
                            interpolate_cross_section_non_cartesian)
from .projection import (interpolate_projection,
                         interpolate_projection_non_cartesian)
