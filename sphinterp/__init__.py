"""
sph-interp
==========

> Smoothed particle hydrodynamics interpolation to a grid.

The core interpolation functions are derived from the Splash Fortran
code rewritten in Python with numba for performance.

For the original Splash source code, see
<https://github.com/danieljprice/splash>.
"""

# Canonical version number
__version__ = '0.0.1'

from ._cross_section import cross_section, cross_section_non_cartesian
from ._projection import projection, projection_non_cartesian
