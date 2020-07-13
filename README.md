sph-interp
==========

> Smoothed particle hydrodynamics interpolation to a grid.

The core interpolation functions are derived from the Splash Fortran code rewritten in Python with numba for performance.

For the original Splash source code, see <https://github.com/danieljprice/splash>.

Install
-------

Install from source.

```bash
git clone https://github.com/dmentipl/sph-interp
cd sph-interp
pip install -e .
```

Requirements
------------

Python 3.6+ with [numpy](https://numpy.org/) and [numba](http://numba.pydata.org/).

To-do
-----

- [x] add license
- [ ] add tests
- [x] modularise code
- [x] add non-Cartesian interpolation
- [x] add perspective rendering
- [ ] add opacity rendering
- [ ] add exact rendering
- [ ] add interpolate to 3D grid
- [ ] add missing docstrings
