sph-interp
==========

> Smoothed particle hydrodynamics interpolation to a grid.

The core functionality of sph-interp is a set of interpolation functions derived from the Splash Fortran code rewritten in Python. It uses numba, a Python JIT compiler, for performance on par with the Fortran code.

For the original Splash source code, see <https://github.com/danieljprice/splash>.

Usage
-----

Import sph-interp.

```python
>>> import sphinterp as interp
```

Interpolate particles via projection.

```python
>>> pixel_grid = interp.projection(x, y, z, h, weight, data, ...)
```

Install
-------

Install from PyPI with pip.

```bash
python -m pip install sphinterp
```

Install from source.

```bash
git clone https://github.com/dmentipl/sph-interp
cd sph-interp
pip install -e .
```

Requirements
------------

Python 3.6+ with [numpy](https://numpy.org/) and [numba](http://numba.pydata.org/).

License
-------

sph-interp is licensed under GPL-2.0 following Splash. See [LICENSE](https://github.com/dmentipl/sph-interp/blob/master/LICENSE) for details.

Citation
--------

If you use sph-interp in a publication, please cite the [Splash paper](https://ui.adsabs.harvard.edu/abs/2007PASA...24..159P).

> Price, D. J., 2007, PASA, 24, 159

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
