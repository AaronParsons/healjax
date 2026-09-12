"""
`HealpixBase` -- pixel-scheme bookkeeping for the HEALPix sphere.

Originally ported from ``aipy.healpix``; this version has no aipy dependency.
Resolution arithmetic (``nside``/``npix``/``order``) is pure numpy, so a map
object can always be constructed.  Coordinate <-> pixel conversion delegates
to healpy, which is an optional dependency (see `HPM` for the JAX path).
"""

from __future__ import annotations

import numpy as np

from . import _optional

__all__ = ["HealpixBase", "HEALPIX_MODES", "mk_arr", "add2array"]

HEALPIX_MODES = ("RING", "NEST")


def mk_arr(val, dtype=np.double):
    """Coerce *val* to a flat array of *dtype*, passing ndarrays through."""
    if type(val) is np.ndarray:
        return val.astype(dtype)
    return np.array(val, dtype=dtype).flatten()


def add2array(a, ind, data):
    """
    Scatter-add *data* into *a* at the multi-dimensional indices in *ind*.

    Semantics match the aipy C extension ``utils.add2array``: repeated indices
    accumulate, unlike plain NumPy fancy-index assignment which keeps only the
    last write for a repeated index.

    Parameters
    ----------
    a : ndarray
        Target array, modified in place.
    ind : ndarray, shape (N, a.ndim)
        Multi-dimensional indices, one row per element of *data*.
    data : ndarray, shape (N,)
        Values to scatter-add.
    """
    if a.ndim == 1:
        np.add.at(a, ind[:, 0], data)
    else:
        idx = tuple(ind[:, j] for j in range(ind.shape[1]))
        np.add.at(a, idx, data)


class HealpixBase:
    """Functionality related to the HEALPix pixelisation."""

    def __init__(self, nside=1, scheme="RING"):
        self._nside = nside
        self._scheme = scheme

    # -- resolution arithmetic (no optional dependencies) ------------------

    def npix2nside(self, npix):
        """Nside for a map of *npix* pixels; raises for invalid counts."""
        npix = int(npix)
        nside = int(np.round(np.sqrt(npix / 12))) if npix > 0 else 0
        if nside < 1 or 12 * nside * nside != npix:
            raise ValueError(f"npix={npix} is not 12*nside**2")
        return nside

    def set_nside_scheme(self, nside=None, scheme=None):
        """Set resolution and/or ordering scheme, validating both."""
        if nside is not None:
            pow2 = np.log2(nside)
            assert pow2 == np.around(pow2)
            self._nside = int(nside)
        if scheme is not None:
            assert scheme in HEALPIX_MODES
            self._scheme = scheme

    def order(self):
        """log2(nside)."""
        return int(np.log2(self._nside))

    def nside(self):
        return self._nside

    def npix(self):
        return 12 * self._nside * self._nside

    def scheme(self):
        return self._scheme

    # -- pixel/coordinate conversion (healpy backed) -----------------------

    def nest_ring_conv(self, px, scheme):
        """
        Translate pixel numbers *px* into the given *scheme*.

        Also records *scheme* as this object's current scheme, matching the
        aipy behaviour that callers such as `change_scheme` rely on.
        """
        healpy = _optional.healpy()
        mode = {"RING": healpy.nest2ring, "NEST": healpy.ring2nest}
        if scheme != self._scheme:
            px = mode[scheme](self._nside, px)
        self._scheme = scheme
        return px

    def crd2px(self, c1, c2, c3=None, interpolate=False):
        """
        Convert coordinates to pixel indices.

        If only *c1*, *c2* are provided they are read as ``theta, phi``; if
        *c3* is also given they are read as ``x, y, z``.  With
        ``interpolate=True`` return ``(px, wgts)``, each row holding the four
        neighbouring pixels and their weights.
        """
        healpy = _optional.healpy()
        is_nest = self._scheme == "NEST"
        if not interpolate:
            if c3 is None:
                return healpy.ang2pix(self._nside, c1, c2, nest=is_nest)
            return healpy.vec2pix(self._nside, c1, c2, c3, nest=is_nest)
        if c3 is not None:
            c1, c2 = healpy.vec2ang(np.array([c1, c2, c3]).T)
        px, wgts = healpy.get_interp_weights(self._nside, c1, c2, nest=is_nest)
        return px.T, wgts.T

    def px2crd(self, px, ncrd=3):
        """Pixel numbers to coordinates: ``ncrd=2`` gives theta/phi, ``ncrd=3``
        gives x/y/z."""
        healpy = _optional.healpy()
        is_nest = self._scheme == "NEST"
        assert ncrd in (2, 3)
        if ncrd == 2:
            return healpy.pix2ang(self._nside, px, nest=is_nest)
        return healpy.pix2vec(self._nside, px, nest=is_nest)
