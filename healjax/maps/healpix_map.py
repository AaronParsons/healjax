"""
`HealpixMap` -- a data array living on a HEALPix sphere.

Ported from ``aipy.healpix`` with the aipy dependency removed.  Indexing
accepts pixel numbers, ``(theta, phi)`` or ``(x, y, z)``; assignment through
coordinates accumulates when several coordinates land in the same pixel.
"""

from __future__ import annotations

import numpy as np

from . import _optional
from .alm import Alm
from .base import HealpixBase, add2array, mk_arr

__all__ = ["HealpixMap", "default_fits_format_codes"]

default_fits_format_codes = {
    np.bool_: "L", np.uint8: "B", np.int16: "I", np.int32: "J",
    np.int64: "K", np.float32: "E", np.float64: "D",
    np.complex64: "C", np.complex128: "M",
}


class HealpixMap(HealpixBase):
    """Data array on a HEALPix sphere."""

    def __init__(self, *args, **kwargs):
        dtype = kwargs.pop("dtype", np.double)
        interp = kwargs.pop("interp", False)
        fromfits = kwargs.pop("fromfits", None)
        HealpixBase.__init__(self, *args, **kwargs)
        self._use_interpol = interp
        if fromfits is None:
            self.set_map(np.zeros((self.npix(),), dtype=dtype),
                         scheme=self.scheme())
        else:
            self.from_fits(fromfits)

    def set_interpol(self, onoff):
        """Enable or disable interpolation for coordinate-based reads."""
        self._use_interpol = onoff

    def set_map(self, data, scheme="RING"):
        """Assign map data, inferring nside from the length of the first axis."""
        try:
            nside = self.npix2nside(data.shape[0])
        except (AssertionError, ValueError):
            raise ValueError("First axis of data must have 12*N**2 elements.")
        self.set_nside_scheme(nside, scheme)
        self.map = data

    def get_map(self):
        return self.map

    def get_dtype(self):
        return self.map.dtype

    def change_scheme(self, scheme):
        """Reorder the map in place into the given ordering scheme."""
        assert scheme in ("RING", "NEST")
        if scheme == self.scheme():
            return
        i = self.nest_ring_conv(np.arange(self.npix()), scheme)
        self[i] = self.map
        self.set_nside_scheme(self.nside(), scheme)

    def __getitem__(self, crd):
        """
        Read data via ``hpm[crd]``, where *crd* is an array of pixel indices or
        a tuple of ``(th, phi)`` or ``(x, y, z)`` coordinate arrays.
        """
        if type(crd) is tuple:
            crd = [mk_arr(c, dtype=np.double) for c in crd]
            if self._use_interpol:
                px, wgts = self.crd2px(*crd, interpolate=True)
                wgts.shape += (1,) * (self.map.ndim - 1)
                return np.sum(self.map[px] * wgts, axis=1)
            px = self.crd2px(*crd)
        else:
            px = mk_arr(crd, dtype=np.int64)
        return self.map[px]

    def __setitem__(self, crd, val):
        """
        Write data via ``hpm[crd] = val``.  Repeated coordinates accumulate
        (scatter-add), matching aipy's behaviour for gridding.
        """
        if type(crd) is tuple:
            crd = [mk_arr(c, dtype=np.double) for c in crd]
            px = self.crd2px(*crd)
        else:
            if type(crd) is np.ndarray:
                assert len(crd.shape) == 1
            px = mk_arr(crd, dtype=int)
        if px.size == 1:
            if type(val) is np.ndarray:
                val = mk_arr(val, dtype=self.map.dtype)
            self.map[px] = val
        else:
            m = np.zeros_like(self.map)
            px = px.reshape(px.size, 1)
            cnt = np.zeros(self.map.shape, dtype=np.bool_)
            val = mk_arr(val, dtype=m.dtype)
            add2array(m, px, val)
            add2array(cnt, px, np.ones(val.shape, dtype=np.bool_))
            self.map = np.where(cnt, m, self.map)

    def from_hpm(self, hpm):
        """Initialise from another map, up- or down-grading the resolution."""
        if hpm.nside() < self.nside():
            interpol = hpm._use_interpol
            hpm.set_interpol(True)
            px = np.arange(self.npix())
            th, phi = self.px2crd(px, ncrd=2)
            self[px] = np.asarray(hpm[th, phi]).astype(self.get_dtype())
            hpm.set_interpol(interpol)
        elif hpm.nside() > self.nside():
            px = np.arange(hpm.npix())
            th, phi = hpm.px2crd(px, ncrd=2)
            self[th, phi] = np.asarray(hpm[px]).astype(self.get_dtype())
        else:
            if hpm.scheme() == self.scheme():
                self.map = hpm.map.astype(self.get_dtype())
            else:
                i = self.nest_ring_conv(np.arange(self.npix()), hpm.scheme())
                self.map = hpm.map[i].astype(self.get_dtype())

    def from_alm(self, alm):
        """Replace the map with the synthesis of *alm* (always RING ordered)."""
        self.set_map(alm.to_map(self.nside()), scheme="RING")

    def to_alm(self, lmax, mmax, iter=1):
        """Analyse this (RING-ordered) map into an `Alm`."""
        assert self.scheme() == "RING"
        alm = Alm(lmax, mmax)
        alm.from_map(self.map, iter)
        return alm

    # -- FITS I/O (optional astropy dependency) ----------------------------

    def from_fits(self, filename, hdunum=1, colnum=0):
        """Read a HEALPix map from a FITS binary table."""
        pyfits = _optional.fits()
        hdu = pyfits.open(filename)[hdunum]
        data = hdu.data.field(colnum)
        if not data.dtype.isnative:
            data = data.byteswap().view(data.dtype.newbyteorder())
        scheme = hdu.header["ORDERING"][:4]
        self.set_map(data, scheme=scheme)

    def to_fits(self, filename, format=None, clobber=True):
        """Write this map to a FITS binary table."""
        pyfits = _optional.fits()
        if format is None:
            format = default_fits_format_codes[self.get_dtype().type]
        hdu0 = pyfits.PrimaryHDU()
        col0 = pyfits.Column(name="signal", format=format,
                             array=np.asarray(self.map))
        cols = pyfits.ColDefs([col0])
        tbhdu = pyfits.BinTableHDU.from_columns(cols)
        self._set_fits_header(tbhdu.header)
        pyfits.HDUList([hdu0, tbhdu]).writeto(filename, overwrite=clobber)

    def _set_fits_header(self, hdr):
        hdr["PIXTYPE"] = ("HEALPIX", "HEALPIX pixelisation")
        scheme = "NESTED" if self.scheme() == "NEST" else self.scheme()
        hdr["ORDERING"] = (scheme, "Pixel ordering scheme")
        hdr["NSIDE"] = (self.nside(), "Resolution parameter for HEALPIX")
        hdr["FIRSTPIX"] = (0, "First pixel # (0 based)")
        hdr["LASTPIX"] = (self.npix() - 1, "Last pixel # (0 based)")
        hdr["INDXSCHM"] = ("IMPLICIT", "Indexing: IMPLICIT or EXPLICIT")
