"""
`HPM` -- the main user-facing HEALPix map container.

`HPM` is a `HealpixMap` whose hot paths (coordinate-to-pixel conversion and
interpolated reads) run through healjax's JAX kernels instead of healpy.  It
is the single class intended to replace both ``aipy.healpix.HealpixMap`` and
the per-package ``HPM`` copies that used to live in ``eigsep_sim`` and
``eigsep_data``.

Only the RING scheme is supported on the JAX interpolation path, matching
`healjax.get_interp_weights`.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ..healjax import ang2pix as _hj_ang2pix
from ..healjax import get_interpol as get_interp_weights
from ..healjax import vec2ang as _hj_vec2ang
from ..healjax import vec2pix as _hj_vec2pix
from ..interp.interpolation import interpolate_map, rotate_interpolate_and_sum
from .base import mk_arr
from .healpix_map import HealpixMap

__all__ = ["HPM", "vec2ang", "ang2pix", "vec2pix"]


def vec2ang(c1, c2, c3):
    """Cartesian to ``(theta, phi)``, elementwise over arrays."""
    return _hj_vec2ang(c1, c2, c3)


def ang2pix(scheme, nside, c1, c2):
    """Vectorised `healjax.ang2pix`, preserving the shape of the inputs."""
    c1, c2 = jnp.asarray(c1), jnp.asarray(c2)
    px_flat = jax.vmap(
        lambda th, ph: _hj_ang2pix(scheme, nside, th, ph)
    )(c1.ravel(), c2.ravel())
    return px_flat.reshape(c1.shape)


def vec2pix(scheme, nside, c1, c2, c3):
    """Vectorised `healjax.vec2pix`, preserving the shape of the inputs."""
    c1, c2, c3 = jnp.asarray(c1), jnp.asarray(c2), jnp.asarray(c3)
    px_flat = jax.vmap(
        lambda x, y, z: _hj_vec2pix(scheme, nside, x, y, z)
    )(c1.ravel(), c2.ravel(), c3.ravel())
    return px_flat.reshape(c1.shape)


class HPM(HealpixMap):
    """A `HealpixMap` with JAX-accelerated pixel lookup and interpolation."""

    def __init__(self, *args, **kwargs):
        HealpixMap.__init__(self, *args, **kwargs)
        self._refresh_jax()

    def _refresh_jax(self):
        """(Re)build the jitted kernels bound to the current nside/scheme."""
        scheme = self._scheme.lower()
        self.jax_ang2pix = jax.jit(partial(ang2pix, scheme, self._nside))
        self.jax_vec2pix = jax.jit(partial(vec2pix, scheme, self._nside))
        self.jax_vec2ang = jax.jit(vec2ang)

    def set_nside_scheme(self, nside=None, scheme=None):
        """Set resolution/scheme, rebuilding the jitted kernels to match."""
        HealpixMap.set_nside_scheme(self, nside=nside, scheme=scheme)
        self._refresh_jax()

    def crd2px(self, c1, c2, c3=None, interpolate=False):
        """
        Convert coordinates to pixel indices using healjax.

        With ``interpolate=True`` returns ``(px, wgts)`` of shape ``(N, 4)``
        each, holding the four neighbouring pixels and their weights.
        """
        if not interpolate:
            if c3 is None:
                return self.jax_ang2pix(c1, c2)
            return self.jax_vec2pix(c1, c2, c3)
        if c3 is not None:
            c1, c2 = self.jax_vec2ang(c1, c2, c3)
        if self._scheme == "NEST":
            raise NotImplementedError(
                "healjax interpolation supports RING ordering only; "
                "call change_scheme('RING') first."
            )
        px, wgts = get_interp_weights(c1, c2, self._nside)
        return px.T, wgts.T

    def __getitem__(self, crd):
        """
        Read data via ``hpm[crd]``, where *crd* is an array of pixel indices or
        a tuple of ``(th, phi)`` or ``(x, y, z)`` coordinate arrays.
        """
        if type(crd) is tuple:
            crd = [mk_arr(c, dtype=np.double) for c in crd]
            if self._use_interpol:
                return interpolate_map(self._nside, self.map, *crd)
            px = self.crd2px(*crd)
        else:
            px = mk_arr(crd, dtype=np.int64)
        return self.map[px]

    def rotate_interpolate_and_sum(self, sky, crds, rot_ms, chunk_size=16):
        """
        Beam-weighted sky integral for each rotation matrix in *rot_ms*.

        The rotations are processed in chunks of *chunk_size* to bound peak
        memory; see `healjax.interp.rotate_interpolate_and_sum` for the
        underlying kernel and argument shapes.

        Returns
        -------
        data_out : ndarray, shape (len(rot_ms), nfreq)
        """
        data_out = []
        for i in range(0, rot_ms.shape[0], chunk_size):
            data_out.append(
                rotate_interpolate_and_sum(
                    self._nside, self.map, sky, crds, rot_ms[i:i + chunk_size]
                )
            )
        return np.concatenate(data_out, axis=0)
