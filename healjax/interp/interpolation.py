"""
JAX-accelerated HEALPix map interpolation.

These kernels are jitted with ``nside`` as a static argument, so calling them
with a traced ``nside`` will fail -- pass a plain Python ``int``.  Only the
RING scheme is supported, matching `healjax.get_interp_weights`.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from ..healjax import get_interpol as get_interp_weights
from ..healjax import vec2ang

__all__ = ["interpolate_map", "rotate_interpolate_and_sum"]


@partial(jax.jit, static_argnums=(0,))
def interpolate_map(nside, map_data, c1, c2, c3=None):
    """
    Bilinearly interpolate a HEALPix map at arbitrary coordinates.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter (static under ``jit``).
    map_data : array, shape (npix, ...)
        Map values in RING ordering.  Trailing axes (e.g. frequency) are
        carried through untouched.
    c1, c2 : array
        Either ``(theta, phi)`` when *c3* is omitted, or ``(x, y)`` when *c3*
        is given.
    c3 : array, optional
        The z component, selecting the Cartesian calling convention.

    Returns
    -------
    interp_data : array, shape ``c1.shape + map_data.shape[1:]``
        Weighted sum over the four neighbouring pixels.
    """
    if c3 is not None:  # translate xyz to th/phi
        c1, c2 = vec2ang(c1, c2, c3)
    px, wgts = get_interp_weights(c1, c2, nside)
    slicing = (slice(None),) * wgts.ndim + (None,) * (map_data.ndim - 1)
    return jnp.sum(map_data[px] * wgts[slicing], axis=0)


@partial(jax.jit, static_argnums=(0,))
def rotate_interpolate_and_sum(nside, map_data, sky, crds, rot_ms):
    """
    Rotate coordinates, interpolate a beam, and take a beam-weighted sky sum.

    For each rotation matrix in *rot_ms* the pointing directions *crds* are
    rotated, *map_data* (the beam) is interpolated at the rotated directions,
    and the beam-weighted average of *sky* is returned.

    Parameters
    ----------
    nside : int
        HEALPix resolution of *map_data* (static under ``jit``).
    map_data : array, shape (npix, nfreq)
        Beam map in RING ordering.
    sky : array, shape (ncrd, nfreq) or (ncrd, 1)
        Sky brightness at each coordinate in *crds*.
    crds : array, shape (3, ncrd)
        Cartesian unit vectors to be rotated.
    rot_ms : array, shape (nrot, 3, 3)
        Stack of rotation matrices, e.g. from `healjax.coord.eq2top_m`.

    Returns
    -------
    data_out : array, shape (nrot, nfreq)
    """
    def body(_, rot_m):
        tx, ty, tz = rot_m @ crds
        wgt = interpolate_map(nside, map_data, tx, ty, tz)
        val = jnp.sum(wgt * sky, axis=0) / jnp.sum(wgt, axis=0)
        return None, val

    _, data_out = jax.lax.scan(body, None, rot_ms)
    return data_out
