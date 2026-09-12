"""
Coordinate transforms for spherical astronomy.

Vectors are 3-dimensional with unit magnitude, representing a point on the
sphere.  Angles are in radians unless a function name says otherwise.

Every function here dispatches on its inputs: pass NumPy arrays (or Python
scalars) and you get NumPy back; pass JAX arrays (or trace under ``jax.jit``)
and you get JAX back.  This lets the same implementation serve both the
NumPy-based map classes and the jitted interpolation kernels.

Historically ``rot_m`` was copy-pasted into three separate EIGSEP packages
(``eigsep_sim.coord``, ``eigsep_terrain.utils``, ``eigsep_data.beam_sim``).
This module is the single implementation those should now import.
"""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

__all__ = [
    "rot_m",
    "xyz2thphi",
    "thphi2xyz",
    "eq2top_m",
    "top2eq_m",
    "eq2radec",
    "radec2eq",
    "latlong2xyz",
    "top2azalt",
    "azalt2top",
    "angles_to_coord",
]


def _is_jax(*args):
    """True if any argument is a JAX array or tracer."""
    return any(isinstance(a, jax.Array) for a in args)


def _xp(*args):
    """Return the array module (``jnp`` or ``np``) appropriate for *args*."""
    return jnp if _is_jax(*args) else np


def _unpack3(x, y, z):
    """Accept either ``f(xyz)`` or ``f(x, y, z)`` calling conventions."""
    if y is None and z is None:
        x, y, z = x
    elif y is None or z is None:
        raise TypeError("provide either a single (3, ...) array or x, y, z")
    return x, y, z


def _unpack2(a, b):
    """Accept either ``f(th_phi)`` or ``f(th, phi)`` calling conventions."""
    if b is None:
        a, b = a
    return a, b


# ---------------------------------------------------------------------------
# Rotations
# ---------------------------------------------------------------------------

def rot_m(ang, vec):
    """
    Rotation matrix for rotation by *ang* about the axis *vec* (Rodrigues).

    Follows the right-hand rule.  Both arguments may be batched, in which case
    a stack of matrices with leading batch axis is returned.  The matrix
    carries a scaling of ``|vec|``, so normalise *vec* for a pure rotation.

    Parameters
    ----------
    ang : float or array_like
        Rotation angle(s) in radians.
    vec : array_like, shape (..., 3)
        Rotation axis (or axes), with the components along the last axis.

    Returns
    -------
    rm : ndarray, shape (3, 3) or (N, 3, 3)
        NumPy array for NumPy input, JAX array for JAX input.
    """
    xp = _xp(ang, vec)
    if not isinstance(vec, (np.ndarray, jax.Array)):
        vec = xp.asarray(vec)

    c = xp.cos(ang)
    s = xp.sin(ang)
    C = 1 - c
    x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
    xs, ys, zs = x * s, y * s, z * s
    xC, yC, zC = x * C, y * C, z * C
    xyC, yzC, zxC = x * yC, y * zC, z * xC
    rows = [[x * xC + c, xyC - zs, zxC + ys],
            [xyC + zs, y * yC + c, yzC - xs],
            [zxC - ys, yzC + xs, z * zC + c]]
    rm = xp.array(rows) if xp is jnp else np.array(rows, dtype=np.double)
    if rm.ndim > 2:
        axes = list(range(rm.ndim))
        return rm.transpose(axes[-1:] + axes[:-1])
    return rm


# ---------------------------------------------------------------------------
# Cartesian <-> spherical
# ---------------------------------------------------------------------------

def xyz2thphi(x, y=None, z=None, *, dtype=None, return_mask=False,
              masked_fill=0.0):
    """
    Convert Cartesian vectors to spherical angles ``(theta, phi)``.

    ``theta`` is the polar angle measured from +z; ``phi`` is the azimuth from
    +x, counter-clockwise about +z.

    Parameters
    ----------
    x : array_like
        Either the x component, or -- when *y* and *z* are omitted -- a
        ``(3, ...)`` array (or length-3 sequence) holding ``[x, y, z]``.
    y, z : array_like, optional
        The remaining components.
    dtype : dtype, optional
        Output dtype.  Defaults to ``np.double`` on the NumPy path and to the
        natural promoted float type on the JAX path.
    return_mask : bool
        Also return a boolean mask of shape ``(2, ...)``, propagated from the
        mask of a masked-array *x*.
    masked_fill : float
        Value substituted for masked elements before the computation.

    Returns
    -------
    out : ndarray, shape (2, ...)
        Stacked ``[theta, phi]``.
    mask : ndarray, shape (2, ...)
        Only when ``return_mask=True``.
    """
    x, y, z = _unpack3(x, y, z)

    in_mask = None
    if isinstance(x, np.ma.MaskedArray):
        in_mask = np.ma.getmaskarray(x)
        x = np.ma.filled(x, masked_fill)
        y = np.ma.filled(y, masked_fill)
        z = np.ma.filled(z, masked_fill)

    xp = _xp(x, y, z)
    x, y, z = xp.asarray(x), xp.asarray(y), xp.asarray(z)

    r = xp.hypot(x, y)
    phi = xp.arctan2(y, x)
    th = xp.arctan2(r, z)

    if dtype is None and xp is np:
        dtype = np.double
    out = xp.stack([th, phi])
    if dtype is not None:
        out = out.astype(dtype)

    if not return_mask:
        return out

    if in_mask is None:
        mask = xp.zeros(x.shape, dtype=bool)
    else:
        mask = xp.broadcast_to(xp.asarray(in_mask, dtype=bool), x.shape)
    return out, xp.stack([mask, mask])


def thphi2xyz(th, phi=None, *, dtype=None):
    """
    Convert spherical angles ``(theta, phi)`` to Cartesian unit vectors.

    Accepts either ``thphi2xyz(th, phi)`` or ``thphi2xyz(th_phi)`` where
    ``th_phi`` is a ``(2, ...)`` array.  Returns a ``(3, ...)`` array of
    ``[x, y, z]``.
    """
    th, phi = _unpack2(th, phi)

    in_mask = None
    if isinstance(th, np.ma.MaskedArray):
        in_mask = np.ma.getmaskarray(th)
        th = np.ma.filled(th, 0.0)
        phi = np.ma.filled(phi, 0.0)

    xp = _xp(th, phi)
    th, phi = xp.asarray(th), xp.asarray(phi)

    z = xp.cos(th)
    r = xp.sin(th)
    x, y = r * xp.cos(phi), r * xp.sin(phi)

    if dtype is None and xp is np:
        dtype = np.double
    out = xp.stack([x, y, z])
    if dtype is not None:
        out = out.astype(dtype)

    if in_mask is not None:
        out = np.ma.array(out, mask=np.stack([in_mask] * 3))
    return out


def angles_to_coord(theta_deg, phi_deg):
    """
    Cartesian unit vector for a polar angle and azimuth given in *degrees*.

    Returns a ``(3, ...)`` array of ``[x, y, z]``.  Convenience wrapper around
    `thphi2xyz` for the degree-valued inputs used by beam measurements.
    """
    xp = _xp(theta_deg, phi_deg)
    return thphi2xyz(xp.deg2rad(theta_deg), xp.deg2rad(phi_deg))


# ---------------------------------------------------------------------------
# Named spherical coordinate systems
# ---------------------------------------------------------------------------

def eq2radec(xyz):
    """
    Equatorial xyz vectors to ``(ra, dec)``.

    ``ra`` runs counter-clockwise about z = north with 0 at the x axis and is
    wrapped onto ``[0, 2*pi)``; ``dec`` is measured from the equator.
    """
    th, phi = xyz2thphi(xyz)
    xp = _xp(th, phi)
    dec = xp.pi / 2 - th
    ra = xp.where(phi < 0, phi + 2 * xp.pi, phi)
    return xp.stack([ra, dec])


def radec2eq(ra_dec):
    """Inverse of `eq2radec`: ``(ra, dec)`` to equatorial xyz vectors."""
    ra, dec = ra_dec
    xp = _xp(ra, dec)
    return thphi2xyz(xp.pi / 2 - xp.asarray(dec), ra)


def latlong2xyz(lat_long):
    """``(lat, long)`` to xyz vectors, with lat measured from the equator."""
    lat, long = lat_long
    return radec2eq((long, lat))


def top2azalt(xyz):
    """
    Topocentric xyz vectors to ``(az, alt)``.

    In the topocentric frame z = up, x = east and y = north, so ``az`` is 0 at
    +y (north) and increases clockwise towards +x (east); it is wrapped onto
    ``[0, 2*pi)``.  ``alt`` is measured from the horizon.
    """
    th, phi = xyz2thphi(xyz)
    xp = _xp(th, phi)
    alt = xp.pi / 2 - th
    az = xp.pi / 2 - phi
    az = xp.where(az < 0, az + 2 * xp.pi, az)
    return xp.stack([az, alt])


def azalt2top(az_alt):
    """
    Inverse of `top2azalt`: ``(az, alt)`` to topocentric xyz vectors, with
    az = 0 at north (+y) and az = pi/2 at east (+x).
    """
    az, alt = az_alt
    xp = _xp(az, alt)
    return thphi2xyz(xp.pi / 2 - xp.asarray(alt), xp.pi / 2 - xp.asarray(az))


# ---------------------------------------------------------------------------
# Equatorial <-> topocentric
# ---------------------------------------------------------------------------

def eq2top_m(ha, dec):
    """
    Matrix converting equatorial coordinates to topocentric.

    Parameters
    ----------
    ha : float or array_like
        Hour angle(s) in radians.
    dec : float or array_like
        Declination(s) in radians.

    Returns
    -------
    m : ndarray, shape (3, 3) or (N, 3, 3)
        Batched over *ha* / *dec* when those are arrays.
    """
    xp = _xp(ha, dec)
    sin_H, cos_H = xp.sin(ha), xp.cos(ha)
    sin_d, cos_d = xp.sin(dec), xp.cos(dec)
    zero = xp.zeros_like(sin_H)
    rows = [[sin_H, cos_H, zero],
            [-sin_d * cos_H, sin_d * sin_H, cos_d],
            [cos_d * cos_H, -cos_d * sin_H, sin_d]]
    m = xp.array(rows) if xp is jnp else np.array(rows, dtype=np.double)
    if m.ndim == 3:
        m = m.transpose([2, 0, 1])
    return m


def top2eq_m(ha, dec):
    """Matrix converting topocentric coordinates to equatorial; inverse of
    `eq2top_m`."""
    m = eq2top_m(ha, dec)
    xp = _xp(m)
    return xp.linalg.inv(m)
