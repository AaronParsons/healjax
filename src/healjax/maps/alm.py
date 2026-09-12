"""
Spherical-harmonic coefficients.

`Alm` wraps healpy's alm indexing and transforms.  The module also carries the
regularised spherical-harmonic *fitting* routines used to fill sparse beam
measurements: `fit_alms_from_maps`, `alms_to_filled_maps` and the `sph_fit`
convenience wrapper.
"""

from __future__ import annotations

import numpy as np

from . import _optional

__all__ = [
    "Alm",
    "sph_fit",
    "fit_alms_from_maps",
    "alms_to_filled_maps",
    "build_real_design_matrix_from_angles",
    "x_to_alm",
]


class Alm:
    """Spherical-harmonic coefficients up to a given order."""

    def __init__(self, lmax, mmax, dtype=np.complex128):
        assert lmax >= mmax
        self._alm = _optional.healpy().Alm()
        self._lmax = lmax
        self._mmax = mmax
        self.dtype = dtype
        self.set_to_zero()

    def size(self):
        return self._alm.getsize(self._lmax, self._mmax)

    def set_to_zero(self):
        self.set_data(np.zeros(self.size(), dtype=self.dtype))

    def lmax(self):
        return self._lmax

    def mmax(self):
        return self._mmax

    def __getitem__(self, lm):
        l, m = lm
        return self.data[self._alm.getidx(self._lmax, l, m)]

    def __setitem__(self, lm, val):
        l, m = lm
        self.data[self._alm.getidx(self._lmax, l, m)] = val

    def to_map(self, nside, pixwin=False, fwhm=0.0, sigma=None, pol=True):
        """Synthesise a RING-ordered map at the given *nside*."""
        return _optional.healpy().alm2map(
            self.get_data(), nside, lmax=self._lmax, mmax=self._mmax,
            pixwin=pixwin, fwhm=fwhm, sigma=sigma, pol=pol,
        )

    def from_map(self, data, iter=3, pol=True, use_weights=False, gal_cut=0):
        """Analyse a RING-ordered map into this object's coefficients."""
        data = _optional.healpy().map2alm(
            data, lmax=self._lmax, mmax=self._mmax, iter=iter, pol=pol,
            use_weights=use_weights, gal_cut=gal_cut,
        )
        self.set_data(data)

    def lm_indices(self):
        """``(size, 2)`` array of the ``(l, m)`` owning each coefficient."""
        return np.array(
            [self._alm.getlm(self._lmax, i) for i in range(self.size())]
        )

    def get_data(self):
        return self.data

    def set_data(self, data):
        assert data.size == self.size()
        self.data = data.astype(self.dtype)


# ---------------------------------------------------------------------------
# Regularised spherical-harmonic fitting
# ---------------------------------------------------------------------------

def build_real_design_matrix_from_angles(theta, phi, lmax):
    """
    Design matrix ``A`` of shape ``(n_obs, (lmax+1)**2)`` mapping a real
    coefficient vector to real map values at ``(theta, phi)``.

    Coefficient layout: ``a_l0``, then for ``m=1..l`` the pair
    ``Re(a_lm), Im(a_lm)``.
    """
    sph_harm_y = _optional.scipy_special().sph_harm_y
    npar = (lmax + 1) ** 2
    A = np.empty((theta.size, npar), dtype=np.float64)
    k = 0
    for l in range(lmax + 1):
        Y = sph_harm_y(l, 0, theta, phi)
        A[:, k] = Y.real
        k += 1
        for m in range(1, l + 1):
            Y = sph_harm_y(l, m, theta, phi)
            A[:, k] = 2.0 * Y.real
            A[:, k + 1] = -2.0 * Y.imag
            k += 2
    return A


def x_to_alm(x, lmax):
    """Convert a real coefficient vector to a complex healpy alm array."""
    hp = _optional.healpy()
    alm = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
    k = 0
    for l in range(lmax + 1):
        alm[hp.Alm.getidx(lmax, l, 0)] = x[k]
        k += 1
        for m in range(1, l + 1):
            alm[hp.Alm.getidx(lmax, l, m)] = x[k] + 1j * x[k + 1]
            k += 2
    return alm


def _make_penalty_diag(lmax, p=2):
    """
    Penalty weights for Laplacian regularisation: ``(l*(l+1))**p`` for every
    coefficient, zero at ``l=0``.
    """
    npar = (lmax + 1) ** 2
    pen = np.zeros(npar, dtype=np.float64)
    k = 0
    for l in range(lmax + 1):
        w = (l * (l + 1)) ** p
        pen[k] = w
        k += 1
        for m in range(1, l + 1):
            pen[k] = w
            pen[k + 1] = w
            k += 2
    return pen


def _solve_regularized_ls(A, y, lam, pen, w=None):
    """
    Solve ``argmin ||W^(1/2)(Ax - y)||^2 + lam * ||diag(pen)^(1/2) x||^2`` by
    row augmentation.

    Parameters
    ----------
    A : (n_obs, npar) ndarray
    y : (n_obs,) ndarray
    lam : float
    pen : (npar,) ndarray
        Penalty weights from `_make_penalty_diag`.
    w : (n_obs,) ndarray, optional
        Per-observation weights.
    """
    lstsq = _optional.scipy_linalg().lstsq
    npar = A.shape[1]
    if w is not None:
        w = np.asarray(w, dtype=np.float64)
        sw = np.sqrt(np.clip(w, 0.0, np.inf))
        A = A * sw[:, None]
        y = y * sw

    sqrt_lam_pen = np.sqrt(lam * pen)
    cols = np.where(sqrt_lam_pen > 0)[0]
    if cols.size > 0:
        R = np.zeros((cols.size, npar), dtype=np.float64)
        R[np.arange(cols.size), cols] = sqrt_lam_pen[cols]
        A = np.vstack([A, R])
        y = np.concatenate([y, np.zeros(cols.size, dtype=np.float64)])

    x, *_ = lstsq(A, y)
    return x


def fit_alms_from_maps(
    maps,
    nside,
    lmax=5,
    lam=1e-2,
    p=2,
    fit_log=False,
    eps=1e-12,
    user_weights=None,
    peak_weight_alpha=0.0,
    peak_weight_gamma=2.0,
    nest=False,
):
    """
    Fit spherical-harmonic coefficients to sparse HEALPix maps.

    Parameters
    ----------
    maps : (n_maps, npix) or (npix,) array
        Input maps.  Pixels equal to ``healpy.UNSEEN`` or non-finite are
        treated as unobserved and ignored.
    nside : int
        HEALPix resolution of *maps*.
    lmax : int
        Maximum spherical-harmonic degree.
    lam : float
        Regularisation strength for the Laplacian penalty.
    p : int
        Regularisation order; the penalty is ``(l*(l+1))**p``.
    fit_log : bool
        Fit ``log(values)`` instead of values.  Requires positive data; *eps*
        floors the input to avoid ``log(0)``.
    eps : float
        Floor applied before taking the log when ``fit_log=True``.
    user_weights : (npix,) array, optional
        Per-pixel weights applied to observed pixels.
    peak_weight_alpha : float
        Strength of extra upweighting for bright pixels.
    peak_weight_gamma : float
        Power applied to the normalised pixel value when upweighting peaks.
    nest : bool
        Ordering of *maps* (``False`` = RING).

    Returns
    -------
    alms : ndarray, shape (n_maps, healpy.Alm.getsize(lmax)), complex128
    """
    hp = _optional.healpy()
    maps = np.asarray(maps)
    if maps.ndim == 1:
        maps = maps[None, :]
    n_maps, npix = maps.shape
    assert npix == hp.nside2npix(nside)

    pen = _make_penalty_diag(lmax, p=p)
    A_cache = {}

    alms = np.empty((n_maps, hp.Alm.getsize(lmax)), dtype=np.complex128)

    for i in range(n_maps):
        m = maps[i]
        known = (m != hp.UNSEEN) & np.isfinite(m)
        ipix = np.where(known)[0]
        if ipix.size == 0:
            raise ValueError(f"Map {i} has no observed pixels.")

        y_lin = m[ipix].astype(np.float64)
        y = np.log(np.maximum(y_lin, eps)) if fit_log else y_lin

        w = None
        if user_weights is not None:
            w = np.asarray(user_weights, dtype=np.float64)[ipix].copy()

        if peak_weight_alpha > 0:
            f = np.maximum(y_lin, 0.0)
            fmax = f.max() if f.size else 1.0
            extra = 1.0 + peak_weight_alpha * (
                f / max(fmax, 1e-30)
            ) ** peak_weight_gamma
            w = extra if w is None else (w * extra)

        key = ipix.tobytes()
        if key not in A_cache:
            theta, phi = hp.pix2ang(nside, ipix, nest=nest)
            A_cache[key] = build_real_design_matrix_from_angles(
                theta, phi, lmax
            )

        x = _solve_regularized_ls(A_cache[key], y, lam=lam, pen=pen, w=w)
        alms[i] = x_to_alm(x, lmax)

    return alms


def alms_to_filled_maps(
    alms,
    nside,
    lmax,
    clamp_known=False,
    original_maps=None,
    fit_log=False,
    log_offset=None,
):
    """
    Synthesise full HEALPix maps from alm coefficients.

    Parameters
    ----------
    alms : (n_maps, alm_size) or (alm_size,) array
    nside : int
    lmax : int
    clamp_known : bool
        With *original_maps* supplied, restore observed pixels to their
        original values after synthesis.
    original_maps : array_like, optional
        The sparse maps used to identify observed pixels for clamping.
    fit_log : bool
        Exponentiate the synthesised map, undoing a log-domain fit.
    log_offset : (n_maps,) array, optional
        Additive offsets applied in the log domain before exponentiating.

    Returns
    -------
    out : ndarray, shape (n_maps, npix)
    """
    hp = _optional.healpy()
    alms = np.asarray(alms)
    if alms.ndim == 1:
        alms = alms[None, :]
    n_maps = alms.shape[0]

    out = np.empty((n_maps, hp.nside2npix(nside)), dtype=np.float64)
    for i in range(n_maps):
        pred = hp.alm2map(
            alms[i].astype(np.complex128), nside=nside, lmax=lmax
        )
        if fit_log:
            if log_offset is not None:
                pred = pred + float(log_offset[i])
            pred = np.exp(pred)
        if clamp_known and original_maps is not None:
            m0 = np.asarray(
                original_maps[i]
                if np.asarray(original_maps).ndim == 2
                else original_maps
            )
            known = (m0 != hp.UNSEEN) & np.isfinite(m0)
            pred[known] = m0[known]
        out[i] = pred

    return out


def sph_fit(maps, nside, lmax, lam, peak_weight_alpha=0.0,
            peak_weight_gamma=2.0):
    """
    Fit alms to sparse maps and synthesise the filled maps in one step.

    A log-domain fit with second-order Laplacian regularisation, which is the
    combination that works for EIGSEP beam measurements.

    Parameters
    ----------
    maps : (n_maps, npix) or (npix,) array
    nside : int
    lmax : int
    lam : float
        Regularisation strength.
    peak_weight_alpha, peak_weight_gamma : float
        Optional peak-upweighting parameters passed to `fit_alms_from_maps`.

    Returns
    -------
    filled_maps : ndarray, shape (n_maps, npix)
    """
    alms = fit_alms_from_maps(
        maps=maps,
        nside=nside,
        lmax=lmax,
        lam=lam,
        p=2,
        fit_log=True,
        eps=1e-12,
        peak_weight_alpha=peak_weight_alpha,
        peak_weight_gamma=peak_weight_gamma,
    )
    return alms_to_filled_maps(
        alms,
        nside=nside,
        lmax=lmax,
        fit_log=True,
        clamp_known=False,
        original_maps=maps,
    )
