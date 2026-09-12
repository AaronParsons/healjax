"""
Lazy accessors for the optional third-party dependencies used by ``maps``.

The core of healjax needs only numpy and jax.  The map classes reach for
healpy (pixel-scheme conversions and spherical-harmonic transforms), astropy
(FITS I/O) and scipy (spherical-harmonic fitting) -- all of which are declared
as the ``maps`` extra rather than hard requirements.  Importing this module
never fails; the error surfaces only when a feature that needs the package is
actually called.
"""

from __future__ import annotations

import importlib

_EXTRA_HINT = "pip install 'healjax[maps]'"


def _require(module, feature):
    try:
        return importlib.import_module(module)
    except ImportError as err:  # pragma: no cover - depends on environment
        raise ImportError(
            f"{feature} requires the optional dependency '{module}'. "
            f"Install it with: {_EXTRA_HINT}"
        ) from err


def healpy():
    """Return the ``healpy`` module, or raise a helpful ImportError."""
    return _require("healpy", "This HEALPix operation")


def fits():
    """Return ``astropy.io.fits``, or raise a helpful ImportError."""
    return _require("astropy.io.fits", "FITS I/O")


def scipy_linalg():
    """Return ``scipy.linalg``, or raise a helpful ImportError."""
    return _require("scipy.linalg", "Spherical-harmonic fitting")


def scipy_special():
    """Return ``scipy.special``, or raise a helpful ImportError."""
    return _require("scipy.special", "Spherical-harmonic fitting")


def have(module):
    """True if *module* can be imported; for tests and feature probes."""
    try:
        importlib.import_module(module)
    except ImportError:
        return False
    return True
