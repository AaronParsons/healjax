"""Tests for healjax.maps."""

import jax

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

import healjax
from healjax.coord import thphi2xyz
from healjax.maps import (
    HPM,
    Alm,
    HealpixBase,
    HealpixMap,
    add2array,
    mk_arr,
)

healpy = pytest.importorskip("healpy")

NSIDE = 16
NPIX = 12 * NSIDE ** 2

# healjax's pixel kernels run in float32, so a small fraction of coordinates
# near a pixel boundary can land in a neighbouring pixel relative to healpy.
PIXEL_AGREEMENT = 0.99


def smooth_map(nside, seed=0):
    """A band-limited map, so interpolation is well behaved."""
    th, phi = healpy.pix2ang(nside, np.arange(12 * nside ** 2))
    rng = np.random.default_rng(seed)
    a, b, c = rng.normal(size=3)
    return (1.0 + a * np.cos(th) + b * np.sin(th) * np.cos(phi)
            + c * np.sin(th) ** 2 * np.sin(2 * phi))


def random_dirs(n, seed=0):
    rng = np.random.default_rng(seed)
    th = np.arccos(rng.uniform(-1, 1, n))
    phi = rng.uniform(0, 2 * np.pi, n)
    return th, phi


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class TestHelpers:

    def test_mk_arr_flattens_scalars_and_lists(self):
        assert mk_arr(3).shape == (1,)
        np.testing.assert_array_equal(mk_arr([1, 2, 3]), [1.0, 2.0, 3.0])
        assert mk_arr([1, 2, 3]).dtype == np.double

    def test_mk_arr_passes_arrays_through_with_cast(self):
        a = np.arange(6, dtype=np.int32).reshape(2, 3)
        out = mk_arr(a, dtype=np.double)
        assert out.shape == (2, 3)  # ndarrays keep their shape
        assert out.dtype == np.double

    def test_add2array_accumulates_repeats(self):
        """This is the whole reason add2array exists: repeats must add up."""
        a = np.zeros(5)
        ind = np.array([[1], [1], [3]])
        add2array(a, ind, np.array([2.0, 3.0, 1.0]))
        np.testing.assert_array_equal(a, [0, 5, 0, 1, 0])

    def test_add2array_multidimensional(self):
        a = np.zeros((3, 3))
        ind = np.array([[0, 1], [0, 1], [2, 2]])
        add2array(a, ind, np.array([1.0, 1.0, 4.0]))
        assert a[0, 1] == 2.0 and a[2, 2] == 4.0
        assert a.sum() == 6.0


# ---------------------------------------------------------------------------
# HealpixBase
# ---------------------------------------------------------------------------

class TestHealpixBase:

    def test_defaults(self):
        h = HealpixBase()
        assert h.nside() == 1 and h.npix() == 12 and h.scheme() == "RING"

    def test_npix_and_order(self):
        h = HealpixBase(nside=32)
        assert h.npix() == healpy.nside2npix(32)
        assert h.order() == healpy.nside2order(32)

    def test_npix2nside_matches_healpy(self):
        h = HealpixBase()
        for nside in (1, 2, 4, 64, 512):
            npix = healpy.nside2npix(nside)
            assert h.npix2nside(npix) == healpy.npix2nside(npix)

    def test_npix2nside_rejects_bad_counts(self):
        h = HealpixBase()
        for bad in (0, 5, 13, -12):
            with pytest.raises(ValueError):
                h.npix2nside(bad)

    def test_set_nside_scheme_requires_power_of_two(self):
        h = HealpixBase()
        with pytest.raises(AssertionError):
            h.set_nside_scheme(nside=3)
        with pytest.raises(AssertionError):
            h.set_nside_scheme(scheme="SPIRAL")

    def test_resolution_arithmetic_needs_no_optional_deps(self, monkeypatch):
        """nside/npix/order must work even without healpy installed."""
        from healjax.maps import _optional
        monkeypatch.setattr(
            _optional, "healpy",
            lambda: (_ for _ in ()).throw(ImportError("no healpy")),
        )
        h = HealpixBase(nside=8)
        assert h.npix() == 768 and h.order() == 3
        assert h.npix2nside(768) == 8

    def test_crd2px_round_trip(self):
        h = HealpixBase(nside=NSIDE)
        px = np.arange(NPIX)
        th, phi = h.px2crd(px, ncrd=2)
        np.testing.assert_array_equal(h.crd2px(th, phi), px)

    def test_crd2px_xyz_matches_thphi(self):
        h = HealpixBase(nside=NSIDE)
        th, phi = random_dirs(200, seed=1)
        x, y, z = thphi2xyz(th, phi)
        np.testing.assert_array_equal(h.crd2px(th, phi), h.crd2px(x, y, z))

    def test_crd2px_interpolate_shapes_and_weights(self):
        h = HealpixBase(nside=NSIDE)
        th, phi = random_dirs(50, seed=2)
        px, wgts = h.crd2px(th, phi, interpolate=True)
        assert px.shape == (50, 4) and wgts.shape == (50, 4)
        np.testing.assert_allclose(wgts.sum(axis=1), 1.0, atol=1e-12)

    def test_nest_ring_conv(self):
        h = HealpixBase(nside=NSIDE, scheme="RING")
        px = np.arange(NPIX)
        nest = h.nest_ring_conv(px, "NEST")
        assert h.scheme() == "NEST"
        np.testing.assert_array_equal(nest, healpy.ring2nest(NSIDE, px))
        back = h.nest_ring_conv(nest, "RING")
        np.testing.assert_array_equal(back, px)

    def test_nest_scheme_pixels_differ(self):
        th, phi = random_dirs(100, seed=3)
        ring = HealpixBase(nside=NSIDE, scheme="RING").crd2px(th, phi)
        nest = HealpixBase(nside=NSIDE, scheme="NEST").crd2px(th, phi)
        np.testing.assert_array_equal(healpy.ring2nest(NSIDE, ring), nest)


# ---------------------------------------------------------------------------
# HealpixMap
# ---------------------------------------------------------------------------

class TestHealpixMap:

    def test_init_zeroed(self):
        m = HealpixMap(nside=NSIDE)
        assert m.map.shape == (NPIX,)
        assert not m.map.any()

    def test_set_map_infers_nside(self):
        m = HealpixMap(nside=1)
        m.set_map(np.arange(NPIX, dtype=float))
        assert m.nside() == NSIDE and m.npix() == NPIX

    def test_set_map_rejects_bad_length(self):
        m = HealpixMap(nside=4)
        with pytest.raises(ValueError):
            m.set_map(np.zeros(100))

    def test_get_map_and_dtype(self):
        m = HealpixMap(nside=4, dtype=np.float32)
        assert m.get_dtype() == np.float32
        assert m.get_map() is m.map

    def test_pixel_indexing(self):
        m = HealpixMap(nside=NSIDE)
        m.set_map(np.arange(NPIX, dtype=float))
        np.testing.assert_array_equal(m[np.array([0, 5, 100])], [0, 5, 100])

    def test_coordinate_indexing(self):
        m = HealpixMap(nside=NSIDE)
        m.set_map(np.arange(NPIX, dtype=float))
        th, phi = healpy.pix2ang(NSIDE, np.array([7, 42, 300]))
        np.testing.assert_array_equal(m[th, phi], [7, 42, 300])

    def test_xyz_indexing(self):
        m = HealpixMap(nside=NSIDE)
        m.set_map(np.arange(NPIX, dtype=float))
        x, y, z = healpy.pix2vec(NSIDE, np.array([7, 42, 300]))
        np.testing.assert_array_equal(m[x, y, z], [7, 42, 300])

    def test_setitem_accumulates_repeated_coordinates(self):
        m = HealpixMap(nside=NSIDE)
        th, phi = healpy.pix2ang(NSIDE, np.array([10, 10, 10, 20]))
        m[th, phi] = np.array([1.0, 2.0, 3.0, 5.0])
        assert m.map[10] == 6.0 and m.map[20] == 5.0
        assert m.map.sum() == 11.0

    def test_setitem_leaves_untouched_pixels_alone(self):
        m = HealpixMap(nside=NSIDE)
        m.set_map(np.full(NPIX, 9.0))
        m[np.array([3, 4])] = np.array([1.0, 2.0])
        assert m.map[3] == 1.0 and m.map[4] == 2.0 and m.map[5] == 9.0

    def test_interpolated_read(self):
        m = HealpixMap(nside=NSIDE, interp=True)
        data = smooth_map(NSIDE, seed=4)
        m.set_map(data)
        th, phi = random_dirs(100, seed=5)
        np.testing.assert_allclose(
            m[th, phi], healpy.get_interp_val(data, th, phi), atol=1e-12
        )

    def test_set_interpol_toggles_behaviour(self):
        data = smooth_map(NSIDE, seed=6)
        m = HealpixMap(nside=NSIDE)
        m.set_map(data)
        th, phi = random_dirs(50, seed=7)
        nearest = m[th, phi].copy()
        m.set_interpol(True)
        assert not np.allclose(nearest, m[th, phi])

    def test_change_scheme_round_trip(self):
        data = np.arange(NPIX, dtype=float)
        m = HealpixMap(nside=NSIDE)
        m.set_map(data)
        m.change_scheme("NEST")
        assert m.scheme() == "NEST"
        np.testing.assert_array_equal(m.map, healpy.reorder(data, r2n=True))
        m.change_scheme("RING")
        np.testing.assert_array_equal(m.map, data)

    def test_from_hpm_same_nside(self):
        src = HealpixMap(nside=NSIDE)
        src.set_map(smooth_map(NSIDE, seed=8))
        dst = HealpixMap(nside=NSIDE)
        dst.from_hpm(src)
        np.testing.assert_allclose(dst.map, src.map)

    def test_from_hpm_upgrade_resolution(self):
        src = HealpixMap(nside=8)
        src.set_map(smooth_map(8, seed=9))
        dst = HealpixMap(nside=16)
        dst.from_hpm(src)
        assert dst.nside() == 16
        assert np.isfinite(dst.map).all() and dst.map.any()

    def test_from_hpm_downgrade_resolution(self):
        src = HealpixMap(nside=16)
        src.set_map(smooth_map(16, seed=10))
        dst = HealpixMap(nside=8)
        dst.from_hpm(src)
        assert dst.nside() == 8 and dst.map.any()


# ---------------------------------------------------------------------------
# Alm
# ---------------------------------------------------------------------------

class TestAlm:

    def test_size_and_zero_init(self):
        a = Alm(8, 8)
        assert a.size() == healpy.Alm.getsize(8, 8)
        assert not a.get_data().any()

    def test_getitem_setitem(self):
        a = Alm(4, 4)
        a[2, 1] = 3 + 4j
        assert a[2, 1] == 3 + 4j
        assert a[2, 0] == 0

    def test_lm_indices(self):
        a = Alm(3, 3)
        lm = a.lm_indices()
        assert lm.shape == (a.size(), 2)
        assert (lm[:, 0] >= lm[:, 1]).all()

    def test_rejects_mmax_above_lmax(self):
        with pytest.raises(AssertionError):
            Alm(2, 5)

    def test_set_data_length_checked(self):
        a = Alm(4, 4)
        with pytest.raises(AssertionError):
            a.set_data(np.zeros(3, dtype=np.complex128))

    def test_monopole_round_trip(self):
        """A pure monopole synthesises to a constant map."""
        a = Alm(4, 4)
        a[0, 0] = 1.0
        m = a.to_map(NSIDE, pol=False)
        np.testing.assert_allclose(m, m[0], rtol=1e-10)
        assert not np.isclose(m[0], 0.0)

    def test_from_map_then_to_map(self):
        data = smooth_map(NSIDE, seed=11)
        a = Alm(8, 8)
        a.from_map(data, iter=3, pol=False)
        recon = a.to_map(NSIDE, pol=False)
        np.testing.assert_allclose(recon, data, atol=1e-3)


class TestHealpixMapAlmInterop:

    def test_from_alm_gives_ring_map(self):
        a = Alm(4, 4)
        a[0, 0] = 2.0
        m = HealpixMap(nside=NSIDE)
        m.from_alm(a)
        assert m.scheme() == "RING"
        np.testing.assert_allclose(m.map, m.map[0], rtol=1e-10)

    def test_from_alm_does_not_apply_a_pixel_window(self):
        """Regression: the scheme string used to be passed as `pixwin`."""
        a = Alm(4, 4)
        a[0, 0] = 1.0
        m = HealpixMap(nside=NSIDE)
        m.from_alm(a)
        np.testing.assert_allclose(m.map, a.to_map(NSIDE, pixwin=False,
                                                   pol=False))

    def test_to_alm_round_trip(self):
        data = smooth_map(NSIDE, seed=12)
        m = HealpixMap(nside=NSIDE)
        m.set_map(data)
        alm = m.to_alm(8, 8, iter=3)
        assert alm.size() == healpy.Alm.getsize(8, 8)
        np.testing.assert_allclose(alm.to_map(NSIDE, pol=False), data,
                                   atol=1e-3)

    def test_to_alm_requires_ring(self):
        m = HealpixMap(nside=NSIDE)
        m.change_scheme("NEST")
        with pytest.raises(AssertionError):
            m.to_alm(4, 4)


# ---------------------------------------------------------------------------
# FITS I/O
# ---------------------------------------------------------------------------

class TestFitsIO:

    def test_round_trip(self, tmp_path):
        pytest.importorskip("astropy")
        data = smooth_map(NSIDE, seed=13)
        m = HealpixMap(nside=NSIDE)
        m.set_map(data)
        path = str(tmp_path / "map.fits")
        m.to_fits(path)

        back = HealpixMap(fromfits=path)
        assert back.nside() == NSIDE and back.scheme() == "RING"
        np.testing.assert_allclose(back.map, data, rtol=1e-6)

    def test_header_is_healpix_conformant(self, tmp_path):
        pyfits = pytest.importorskip("astropy.io.fits")
        m = HealpixMap(nside=NSIDE)
        m.change_scheme("NEST")
        path = str(tmp_path / "nest.fits")
        m.to_fits(path)
        hdr = pyfits.open(path)[1].header
        assert hdr["PIXTYPE"] == "HEALPIX"
        assert hdr["ORDERING"] == "NESTED"
        assert hdr["NSIDE"] == NSIDE
        assert hdr["LASTPIX"] == NPIX - 1

    def test_nest_scheme_survives_round_trip(self, tmp_path):
        pytest.importorskip("astropy")
        data = np.arange(NPIX, dtype=float)
        m = HealpixMap(nside=NSIDE)
        m.set_map(data, scheme="NEST")
        path = str(tmp_path / "n.fits")
        m.to_fits(path)
        back = HealpixMap(fromfits=path)
        assert back.scheme() == "NEST"

    def test_healpy_can_read_our_output(self, tmp_path):
        data = smooth_map(NSIDE, seed=14)
        m = HealpixMap(nside=NSIDE)
        m.set_map(data)
        path = str(tmp_path / "hp.fits")
        m.to_fits(path)
        np.testing.assert_allclose(healpy.read_map(path, verbose=False)
                                   if "verbose" in
                                   healpy.read_map.__code__.co_varnames
                                   else healpy.read_map(path),
                                   data, rtol=1e-6)

    def test_missing_astropy_raises_helpful_error(self, tmp_path, monkeypatch):
        from healjax.maps import _optional
        monkeypatch.setattr(
            _optional, "fits",
            lambda: (_ for _ in ()).throw(
                ImportError("FITS I/O requires the optional dependency")),
        )
        m = HealpixMap(nside=4)
        with pytest.raises(ImportError, match="optional dependency"):
            m.to_fits(str(tmp_path / "x.fits"))


# ---------------------------------------------------------------------------
# HPM
# ---------------------------------------------------------------------------

class TestHPM:

    def test_is_a_healpix_map(self):
        h = HPM(nside=NSIDE)
        assert isinstance(h, HealpixMap) and isinstance(h, HealpixBase)
        assert h.nside() == NSIDE and h.map.shape == (NPIX,)

    def test_pixel_indexing_matches_healpix_map(self):
        data = smooth_map(NSIDE, seed=15)
        h, m = HPM(nside=NSIDE), HealpixMap(nside=NSIDE)
        h.set_map(data)
        m.set_map(data)
        px = np.array([0, 17, 512])
        np.testing.assert_array_equal(np.asarray(h[px]), m[px])

    def test_crd2px_mostly_matches_healpy(self):
        """float32 kernels allow rare boundary disagreements; most must match."""
        h = HPM(nside=NSIDE)
        th, phi = random_dirs(2000, seed=16)
        ours = np.asarray(h.crd2px(th, phi))
        theirs = healpy.ang2pix(NSIDE, th, phi)
        assert (ours == theirs).mean() > PIXEL_AGREEMENT

    def test_crd2px_xyz_mostly_matches_healpy(self):
        h = HPM(nside=NSIDE)
        th, phi = random_dirs(2000, seed=17)
        x, y, z = thphi2xyz(th, phi)
        ours = np.asarray(h.crd2px(x, y, z))
        theirs = healpy.vec2pix(NSIDE, x, y, z)
        assert (ours == theirs).mean() > PIXEL_AGREEMENT

    def test_crd2px_preserves_input_shape(self):
        h = HPM(nside=NSIDE)
        th, phi = random_dirs(12, seed=18)
        px = h.crd2px(th.reshape(3, 4), phi.reshape(3, 4))
        assert px.shape == (3, 4)

    def test_crd2px_interpolate_shapes(self):
        h = HPM(nside=NSIDE)
        th, phi = random_dirs(40, seed=19)
        px, wgts = h.crd2px(th, phi, interpolate=True)
        assert px.shape == (40, 4) and wgts.shape == (40, 4)
        np.testing.assert_allclose(np.asarray(wgts).sum(axis=1), 1.0,
                                   rtol=1e-5)

    def test_interpolation_rejects_nest(self):
        h = HPM(nside=NSIDE)
        h.change_scheme("NEST")
        th, phi = random_dirs(5, seed=20)
        with pytest.raises(NotImplementedError):
            h.crd2px(th, phi, interpolate=True)

    def test_interpolated_read_matches_healpy(self):
        data = smooth_map(NSIDE, seed=21)
        h = HPM(nside=NSIDE, interp=True)
        h.set_map(data)
        th, phi = random_dirs(200, seed=22)
        np.testing.assert_allclose(
            np.asarray(h[th, phi]), healpy.get_interp_val(data, th, phi),
            rtol=1e-5, atol=1e-6,
        )

    def test_interpolated_read_xyz(self):
        data = smooth_map(NSIDE, seed=23)
        h = HPM(nside=NSIDE, interp=True)
        h.set_map(data)
        th, phi = random_dirs(100, seed=24)
        x, y, z = thphi2xyz(th, phi)
        np.testing.assert_allclose(np.asarray(h[x, y, z]),
                                   np.asarray(h[th, phi]),
                                   rtol=1e-5, atol=1e-6)

    def test_jax_kernels_track_resolution_changes(self):
        """set_map at a new nside must rebuild the cached jitted kernels."""
        h = HPM(nside=8)
        assert h.nside() == 8
        h.set_map(np.arange(12 * 32 ** 2, dtype=float))
        assert h.nside() == 32
        th, phi = healpy.pix2ang(32, np.array([5, 900, 7000]))
        np.testing.assert_array_equal(np.asarray(h.crd2px(th, phi)),
                                      [5, 900, 7000])

    def test_jax_kernels_track_scheme_changes(self):
        h = HPM(nside=NSIDE)
        h.set_map(np.arange(NPIX, dtype=float))
        h.change_scheme("NEST")
        assert h.scheme() == "NEST"
        th, phi = healpy.pix2ang(NSIDE, np.array([11, 220]), nest=True)
        np.testing.assert_array_equal(np.asarray(h.crd2px(th, phi)),
                                      [11, 220])

    def test_rotate_interpolate_and_sum(self):
        from healjax.coord import rot_m

        beam = np.abs(smooth_map(NSIDE, seed=25)) + 0.5
        nfreq = 3
        h = HPM(nside=NSIDE)
        h.set_map(np.stack([beam] * nfreq, axis=-1))
        th, phi = healpy.pix2ang(NSIDE, np.arange(NPIX))
        crds = np.asarray(thphi2xyz(th, phi))
        sky = np.repeat(smooth_map(NSIDE, seed=26)[:, None], nfreq, axis=-1)
        rots = np.stack([rot_m(a, np.array([0.0, 0.0, 1.0]))
                         for a in np.linspace(0, 1, 20)])

        out = h.rotate_interpolate_and_sum(sky, crds, rots, chunk_size=7)
        assert out.shape == (20, nfreq)
        assert np.all(np.isfinite(out))

    def test_chunking_does_not_change_the_result(self):
        from healjax.coord import rot_m

        beam = np.abs(smooth_map(NSIDE, seed=27)) + 0.5
        h = HPM(nside=NSIDE)
        h.set_map(beam[:, None])
        th, phi = healpy.pix2ang(NSIDE, np.arange(NPIX))
        crds = np.asarray(thphi2xyz(th, phi))
        sky = smooth_map(NSIDE, seed=28)[:, None]
        rots = np.stack([rot_m(a, np.array([0.0, 1.0, 0.0]))
                         for a in np.linspace(0, 0.5, 9)])
        a = h.rotate_interpolate_and_sum(sky, crds, rots, chunk_size=2)
        b = h.rotate_interpolate_and_sum(sky, crds, rots, chunk_size=100)
        np.testing.assert_allclose(a, b, rtol=1e-6)


class TestAipyParity:
    """
    `HPM` is meant to be a drop-in replacement for
    ``aipy.healpix.HealpixMap``; these pin the observable behaviour.
    """

    @pytest.fixture(autouse=True)
    def _aipy(self):
        self.ahp = pytest.importorskip("aipy").healpix

    def pair(self, data, cls=HealpixMap):
        a = self.ahp.HealpixMap(nside=NSIDE)
        a.set_map(data.copy())
        o = cls(nside=NSIDE)
        o.set_map(data.copy())
        return a, o

    def test_nearest_reads(self):
        data = smooth_map(NSIDE, seed=40)
        a, o = self.pair(data)
        th, phi = random_dirs(500, seed=41)
        np.testing.assert_array_equal(o[th, phi], a[th, phi])
        x, y, z = thphi2xyz(th, phi)
        np.testing.assert_array_equal(o[x, y, z], a[x, y, z])

    def test_interpolated_reads(self):
        data = smooth_map(NSIDE, seed=42)
        a, o = self.pair(data)
        a.set_interpol(True)
        o.set_interpol(True)
        th, phi = random_dirs(300, seed=43)
        np.testing.assert_allclose(o[th, phi], a[th, phi], atol=1e-12)

    def test_setitem_accumulation(self):
        a, o = self.pair(np.zeros(NPIX))
        th, phi = healpy.pix2ang(NSIDE, np.array([3, 3, 3, 9, 9, 100]))
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        a[th, phi] = v
        o[th, phi] = v
        np.testing.assert_array_equal(o.map, a.map)

    def test_change_scheme(self):
        a, o = self.pair(smooth_map(NSIDE, seed=44))
        a.change_scheme("NEST")
        o.change_scheme("NEST")
        np.testing.assert_array_equal(o.map, a.map)

    @pytest.mark.parametrize("src_n,dst_n", [(8, 16), (16, 8), (16, 16)])
    def test_from_hpm(self, src_n, dst_n):
        data = smooth_map(src_n, seed=45)
        sa = self.ahp.HealpixMap(nside=src_n)
        sa.set_map(data.copy())
        da = self.ahp.HealpixMap(nside=dst_n)
        da.from_hpm(sa)

        so = HealpixMap(nside=src_n)
        so.set_map(data.copy())
        do = HealpixMap(nside=dst_n)
        do.from_hpm(so)
        np.testing.assert_allclose(do.map, da.map, atol=1e-12)

    def test_hpm_nearest_reads_match_aipy(self):
        data = smooth_map(NSIDE, seed=46)
        a, h = self.pair(data, cls=HPM)
        th, phi = random_dirs(500, seed=47)
        agree = (np.asarray(h[th, phi]) == a[th, phi]).mean()
        assert agree > PIXEL_AGREEMENT


class TestTopLevelReexports:

    def test_convenience_symbols(self):
        assert healjax.HPM is HPM
        assert healjax.HealpixMap is HealpixMap
        assert healjax.HealpixBase is HealpixBase
        assert healjax.Alm is Alm

    def test_existing_pixel_api_still_works(self):
        """Backward compatibility: the original top-level functions survive."""
        assert healjax.nside2npix(NSIDE) == NPIX
        px = jax.jit(lambda t, p: healjax.ang2pix("ring", NSIDE, t, p))(0.5,
                                                                        1.0)
        assert int(px) == healpy.ang2pix(NSIDE, 0.5, 1.0)


# ---------------------------------------------------------------------------
# Spherical-harmonic fitting
# ---------------------------------------------------------------------------

class TestSphFit:

    @pytest.fixture(autouse=True)
    def _scipy(self):
        pytest.importorskip("scipy")

    def sparse_map(self, nside=8, seed=30, frac=0.4):
        """A positive smooth map with most pixels marked UNSEEN."""
        full = np.abs(smooth_map(nside, seed=seed)) + 0.5
        rng = np.random.default_rng(seed)
        m = np.full_like(full, healpy.UNSEEN)
        keep = rng.random(full.size) < frac
        m[keep] = full[keep]
        return m, full, keep

    def test_fit_alms_shape(self):
        from healjax.maps import fit_alms_from_maps

        m, _, _ = self.sparse_map()
        alms = fit_alms_from_maps(m, nside=8, lmax=4, lam=1e-3)
        assert alms.shape == (1, healpy.Alm.getsize(4))
        assert alms.dtype == np.complex128

    def test_fit_alms_multiple_maps(self):
        from healjax.maps import fit_alms_from_maps

        m1, _, _ = self.sparse_map(seed=31)
        m2, _, _ = self.sparse_map(seed=32)
        alms = fit_alms_from_maps(np.stack([m1, m2]), nside=8, lmax=3,
                                  lam=1e-3)
        assert alms.shape == (2, healpy.Alm.getsize(3))
        assert not np.allclose(alms[0], alms[1])

    def test_fit_alms_rejects_empty_map(self):
        from healjax.maps import fit_alms_from_maps

        m = np.full(healpy.nside2npix(8), healpy.UNSEEN)
        with pytest.raises(ValueError, match="no observed pixels"):
            fit_alms_from_maps(m, nside=8, lmax=2, lam=1e-3)

    def test_alms_to_filled_maps_shape(self):
        from healjax.maps import alms_to_filled_maps, fit_alms_from_maps

        m, _, _ = self.sparse_map()
        alms = fit_alms_from_maps(m, nside=8, lmax=4, lam=1e-3)
        out = alms_to_filled_maps(alms, nside=8, lmax=4)
        assert out.shape == (1, healpy.nside2npix(8))
        assert np.isfinite(out).all()

    def test_clamp_known_restores_observed_pixels(self):
        from healjax.maps import alms_to_filled_maps, fit_alms_from_maps

        m, _, keep = self.sparse_map()
        alms = fit_alms_from_maps(m, nside=8, lmax=4, lam=1e-3)
        out = alms_to_filled_maps(alms, nside=8, lmax=4, clamp_known=True,
                                  original_maps=m)
        np.testing.assert_allclose(out[0][keep], m[keep])

    def test_sph_fit_fills_every_pixel_positively(self):
        from healjax.maps import sph_fit

        m, full, keep = self.sparse_map(frac=0.6)
        out = sph_fit(m, nside=8, lmax=6, lam=1e-4)
        assert out.shape == (1, healpy.nside2npix(8))
        assert np.isfinite(out).all()
        assert (out > 0).all()  # log-domain fit cannot produce negatives
        assert healpy.UNSEEN not in out

    def test_sph_fit_recovers_a_band_limited_map(self):
        """With enough observed pixels the fit should track the truth."""
        from healjax.maps import sph_fit

        m, full, keep = self.sparse_map(seed=33, frac=0.8)
        out = sph_fit(m, nside=8, lmax=8, lam=1e-6)[0]
        # correlate rather than demand equality: regularisation biases scale
        r = np.corrcoef(out[~keep], full[~keep])[0, 1]
        assert r > 0.9

    def test_x_to_alm_layout(self):
        from healjax.maps import x_to_alm

        lmax = 3
        x = np.zeros((lmax + 1) ** 2)
        x[0] = 1.0  # a_00
        alm = x_to_alm(x, lmax)
        assert alm[healpy.Alm.getidx(lmax, 0, 0)] == 1.0
        assert np.count_nonzero(alm) == 1

    def test_design_matrix_shape(self):
        from healjax.maps import build_real_design_matrix_from_angles

        theta, phi = random_dirs(30, seed=34)
        A = build_real_design_matrix_from_angles(theta, phi, lmax=4)
        assert A.shape == (30, 25)
        assert np.isfinite(A).all()
