"""
Copyright (c) 2023 ghcollin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import unittest
import jax
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import healjax
import astropy_healpix
import astropy_healpix.healpy
import healpy
import numpy
from functools import partial

n_test = 1000000
test_nsides = [  2**(i+1) for i in range(10) ]
edge_eps = 1e-15

class TestIndexingMethods(unittest.TestCase):

    def test_north_pole(self):
        phis = numpy.array(jnp.linspace(edge_eps, 2*jnp.pi-edge_eps, 10))
        norths = numpy.array(jnp.full_like(phis, 0) + edge_eps)

        for nside in test_nsides:
            true_north_ring = astropy_healpix.healpy.ang2pix(nside, norths, phis, nest=False)
            test_north_ring = jax.jit(jax.vmap(partial(healjax.ang2pix, 'ring', nside)))(norths, phis)
            self.assertTrue((true_north_ring == test_north_ring).all(), (true_north_ring, test_north_ring))

        for nside in test_nsides:
            true_north_nest = astropy_healpix.healpy.ang2pix(nside, norths, phis, nest=True)
            test_north_nest = jax.jit(jax.vmap(partial(healjax.ang2pix, 'nest', nside)))(norths, phis)
            self.assertTrue((true_north_nest == test_north_nest).all(), (true_north_nest, test_north_nest))

    def test_south_pole(self):
        phis = numpy.array(jnp.linspace(edge_eps, 2*jnp.pi-edge_eps, 10))
        souths = numpy.array(jnp.full_like(phis, jnp.pi) - edge_eps)

        for nside in test_nsides:
            true_south_ring = astropy_healpix.healpy.ang2pix(nside, souths, phis, nest=False)
            test_south_ring = jax.jit(jax.vmap(partial(healjax.ang2pix, 'ring', nside)))(souths, phis)
            self.assertTrue((true_south_ring == test_south_ring).all(), (true_south_ring, test_south_ring))

        for nside in test_nsides:
            true_south_nest = astropy_healpix.healpy.ang2pix(nside, souths, phis, nest=True)
            test_south_nest = jax.jit(jax.vmap(partial(healjax.ang2pix, 'nest', nside)))(souths, phis)
            self.assertTrue((true_south_nest == test_south_nest).all(), (true_south_nest, test_south_nest))

    def test_seam(self):
        thetas_half = jnp.linspace(0+edge_eps, jnp.pi-edge_eps, 10)
        phis = numpy.array(jnp.concatenate([jnp.full_like(thetas_half, edge_eps), jnp.full_like(thetas_half, 2*jnp.pi - edge_eps)]))
        thetas = numpy.array(jnp.concatenate([thetas_half, thetas_half]))

        for nside in test_nsides:
            true_ring = astropy_healpix.healpy.ang2pix(nside, thetas, phis, nest=False)
            test_ring = jax.jit(jax.vmap(partial(healjax.ang2pix, 'ring', nside)))(thetas, phis)
            self.assertTrue((true_ring == test_ring).all(), (true_ring, test_ring))

        for nside in test_nsides:
            true_nest = astropy_healpix.healpy.ang2pix(nside, thetas, phis, nest=True)
            test_nest = jax.jit(jax.vmap(partial(healjax.ang2pix, 'nest', nside)))(thetas, phis)
            self.assertTrue((true_nest == test_nest).all(), (true_nest, test_nest))

    def test_bulk(self):
        key1, key2 = jax.random.split(jax.random.PRNGKey(0))
        test_phis = numpy.array(jax.random.uniform(key1, minval=edge_eps, maxval=2*jnp.pi-edge_eps, shape=(n_test,)))
        test_thetas = numpy.array(jax.random.uniform(key2, minval=edge_eps, maxval=jnp.pi-edge_eps, shape=(n_test,)))

        for nside in test_nsides:
            true_hp_idxs_ring = astropy_healpix.healpy.ang2pix(nside, test_thetas, test_phis, nest=False)
            test_hp_idxs_ring = jax.jit(jax.vmap(partial(healjax.ang2pix, 'ring', nside)))(test_thetas, test_phis)
            self.assertTrue((true_hp_idxs_ring == test_hp_idxs_ring).all(), (true_hp_idxs_ring, test_hp_idxs_ring))
        
        for nside in test_nsides:
            true_hp_idxs_nest = astropy_healpix.healpy.ang2pix(nside, test_thetas, test_phis, nest=True)
            test_hp_idxs_nest = jax.jit(jax.vmap(partial(healjax.ang2pix, 'nest', nside)))(test_thetas, test_phis)
            self.assertTrue((true_hp_idxs_nest == test_hp_idxs_nest).all(), (true_hp_idxs_nest, test_hp_idxs_nest))

    def test_indices(self):
        error_tol = 1000
        for nside in test_nsides:
            hp_idxs = numpy.arange(astropy_healpix.nside_to_npix(nside))
            true_angs_ring = numpy.array(astropy_healpix.healpy.pix2ang(nside, hp_idxs, nest=False))
            test_angs_ring = numpy.array(jax.jit(jax.vmap(partial(healjax.pix2ang, 'ring', nside)))(hp_idxs))
            self.assertTrue((jnp.abs(true_angs_ring - test_angs_ring) <= error_tol*jnp.finfo(test_angs_ring.dtype).eps * jnp.abs(true_angs_ring)).all(), (nside, true_angs_ring, test_angs_ring, numpy.max(jnp.abs((true_angs_ring - test_angs_ring)))))

        for nside in test_nsides:
            hp_idxs = numpy.arange(astropy_healpix.nside_to_npix(nside))
            true_angs_nest = numpy.array(astropy_healpix.healpy.pix2ang(nside, hp_idxs, nest=True))
            test_angs_nest = numpy.array(jax.jit(jax.vmap(partial(healjax.pix2ang, 'nest', nside)))(hp_idxs))
            self.assertTrue((jnp.abs(true_angs_nest - test_angs_nest) <= error_tol*jnp.finfo(test_angs_nest.dtype).eps * jnp.abs(true_angs_nest)).all(), (nside, true_angs_nest, test_angs_nest, numpy.max(jnp.abs((true_angs_nest - test_angs_nest)))))

    def test_indices_sharp(self):
        error_tol = 1000
        for nside in test_nsides:
            hp_idxs = numpy.arange(astropy_healpix.nside_to_npix(nside))
            true_angs_ring = numpy.array(astropy_healpix.healpy.pix2ang(nside, hp_idxs, nest=False))
            test_angs_ring = numpy.array(jax.jit(jax.vmap(partial(healjax.pix2ang_colatlong, 'ring', nside)))(hp_idxs))
            self.assertTrue((jnp.abs(true_angs_ring - test_angs_ring) <= error_tol*jnp.finfo(test_angs_ring.dtype).eps * jnp.abs(true_angs_ring)).all(), (nside, true_angs_ring, test_angs_ring, numpy.max(jnp.abs((true_angs_ring - test_angs_ring)))))

        for nside in test_nsides:
            hp_idxs = numpy.arange(astropy_healpix.nside_to_npix(nside))
            true_angs_nest = numpy.array(astropy_healpix.healpy.pix2ang(nside, hp_idxs, nest=True))
            test_angs_nest = numpy.array(jax.jit(jax.vmap(partial(healjax.pix2ang_colatlong, 'nest', nside)))(hp_idxs))
            self.assertTrue((jnp.abs(true_angs_nest - test_angs_nest) <= error_tol*jnp.finfo(test_angs_nest.dtype).eps * jnp.abs(true_angs_nest)).all(), (nside, true_angs_nest, test_angs_nest, numpy.max(jnp.abs((true_angs_nest - test_angs_nest)))))

    def test_neighbours(self):
        # Capped at nside=128 to avoid OOM: nside=256 has ~786K pixels and
        # nside=1024 has ~12.6M pixels, which exhausts memory when vmapped.
        neighbour_nsides = [n for n in test_nsides if n <= 128]
        for nside in neighbour_nsides:
            hp_idxs = numpy.arange(astropy_healpix.nside_to_npix(nside))
            true_neighs_ring = astropy_healpix.neighbours(hp_idxs, nside, order='ring').T
            test_neighs_ring = numpy.array(jax.jit(jax.vmap(partial(healjax.get_neighbours, 'ring', nside)))(hp_idxs))
            self.assertTrue((true_neighs_ring  == test_neighs_ring ).all(), (true_neighs_ring , test_neighs_ring, true_neighs_ring - test_neighs_ring ))

        for nside in neighbour_nsides:
            hp_idxs = numpy.arange(astropy_healpix.nside_to_npix(nside))
            true_neighs_nest = astropy_healpix.neighbours(hp_idxs, nside, order='nested').T
            test_neighs_nest = numpy.array(jax.jit(jax.vmap(partial(healjax.get_neighbours, 'nest', nside)))(hp_idxs))
            self.assertTrue((true_neighs_nest  == test_neighs_nest ).all(), (true_neighs_nest , test_neighs_nest, true_neighs_nest - test_neighs_nest ))

    def test_convert(self):
        for nside in test_nsides:
            test_map = numpy.arange(astropy_healpix.nside_to_npix(nside))
            true_convert_to_nest = healpy.reorder(test_map, r2n=True)
            test_convert_to_nest = jax.jit(partial(healjax.convert_map, 'ring', 'nest'))(test_map)
            self.assertTrue((true_convert_to_nest == test_convert_to_nest).all(), (true_convert_to_nest, test_convert_to_nest))

        for nside in test_nsides:
            test_map = numpy.arange(astropy_healpix.nside_to_npix(nside))
            true_convert_to_ring = healpy.reorder(test_map, n2r=True)
            test_convert_to_ring = jax.jit(partial(healjax.convert_map, 'nest', 'ring'))(test_map)
            self.assertTrue((true_convert_to_ring == test_convert_to_ring).all(), (true_convert_to_ring, test_convert_to_ring))

    def test_convert_identity(self):
        for scheme in ['ring', 'nest', 'xy']:
            for nside in test_nsides:
                test_map = numpy.arange(astropy_healpix.nside_to_npix(nside))
                id_map = jax.jit(partial(healjax.convert_map, scheme, scheme))(test_map)
                self.assertTrue((test_map == id_map).all(), (scheme, test_map, id_map))

    def test_convert_from_xy(self):
        error_tol = 1e7
        for nside in test_nsides:
            truth_ras, truth_decs = astropy_healpix.healpy.pix2ang(nside, numpy.arange(astropy_healpix.nside_to_npix(nside)), lonlat=True, nest=True)
            truth_map = (truth_ras + 2*numpy.pi*truth_decs)*(numpy.pi/180)

            test_ras, test_decs = jax.jit(jax.vmap(partial(healjax.pix2ang_radec, 'xy', nside)))(numpy.arange(astropy_healpix.nside_to_npix(nside)))
            test_map_xy = test_ras + 2*jnp.pi*test_decs
            test_map = jax.jit(partial(healjax.convert_map, 'xy', 'nest'))(test_map_xy)

            self.assertTrue((numpy.abs(truth_map - test_map) <= error_tol * numpy.finfo(truth_map.dtype).eps * numpy.abs(truth_map)).all(), (truth_map, test_map, test_map_xy, numpy.max(numpy.abs(truth_map - test_map)/(numpy.finfo(truth_map.dtype).eps * numpy.abs(truth_map)))))

        for nside in test_nsides:
            truth_ras, truth_decs = astropy_healpix.healpy.pix2ang(nside, numpy.arange(astropy_healpix.nside_to_npix(nside)), lonlat=True)
            truth_map = (truth_ras + 2*numpy.pi*truth_decs)*(numpy.pi/180)

            test_ras, test_decs = jax.jit(jax.vmap(partial(healjax.pix2ang_radec, 'xy', nside)))(numpy.arange(astropy_healpix.nside_to_npix(nside)))
            test_map_xy = test_ras + 2*jnp.pi*test_decs
            test_map = jax.jit(partial(healjax.convert_map, 'xy', 'ring'))(test_map_xy)

            self.assertTrue((numpy.abs(truth_map - test_map) <= error_tol * numpy.finfo(truth_map.dtype).eps * numpy.abs(truth_map)).all(), (truth_map, test_map, test_map_xy, numpy.max(numpy.abs(truth_map - test_map)/(numpy.finfo(truth_map.dtype).eps * numpy.abs(truth_map)))))

    def test_convert_to_xy(self):
        error_tol = 1e7
        for nside in test_nsides:
            truth_ras, truth_decs = astropy_healpix.healpy.pix2ang(nside, numpy.arange(astropy_healpix.nside_to_npix(nside)), lonlat=True, nest=True)
            truth_map = (truth_ras + 2*numpy.pi*truth_decs)*(numpy.pi/180)
            truth_map_xy = jax.jit(partial(healjax.convert_map, 'nest', 'xy'))(truth_map)

            test_ras, test_decs = jax.jit(jax.vmap(partial(healjax.pix2ang_radec, 'xy', nside)))(numpy.arange(astropy_healpix.nside_to_npix(nside)))
            test_map_xy = test_ras + 2*jnp.pi*test_decs

            self.assertTrue((numpy.abs(truth_map_xy - test_map_xy) <= error_tol * numpy.finfo(truth_map.dtype).eps * numpy.abs(truth_map_xy)).all(), (truth_map_xy, test_map_xy, truth_map))

        for nside in test_nsides:
            truth_ras, truth_decs = astropy_healpix.healpy.pix2ang(nside, numpy.arange(astropy_healpix.nside_to_npix(nside)), lonlat=True)
            truth_map = (truth_ras + 2*numpy.pi*truth_decs)*(numpy.pi/180)
            truth_map_xy = jax.jit(partial(healjax.convert_map, 'ring', 'xy'))(truth_map)

            test_ras, test_decs = jax.jit(jax.vmap(partial(healjax.pix2ang_radec, 'xy', nside)))(numpy.arange(astropy_healpix.nside_to_npix(nside)))
            test_map_xy = test_ras + 2*jnp.pi*test_decs

            self.assertTrue((numpy.abs(truth_map_xy - test_map_xy) <= error_tol * numpy.finfo(truth_map.dtype).eps * numpy.abs(truth_map_xy)).all(), (truth_map_xy, test_map_xy, truth_map))


class TestInterpolationMethods(unittest.TestCase):
    def test_get_interp_weights(self):
        numpy.random.seed(0)
        theta_all =  numpy.random.uniform(0, numpy.pi, 900)
        phi_all =  numpy.random.uniform(0, numpy.pi, 2*900)
        _phi, _theta = numpy.meshgrid(phi_all, theta_all)
        _phi = _phi.astype(healjax.FLOAT_TYPE)
        _theta = _theta.astype(healjax.FLOAT_TYPE)
        get_interpol = jax.jit(healjax.get_interp_weights)
        for nside in test_nsides:
            nside = healjax.INT_TYPE(nside)
            data = numpy.random.normal(size=12*nside**2).astype(healjax.FLOAT_TYPE)
            pix_true, wgts_true = astropy_healpix.healpy.get_interp_weights(nside, _theta, _phi, nest=False)
            dinterp_true = numpy.sum(data[pix_true] * wgts_true, axis=0)
            pix, wgts = get_interpol(_theta, _phi, nside)
            dinterp = numpy.sum(data[pix] * wgts, axis=0)
            diff = (dinterp - dinterp_true)
            x, y = numpy.where(numpy.abs(diff) > 3e-3)  # XXX increasing error with nside
            self.assertTrue(x.size == 0)

class TestCoordinateConversions(unittest.TestCase):
    """Tests for ang2vec, vec2ang, ang2vec_radec, vec2ang_radec round-trips
    and agreement with numpy reference values."""

    def test_ang2vec_unit_norm(self):
        """ang2vec output should be unit vectors."""
        key1, key2 = jax.random.split(jax.random.PRNGKey(42))
        thetas = numpy.array(jax.random.uniform(key1, minval=edge_eps, maxval=jnp.pi - edge_eps, shape=(10000,)))
        phis   = numpy.array(jax.random.uniform(key2, minval=edge_eps, maxval=2*jnp.pi - edge_eps, shape=(10000,)))
        x, y, z = jax.vmap(healjax.ang2vec)(thetas, phis)
        norms = numpy.sqrt(numpy.array(x)**2 + numpy.array(y)**2 + numpy.array(z)**2)
        self.assertTrue(numpy.allclose(norms, 1.0, atol=1e-6), f"max norm deviation: {numpy.max(numpy.abs(norms - 1.0))}")

    def test_ang2vec_radec_unit_norm(self):
        """ang2vec_radec output should be unit vectors."""
        key1, key2 = jax.random.split(jax.random.PRNGKey(43))
        ras  = numpy.array(jax.random.uniform(key1, minval=edge_eps, maxval=2*jnp.pi - edge_eps, shape=(10000,)))
        decs = numpy.array(jax.random.uniform(key2, minval=-jnp.pi/2 + edge_eps, maxval=jnp.pi/2 - edge_eps, shape=(10000,)))
        x, y, z = jax.vmap(healjax.ang2vec_radec)(ras, decs)
        norms = numpy.sqrt(numpy.array(x)**2 + numpy.array(y)**2 + numpy.array(z)**2)
        self.assertTrue(numpy.allclose(norms, 1.0, atol=1e-6), f"max norm deviation: {numpy.max(numpy.abs(norms - 1.0))}")

    def test_vec2ang_roundtrip(self):
        """ang2vec → vec2ang should recover original (theta, phi)."""
        key1, key2 = jax.random.split(jax.random.PRNGKey(44))
        thetas = numpy.array(jax.random.uniform(key1, minval=edge_eps, maxval=jnp.pi - edge_eps, shape=(10000,)))
        phis   = numpy.array(jax.random.uniform(key2, minval=edge_eps, maxval=2*jnp.pi - edge_eps, shape=(10000,)))
        x, y, z = jax.vmap(healjax.ang2vec)(thetas, phis)
        theta_rt, phi_rt = jax.vmap(healjax.vec2ang)(x, y, z)
        self.assertTrue(numpy.allclose(numpy.array(theta_rt), thetas, atol=1e-6),
                        f"theta round-trip max error: {numpy.max(numpy.abs(numpy.array(theta_rt) - thetas))}")
        self.assertTrue(numpy.allclose(numpy.array(phi_rt), phis, atol=1e-6),
                        f"phi round-trip max error: {numpy.max(numpy.abs(numpy.array(phi_rt) - phis))}")

    def test_vec2ang_radec_roundtrip(self):
        """ang2vec_radec → vec2ang_radec should recover original (ra, dec)."""
        key1, key2 = jax.random.split(jax.random.PRNGKey(45))
        ras  = numpy.array(jax.random.uniform(key1, minval=edge_eps, maxval=2*jnp.pi - edge_eps, shape=(10000,)))
        decs = numpy.array(jax.random.uniform(key2, minval=-jnp.pi/2 + edge_eps, maxval=jnp.pi/2 - edge_eps, shape=(10000,)))
        x, y, z = jax.vmap(healjax.ang2vec_radec)(ras, decs)
        ra_rt, dec_rt = jax.vmap(healjax.vec2ang_radec)(x, y, z)
        self.assertTrue(numpy.allclose(numpy.array(ra_rt), ras, atol=1e-6),
                        f"ra round-trip max error: {numpy.max(numpy.abs(numpy.array(ra_rt) - ras))}")
        self.assertTrue(numpy.allclose(numpy.array(dec_rt), decs, atol=1e-6),
                        f"dec round-trip max error: {numpy.max(numpy.abs(numpy.array(dec_rt) - decs))}")

    def test_ang2vec_specific_values(self):
        """Check ang2vec against known analytical values."""
        # North pole: theta=0 → (0, 0, 1)
        x, y, z = healjax.ang2vec(0.0, 0.0)
        self.assertAlmostEqual(float(z), 1.0, places=6)
        # South pole: theta=pi → (0, 0, -1)
        x, y, z = healjax.ang2vec(jnp.pi, 0.0)
        self.assertAlmostEqual(float(z), -1.0, places=6)
        # Equator, phi=0: theta=pi/2, phi=0 → (1, 0, 0)
        x, y, z = healjax.ang2vec(jnp.pi/2, 0.0)
        self.assertAlmostEqual(float(x), 1.0, places=6)
        self.assertAlmostEqual(float(z), 0.0, places=6)


class TestPixelConversionsExtended(unittest.TestCase):
    """Tests for vec2pix, ang2pix_radec, pix2vec, and pix2ang dx/dy offsets."""

    def test_vec2pix_vs_healpy(self):
        """vec2pix should match healpy.vec2pix for ring and nest schemes."""
        key1, key2 = jax.random.split(jax.random.PRNGKey(10))
        thetas = numpy.array(jax.random.uniform(key1, minval=edge_eps, maxval=jnp.pi - edge_eps, shape=(n_test,)))
        phis   = numpy.array(jax.random.uniform(key2, minval=edge_eps, maxval=2*jnp.pi - edge_eps, shape=(n_test,)))
        # Convert to unit vectors using numpy for clean reference
        vx = numpy.sin(thetas) * numpy.cos(phis)
        vy = numpy.sin(thetas) * numpy.sin(phis)
        vz = numpy.cos(thetas)

        for nside in test_nsides:
            true_ring = astropy_healpix.healpy.vec2pix(nside, vx, vy, vz, nest=False)
            test_ring = numpy.array(jax.jit(jax.vmap(partial(healjax.vec2pix, 'ring', nside)))(vx, vy, vz))
            self.assertTrue((true_ring == test_ring).all(),
                            f"nside={nside} ring: mismatch at {numpy.sum(true_ring != test_ring)} pixels")

        for nside in test_nsides:
            true_nest = astropy_healpix.healpy.vec2pix(nside, vx, vy, vz, nest=True)
            test_nest = numpy.array(jax.jit(jax.vmap(partial(healjax.vec2pix, 'nest', nside)))(vx, vy, vz))
            self.assertTrue((true_nest == test_nest).all(),
                            f"nside={nside} nest: mismatch at {numpy.sum(true_nest != test_nest)} pixels")

    def test_ang2pix_radec_vs_healpy(self):
        """ang2pix_radec should match healpy.ang2pix with lonlat=True for ring and nest."""
        key1, key2 = jax.random.split(jax.random.PRNGKey(11))
        ras  = numpy.array(jax.random.uniform(key1, minval=edge_eps, maxval=2*jnp.pi - edge_eps, shape=(n_test,)))
        decs = numpy.array(jax.random.uniform(key2, minval=-jnp.pi/2 + edge_eps, maxval=jnp.pi/2 - edge_eps, shape=(n_test,)))
        # healpy expects lonlat in degrees
        ras_deg  = numpy.degrees(ras)
        decs_deg = numpy.degrees(decs)

        for nside in test_nsides:
            true_ring = astropy_healpix.healpy.ang2pix(nside, ras_deg, decs_deg, nest=False, lonlat=True)
            test_ring = numpy.array(jax.jit(jax.vmap(partial(healjax.ang2pix_radec, 'ring', nside)))(ras, decs))
            self.assertTrue((true_ring == test_ring).all(),
                            f"nside={nside} ring: {numpy.sum(true_ring != test_ring)} mismatches")

        for nside in test_nsides:
            true_nest = astropy_healpix.healpy.ang2pix(nside, ras_deg, decs_deg, nest=True, lonlat=True)
            test_nest = numpy.array(jax.jit(jax.vmap(partial(healjax.ang2pix_radec, 'nest', nside)))(ras, decs))
            self.assertTrue((true_nest == test_nest).all(),
                            f"nside={nside} nest: {numpy.sum(true_nest != test_nest)} mismatches")

    def test_pix2vec_vs_healpy(self):
        """pix2vec should match healpy.pix2vec for all pixels at several nsides."""
        error_tol = 1000
        for nside in test_nsides:
            hp_idxs = numpy.arange(astropy_healpix.nside_to_npix(nside))
            true_vecs_ring = numpy.array(astropy_healpix.healpy.pix2vec(nside, hp_idxs, nest=False))
            test_vecs_ring = numpy.array(jax.jit(jax.vmap(partial(healjax.pix2vec, 'ring', nside)))(hp_idxs))
            eps = numpy.finfo(test_vecs_ring.dtype).eps
            diff = numpy.abs(true_vecs_ring - test_vecs_ring)
            scale = numpy.maximum(numpy.abs(true_vecs_ring), eps)
            self.assertTrue((diff <= error_tol * eps * scale).all(),
                            f"nside={nside} ring pix2vec: max rel err={numpy.max(diff/scale):.2e}")

        for nside in test_nsides:
            hp_idxs = numpy.arange(astropy_healpix.nside_to_npix(nside))
            true_vecs_nest = numpy.array(astropy_healpix.healpy.pix2vec(nside, hp_idxs, nest=True))
            test_vecs_nest = numpy.array(jax.jit(jax.vmap(partial(healjax.pix2vec, 'nest', nside)))(hp_idxs))
            eps = numpy.finfo(test_vecs_nest.dtype).eps
            diff = numpy.abs(true_vecs_nest - test_vecs_nest)
            scale = numpy.maximum(numpy.abs(true_vecs_nest), eps)
            self.assertTrue((diff <= error_tol * eps * scale).all(),
                            f"nside={nside} nest pix2vec: max rel err={numpy.max(diff/scale):.2e}")

    def test_pix2ang_dx_dy_within_pixel(self):
        """pix2ang with dx/dy in [0,1] should stay within the pixel's angular bounds.
        Specifically, the center (dx=0.5, dy=0.5) should be the closest direction
        to the pixel boundary midpoints returned by the dx=0.0 and dx=1.0 corners."""
        nside = 16
        hp_idxs = numpy.arange(astropy_healpix.nside_to_npix(nside))

        # Get center and corner angles for ring scheme
        theta_center, phi_center = numpy.array(
            jax.jit(jax.vmap(partial(healjax.pix2ang, 'ring', nside)))(hp_idxs))
        theta_corner00, phi_corner00 = numpy.array(
            jax.jit(jax.vmap(partial(healjax.pix2ang, 'ring', nside, dx=0.0, dy=0.0)))(hp_idxs))
        theta_corner11, phi_corner11 = numpy.array(
            jax.jit(jax.vmap(partial(healjax.pix2ang, 'ring', nside, dx=1.0, dy=1.0)))(hp_idxs))

        # All angles must be in valid range
        self.assertTrue((theta_center >= 0).all() and (theta_center <= jnp.pi).all(),
                        "Center theta out of range")
        self.assertTrue((theta_corner00 >= 0).all() and (theta_corner00 <= jnp.pi).all(),
                        "Corner (0,0) theta out of range")
        self.assertTrue((theta_corner11 >= 0).all() and (theta_corner11 <= jnp.pi).all(),
                        "Corner (1,1) theta out of range")

        # Verify pix2vec with corners are unit vectors
        x00, y00, z00 = numpy.array(
            jax.jit(jax.vmap(partial(healjax.pix2vec, 'ring', nside, dx=0.0, dy=0.0)))(hp_idxs))
        norms = numpy.sqrt(x00**2 + y00**2 + z00**2)
        self.assertTrue(numpy.allclose(norms, 1.0, atol=1e-6),
                        f"pix2vec corner norms: max deviation {numpy.max(numpy.abs(norms - 1.0))}")


class TestInternalFunctions(unittest.TestCase):
    """Tests for ring_above and get_ring_info2 against healpy ring structure."""

    def _healpy_ring_info(self, ring, nside):
        """Compute reference ring info using astropy_healpix pixel structure."""
        npix = 12 * nside * nside
        ncap = 2 * nside * (nside - 1)
        northring = (4 * nside - ring) if ring > 2 * nside else ring
        if northring < nside:
            # polar cap
            ringpix  = 4 * northring
            startpix = 2 * northring * (northring - 1)
            shifted  = True
            costheta = 1 - northring**2 / (3 * nside**2)
            theta    = numpy.arccos(costheta)
        else:
            # equatorial
            ringpix  = 4 * nside
            startpix = ncap + (northring - nside) * ringpix
            shifted  = ((northring - nside) % 2 == 0)
            theta    = numpy.arccos((2 * nside - northring) * 2 / (3 * nside))
        if northring != ring:
            # southern hemisphere mirror
            theta    = numpy.pi - theta
            startpix = npix - startpix - ringpix
        return startpix, ringpix, theta, shifted

    def test_get_ring_info2(self):
        """get_ring_info2 should match reference ring info for all rings."""
        for nside in [2, 4, 8, 16, 32]:
            for ring in range(1, 4 * nside):
                sp_true, rp_true, th_true, sh_true = self._healpy_ring_info(ring, nside)
                ring_j = healjax.INT_TYPE(ring)
                nside_j = healjax.INT_TYPE(nside)
                sp, rp, th, sh = healjax.get_ring_info2(ring_j, nside_j)
                self.assertEqual(int(sp), sp_true, f"startpix mismatch nside={nside} ring={ring}")
                self.assertEqual(int(rp), rp_true, f"ringpix mismatch nside={nside} ring={ring}")
                self.assertAlmostEqual(float(th), th_true, places=5,
                                       msg=f"theta mismatch nside={nside} ring={ring}")
                self.assertEqual(bool(sh), sh_true, f"shifted mismatch nside={nside} ring={ring}")

    def test_ring_above(self):
        """ring_above(cos(theta), nside) should return the ring just above theta."""
        for nside in [4, 16, 64]:
            # Sample colatitudes avoiding exact ring boundaries
            thetas = numpy.linspace(0.01, numpy.pi - 0.01, 500)
            zs = numpy.cos(thetas)
            rings = numpy.array(jax.vmap(lambda z: healjax.ring_above(z, healjax.INT_TYPE(nside)))(
                jnp.array(zs, dtype=healjax.FLOAT_TYPE)))
            # ring_above returns the ring whose theta is just above (smaller than) the input theta.
            # So theta of ring_above(z) <= theta_input < theta of (ring_above(z)+1).
            for i, (ring, theta_in) in enumerate(zip(rings, thetas)):
                ring = int(ring)
                if ring == 0:
                    continue  # at north pole, ring 0 means above all rings
                _, _, theta_ring, _ = self._healpy_ring_info(ring, nside)
                self.assertLessEqual(theta_ring, theta_in + 1e-4,
                                     f"ring_above wrong: nside={nside}, theta={theta_in:.4f}, ring={ring}, ring_theta={theta_ring:.4f}")


class TestSchemeConversions(unittest.TestCase):
    """Tests for scheme2bighpxy/bighpxy2scheme round-trips and get_patch."""

    def test_scheme2bighpxy_bighpxy2scheme_roundtrip_ring(self):
        """scheme2bighpxy → bighpxy2scheme should be identity for ring scheme."""
        for nside in [2, 4, 8, 32]:
            hp_idxs = numpy.arange(astropy_healpix.nside_to_npix(nside))
            def roundtrip(hp):
                bighp, x, y = healjax.scheme2bighpxy('ring', nside, hp)
                return healjax.bighpxy2scheme('ring', nside, bighp, x, y)
            recovered = numpy.array(jax.jit(jax.vmap(roundtrip))(hp_idxs))
            self.assertTrue((recovered == hp_idxs).all(),
                            f"ring round-trip failed at nside={nside}: {numpy.sum(recovered != hp_idxs)} errors")

    def test_scheme2bighpxy_bighpxy2scheme_roundtrip_nest(self):
        """scheme2bighpxy → bighpxy2scheme should be identity for nest scheme."""
        for nside in [2, 4, 8, 32]:
            hp_idxs = numpy.arange(astropy_healpix.nside_to_npix(nside))
            def roundtrip(hp):
                bighp, x, y = healjax.scheme2bighpxy('nest', nside, hp)
                return healjax.bighpxy2scheme('nest', nside, bighp, x, y)
            recovered = numpy.array(jax.jit(jax.vmap(roundtrip))(hp_idxs))
            self.assertTrue((recovered == hp_idxs).all(),
                            f"nest round-trip failed at nside={nside}: {numpy.sum(recovered != hp_idxs)} errors")

    def test_get_patch_center_is_input(self):
        """get_patch should have the input pixel at the centre [1,1] position."""
        for nside in [4, 16, 64]:
            hp_idxs = numpy.arange(astropy_healpix.nside_to_npix(nside))
            patches = numpy.array(jax.jit(jax.vmap(partial(healjax.get_patch, 'ring', nside)))(hp_idxs))
            # patches shape: (npix, 3, 3)
            center = patches[:, 1, 1]
            self.assertTrue((center == hp_idxs).all(),
                            f"nside={nside}: centre of patch != input pixel at {numpy.sum(center != hp_idxs)} pixels")

    def test_get_patch_neighbours_consistent(self):
        """The 8 non-center elements of get_patch should contain the same values as get_neighbours
        (though possibly in a different arrangement)."""
        nside = 16
        hp_idxs = numpy.arange(astropy_healpix.nside_to_npix(nside))
        patches   = numpy.array(jax.jit(jax.vmap(partial(healjax.get_patch, 'ring', nside)))(hp_idxs))
        neighs    = numpy.array(jax.jit(jax.vmap(partial(healjax.get_neighbours, 'ring', nside)))(hp_idxs))
        # Extract the 8 off-center elements from the 3x3 patch (exclude [1,1])
        mask = numpy.ones((3, 3), dtype=bool)
        mask[1, 1] = False
        patch_neighs = patches[:, mask]  # shape (npix, 8)
        # Both sets should be equal when sorted
        patch_sorted = numpy.sort(patch_neighs, axis=1)
        neigh_sorted = numpy.sort(neighs, axis=1)
        self.assertTrue((patch_sorted == neigh_sorted).all(),
                        f"patch and neighbours disagree at {numpy.sum(numpy.any(patch_sorted != neigh_sorted, axis=1))} pixels")


class TestUtilities(unittest.TestCase):
    """Tests for nside2npix, npix2nside, and get_nside."""

    def test_nside2npix(self):
        """nside2npix should match astropy_healpix.nside_to_npix."""
        for nside in test_nsides:
            expected = astropy_healpix.nside_to_npix(nside)
            result   = healjax.nside2npix(nside)
            self.assertEqual(result, expected, f"nside2npix({nside}): got {result}, expected {expected}")

    def test_npix2nside(self):
        """npix2nside should invert nside2npix."""
        for nside in test_nsides:
            npix   = healjax.nside2npix(nside)
            result = healjax.npix2nside(npix)
            self.assertEqual(result, nside, f"npix2nside({npix}): got {result}, expected {nside}")

    def test_get_nside(self):
        """get_nside should extract correct nside from a map array."""
        for nside in test_nsides:
            test_map = numpy.zeros(healjax.nside2npix(nside))
            result   = healjax.get_nside(test_map)
            self.assertEqual(result, nside, f"get_nside for nside={nside}: got {result}")


class TestDifferentiability(unittest.TestCase):
    """Tests that JAX differentiability works through the continuous-output functions."""

    def test_pix2vec_grad_dx_dy(self):
        """jax.grad through pix2vec w.r.t. dx and dy should give finite non-zero values.
        Both dx and dy must be the same FLOAT_TYPE to avoid dtype promotion issues inside
        hp_to_zphi_polar where x and y positions are swapped between cond branches."""
        nside = 16
        # Pick a pixel in the equatorial band (not on a face edge)
        hp = healjax.INT_TYPE(100)
        half = healjax.FLOAT_TYPE(0.5)

        def fn_dx(dx):
            x, y, z = healjax.pix2vec('ring', nside, hp, dx=dx, dy=half)
            return x + y + z

        def fn_dy(dy):
            x, y, z = healjax.pix2vec('ring', nside, hp, dx=half, dy=dy)
            return x + y + z

        grad_dx = jax.grad(fn_dx)(half)
        grad_dy = jax.grad(fn_dy)(half)

        self.assertTrue(jnp.isfinite(grad_dx), f"grad w.r.t. dx is not finite: {grad_dx}")
        self.assertTrue(jnp.isfinite(grad_dy), f"grad w.r.t. dy is not finite: {grad_dy}")
        self.assertNotEqual(float(grad_dx), 0.0, "grad w.r.t. dx is zero")
        self.assertNotEqual(float(grad_dy), 0.0, "grad w.r.t. dy is zero")

    def test_pix2ang_radec_grad_dx_dy(self):
        """jax.grad through pix2ang_radec w.r.t. dx and dy should be finite."""
        nside = 16
        hp = healjax.INT_TYPE(200)

        def fn(dx, dy):
            ra, dec = healjax.pix2ang_radec('ring', nside, hp, dx=dx, dy=dy)
            return ra + dec

        grad_fn = jax.grad(fn, argnums=(0, 1))
        g_dx, g_dy = grad_fn(healjax.FLOAT_TYPE(0.5), healjax.FLOAT_TYPE(0.5))

        self.assertTrue(jnp.isfinite(g_dx), f"grad pix2ang_radec w.r.t. dx not finite: {g_dx}")
        self.assertTrue(jnp.isfinite(g_dy), f"grad pix2ang_radec w.r.t. dy not finite: {g_dy}")

    def test_interpol_weights_sum_to_one(self):
        """Interpolation weights should always sum to 1.0 (partition of unity)."""
        numpy.random.seed(99)
        thetas = numpy.random.uniform(edge_eps, numpy.pi - edge_eps, 500).astype(healjax.FLOAT_TYPE)
        phis   = numpy.random.uniform(edge_eps, 2*numpy.pi - edge_eps, 500).astype(healjax.FLOAT_TYPE)
        for nside in [4, 16, 64]:
            nside_j = healjax.INT_TYPE(nside)
            pix, wgt = healjax.get_interp_weights(thetas, phis, nside_j)
            wgt_sum = numpy.array(wgt).sum(axis=0)
            self.assertTrue(numpy.allclose(wgt_sum, 1.0, atol=1e-5),
                            f"nside={nside}: weights don't sum to 1; max deviation={numpy.max(numpy.abs(wgt_sum - 1.0)):.2e}")

    def test_interpol_pixel_indices_valid(self):
        """All pixel indices from get_interp_weights should be in [0, npix)."""
        numpy.random.seed(88)
        thetas = numpy.random.uniform(edge_eps, numpy.pi - edge_eps, 500).astype(healjax.FLOAT_TYPE)
        phis   = numpy.random.uniform(edge_eps, 2*numpy.pi - edge_eps, 500).astype(healjax.FLOAT_TYPE)
        for nside in [4, 16, 64]:
            nside_j = healjax.INT_TYPE(nside)
            npix = healjax.nside2npix(nside)
            pix, wgt = healjax.get_interp_weights(thetas, phis, nside_j)
            pix_arr = numpy.array(pix)
            self.assertTrue((pix_arr >= 0).all() and (pix_arr < npix).all(),
                            f"nside={nside}: pixel indices out of range; min={pix_arr.min()}, max={pix_arr.max()}, npix={npix}")


if __name__ == '__main__':
    unittest.main()
