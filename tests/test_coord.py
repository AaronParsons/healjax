"""Tests for healjax.coord."""

import jax

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from healjax.coord import (
    angles_to_coord,
    azalt2top,
    eq2radec,
    eq2top_m,
    latlong2xyz,
    radec2eq,
    rot_m,
    thphi2xyz,
    top2azalt,
    top2eq_m,
    xyz2thphi,
)

X, Y, Z = np.eye(3)


# ---------------------------------------------------------------------------
# rot_m
# ---------------------------------------------------------------------------

class TestRotM:

    def test_identity_at_zero_angle(self):
        for axis in (X, Y, Z):
            np.testing.assert_allclose(rot_m(0.0, axis), np.eye(3), atol=1e-15)

    def test_quarter_turn_about_z(self):
        m = rot_m(np.pi / 2, Z)
        np.testing.assert_allclose(m @ X, Y, atol=1e-15)
        np.testing.assert_allclose(m @ Y, -X, atol=1e-15)
        np.testing.assert_allclose(m @ Z, Z, atol=1e-15)

    def test_right_hand_rule_about_x_and_y(self):
        np.testing.assert_allclose(rot_m(np.pi / 2, X) @ Y, Z, atol=1e-15)
        np.testing.assert_allclose(rot_m(np.pi / 2, Y) @ Z, X, atol=1e-15)

    def test_orthogonal_and_unit_determinant(self):
        rng = np.random.default_rng(0)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        m = rot_m(0.7, axis)
        np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(m), 1.0)

    def test_axis_is_invariant(self):
        axis = np.array([1.0, 2.0, -3.0])
        axis /= np.linalg.norm(axis)
        np.testing.assert_allclose(rot_m(1.234, axis) @ axis, axis, atol=1e-12)

    def test_composition_is_additive(self):
        a, b = 0.3, 0.9
        np.testing.assert_allclose(
            rot_m(a, Z) @ rot_m(b, Z), rot_m(a + b, Z), atol=1e-12
        )

    def test_inverse_is_negative_angle(self):
        np.testing.assert_allclose(
            rot_m(0.5, X) @ rot_m(-0.5, X), np.eye(3), atol=1e-12
        )

    def test_scales_with_axis_magnitude(self):
        """|vec| != 1 scales the matrix, as documented."""
        m1 = rot_m(0.4, Z)
        m2 = rot_m(0.4, 2 * Z)
        assert not np.allclose(m1, m2)

    def test_batched_angles(self):
        angs = np.array([0.0, np.pi / 2, np.pi])
        vecs = np.tile(Z, (3, 1))
        batch = rot_m(angs, vecs)
        assert batch.shape == (3, 3, 3)
        for i, a in enumerate(angs):
            np.testing.assert_allclose(batch[i], rot_m(a, Z), atol=1e-15)

    def test_jax_input_returns_jax_array(self):
        m = rot_m(jnp.float64(np.pi / 2), jnp.asarray(Z))
        assert isinstance(m, jax.Array)
        np.testing.assert_allclose(np.asarray(m), rot_m(np.pi / 2, Z),
                                   atol=1e-12)

    def test_jittable(self):
        f = jax.jit(rot_m)
        m = f(jnp.float64(0.7), jnp.asarray(X))
        np.testing.assert_allclose(np.asarray(m), rot_m(0.7, X), atol=1e-12)

    def test_matches_legacy_aipy_formula(self):
        """Guard against drift from the triplicated aipy.coord implementation."""
        rng = np.random.default_rng(1)
        ang = rng.uniform(0, 2 * np.pi)
        vec = rng.normal(size=3)
        c, s, C = np.cos(ang), np.sin(ang), 1 - np.cos(ang)
        x, y, z = vec
        expected = np.array([
            [x * x * C + c, x * y * C - z * s, z * x * C + y * s],
            [x * y * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, y * z * C + x * s, z * z * C + c],
        ])
        np.testing.assert_allclose(rot_m(ang, vec), expected, atol=1e-14)


# ---------------------------------------------------------------------------
# xyz2thphi / thphi2xyz
# ---------------------------------------------------------------------------

class TestSphericalCartesian:

    def test_poles_and_axes(self):
        np.testing.assert_allclose(xyz2thphi(0, 0, 1), [0.0, 0.0], atol=1e-15)
        th, phi = xyz2thphi(0, 0, -1)
        assert np.isclose(th, np.pi)
        np.testing.assert_allclose(xyz2thphi(1, 0, 0), [np.pi / 2, 0.0],
                                   atol=1e-15)
        np.testing.assert_allclose(xyz2thphi(0, 1, 0),
                                   [np.pi / 2, np.pi / 2], atol=1e-15)

    def test_thphi2xyz_known_values(self):
        np.testing.assert_allclose(thphi2xyz(0.0, 0.0), [0, 0, 1], atol=1e-15)
        np.testing.assert_allclose(thphi2xyz(np.pi / 2, 0.0), [1, 0, 0],
                                   atol=1e-15)
        np.testing.assert_allclose(thphi2xyz(np.pi / 2, np.pi / 2),
                                   [0, 1, 0], atol=1e-15)

    def test_round_trip(self):
        rng = np.random.default_rng(2)
        th = rng.uniform(0, np.pi, 50)
        phi = rng.uniform(-np.pi, np.pi, 50)
        xyz = thphi2xyz(th, phi)
        th2, phi2 = xyz2thphi(xyz)
        np.testing.assert_allclose(th2, th, atol=1e-12)
        np.testing.assert_allclose(np.cos(phi2), np.cos(phi), atol=1e-12)
        np.testing.assert_allclose(np.sin(phi2), np.sin(phi), atol=1e-12)

    def test_both_calling_conventions_agree(self):
        rng = np.random.default_rng(3)
        xyz = rng.normal(size=(3, 20))
        np.testing.assert_allclose(xyz2thphi(xyz), xyz2thphi(*xyz))
        th_phi = xyz2thphi(xyz)
        np.testing.assert_allclose(thphi2xyz(th_phi), thphi2xyz(*th_phi))

    def test_mismatched_partial_args_raise(self):
        with pytest.raises(TypeError):
            xyz2thphi(np.zeros(3), np.zeros(3))

    def test_output_is_unit_length(self):
        rng = np.random.default_rng(4)
        th = rng.uniform(0, np.pi, 30)
        phi = rng.uniform(0, 2 * np.pi, 30)
        xyz = thphi2xyz(th, phi)
        np.testing.assert_allclose(np.linalg.norm(xyz, axis=0), 1.0,
                                   atol=1e-14)

    def test_masked_array_support(self):
        x = np.ma.array([1.0, 0.0, 0.0], mask=[False, True, False])
        y = np.ma.array([0.0, 1.0, 0.0], mask=[False, True, False])
        z = np.ma.array([0.0, 0.0, 1.0], mask=[False, True, False])
        out, mask = xyz2thphi(x, y, z, return_mask=True)
        assert out.shape == (2, 3)
        assert mask.shape == (2, 3)
        assert mask[0, 1] and not mask[0, 0]
        np.testing.assert_allclose(out[:, 0], [np.pi / 2, 0.0], atol=1e-15)

    def test_jax_arrays_stay_jax(self):
        out = xyz2thphi(jnp.asarray([1.0]), jnp.asarray([0.0]),
                        jnp.asarray([0.0]))
        assert isinstance(out, jax.Array)
        assert isinstance(thphi2xyz(jnp.asarray([0.1]), jnp.asarray([0.2])),
                          jax.Array)

    def test_numpy_arrays_stay_numpy(self):
        out = xyz2thphi(np.array([1.0]), np.array([0.0]), np.array([0.0]))
        assert isinstance(out, np.ndarray) and not isinstance(out, jax.Array)

    def test_agrees_with_healpy_vec2ang(self):
        healpy = pytest.importorskip("healpy")
        rng = np.random.default_rng(5)
        v = rng.normal(size=(3, 100))
        v /= np.linalg.norm(v, axis=0)
        th_hp, phi_hp = healpy.vec2ang(v.T)
        th, phi = xyz2thphi(v)
        np.testing.assert_allclose(th, th_hp, atol=1e-12)
        np.testing.assert_allclose(np.mod(phi, 2 * np.pi), phi_hp, atol=1e-12)

    def test_angles_to_coord_degrees(self):
        np.testing.assert_allclose(angles_to_coord(0.0, 0.0), [0, 0, 1],
                                   atol=1e-15)
        np.testing.assert_allclose(angles_to_coord(90.0, 90.0), [0, 1, 0],
                                   atol=1e-15)


# ---------------------------------------------------------------------------
# Named coordinate systems
# ---------------------------------------------------------------------------

class TestNamedSystems:

    def test_eq2radec_wraps_ra(self):
        ra, dec = eq2radec(np.array([[1.0], [-1e-9], [0.0]]))
        assert 0 <= ra[0] < 2 * np.pi
        assert np.isclose(dec[0], 0.0, atol=1e-8)

    def test_radec_round_trip(self):
        rng = np.random.default_rng(6)
        ra = rng.uniform(0, 2 * np.pi, 40)
        dec = rng.uniform(-np.pi / 2, np.pi / 2, 40)
        xyz = radec2eq((ra, dec))
        ra2, dec2 = eq2radec(xyz)
        np.testing.assert_allclose(ra2, ra, atol=1e-12)
        np.testing.assert_allclose(dec2, dec, atol=1e-12)

    def test_azalt_round_trip(self):
        rng = np.random.default_rng(7)
        az = rng.uniform(0, 2 * np.pi, 40)
        alt = rng.uniform(-np.pi / 2, np.pi / 2, 40)
        xyz = azalt2top((az, alt))
        az2, alt2 = top2azalt(xyz)
        np.testing.assert_allclose(np.mod(az2, 2 * np.pi),
                                   np.mod(az, 2 * np.pi), atol=1e-12)
        np.testing.assert_allclose(alt2, alt, atol=1e-12)

    def test_topocentric_axis_convention(self):
        """z = up, y = north (az 0), x = east (az pi/2)."""
        np.testing.assert_allclose(azalt2top((0.0, 0.0)), [0, 1, 0],
                                   atol=1e-15)
        np.testing.assert_allclose(azalt2top((np.pi / 2, 0.0)), [1, 0, 0],
                                   atol=1e-15)
        np.testing.assert_allclose(azalt2top((0.0, np.pi / 2)), [0, 0, 1],
                                   atol=1e-15)

    def test_latlong2xyz(self):
        np.testing.assert_allclose(latlong2xyz((np.pi / 2, 0.0)), [0, 0, 1],
                                   atol=1e-15)


# ---------------------------------------------------------------------------
# eq2top_m / top2eq_m
# ---------------------------------------------------------------------------

class TestEq2TopM:

    def test_is_a_rotation(self):
        m = eq2top_m(0.4, 0.9)
        np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(m), 1.0)

    def test_zenith_maps_to_up(self):
        """A source at ha=0, dec=latitude sits at topocentric zenith."""
        dec = 0.65
        src = radec2eq((0.0, dec))
        top = eq2top_m(0.0, dec) @ src
        np.testing.assert_allclose(top, [0, 0, 1], atol=1e-12)

    def test_altitude_drops_away_from_transit(self):
        """Altitude is maximal at transit (ha=0) and falls off either side."""
        dec = 0.5
        src = radec2eq((0.0, dec))
        alts = [(eq2top_m(ha, dec) @ src)[2] for ha in (-0.4, -0.2, 0.0,
                                                        0.2, 0.4)]
        assert np.isclose(alts[2], 1.0)
        assert alts[1] > alts[0] and alts[3] > alts[4]
        assert np.isclose(alts[1], alts[3]) and np.isclose(alts[0], alts[4])

    def test_top2eq_m_is_inverse(self):
        m = eq2top_m(0.3, -0.2)
        np.testing.assert_allclose(top2eq_m(0.3, -0.2) @ m, np.eye(3),
                                   atol=1e-12)

    def test_batched(self):
        ha = np.array([0.0, 0.1, 0.2])
        dec = np.full(3, 0.4)
        m = eq2top_m(ha, dec)
        assert m.shape == (3, 3, 3)
        for i in range(3):
            np.testing.assert_allclose(m[i], eq2top_m(ha[i], dec[i]),
                                       atol=1e-15)

    def test_batched_inverse(self):
        ha = np.array([0.0, 0.1])
        dec = np.array([0.4, -0.3])
        m = eq2top_m(ha, dec)
        inv = top2eq_m(ha, dec)
        for i in range(2):
            np.testing.assert_allclose(inv[i] @ m[i], np.eye(3), atol=1e-12)

    def test_jax_backend(self):
        m = eq2top_m(jnp.float64(0.4), jnp.float64(0.9))
        assert isinstance(m, jax.Array)
        np.testing.assert_allclose(np.asarray(m), eq2top_m(0.4, 0.9),
                                   atol=1e-12)


class TestAipyParity:
    """These transforms were extracted from aipy.coord; stay bit-compatible."""

    @pytest.fixture(autouse=True)
    def _aipy(self):
        self.ac = pytest.importorskip("aipy").coord

    def test_rot_m(self):
        rng = np.random.default_rng(11)
        for _ in range(5):
            ang, vec = rng.uniform(0, 6), rng.normal(size=3)
            np.testing.assert_allclose(rot_m(ang, vec),
                                       self.ac.rot_m(ang, vec), atol=1e-14)

    def test_rot_m_batched(self):
        rng = np.random.default_rng(12)
        angs, vecs = rng.uniform(0, 6, 4), rng.normal(size=(4, 3))
        np.testing.assert_allclose(rot_m(angs, vecs),
                                   self.ac.rot_m(angs, vecs), atol=1e-14)

    def test_eq2top_m(self):
        for ha, dec in [(0.0, 0.0), (0.2, 0.5), (-1.1, -0.4)]:
            np.testing.assert_allclose(eq2top_m(ha, dec),
                                       self.ac.eq2top_m(ha, dec), atol=1e-14)

    def test_top2eq_m(self):
        for ha, dec in [(0.2, 0.5), (-1.1, -0.4)]:
            np.testing.assert_allclose(top2eq_m(ha, dec),
                                       self.ac.top2eq_m(ha, dec), atol=1e-12)

    def test_angle_conversions(self):
        rng = np.random.default_rng(13)
        xyz = rng.normal(size=(3, 25))
        xyz /= np.linalg.norm(xyz, axis=0)
        np.testing.assert_allclose(xyz2thphi(xyz), self.ac.xyz2thphi(xyz),
                                   atol=1e-14)
        th_phi = self.ac.xyz2thphi(xyz)
        np.testing.assert_allclose(thphi2xyz(th_phi),
                                   self.ac.thphi2xyz(th_phi), atol=1e-14)
        np.testing.assert_allclose(eq2radec(xyz), self.ac.eq2radec(xyz),
                                   atol=1e-14)
        np.testing.assert_allclose(top2azalt(xyz), self.ac.top2azalt(xyz),
                                   atol=1e-14)
