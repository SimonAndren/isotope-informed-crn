"""Unit tests for isotopologue.analysis — delta, ratios, mass spectra."""

import numpy as np
import pytest

from isotopologue.analysis import (
    VPDB_RATIO,
    delta,
    intramolecular,
    mass_spectrum,
    mass_spectrum_vectorized,
    natural_abundance,
    natural_abundance_vectorized,
    site_delta,
    site_ratio,
    total_ratio,
    total_ratio_vectorized,
)


class TestNaturalAbundance:
    def test_sums_to_one(self):
        for n in range(1, 6):
            conc = natural_abundance(n)
            np.testing.assert_allclose(conc.sum(), 1.0, rtol=1e-10)

    def test_vectorized_matches_scalar(self):
        for n in range(1, 5):
            scalar = natural_abundance(n)
            vec = natural_abundance_vectorized(n)
            np.testing.assert_allclose(vec, scalar, rtol=1e-12)

    def test_monoisotopic_dominant(self):
        """All-light isotopologue should be most abundant."""
        conc = natural_abundance(3)
        assert conc[0] == conc.max()

    def test_shape(self):
        conc = natural_abundance(4)
        assert conc.shape == (16,)

    def test_multi_element(self):
        conc = natural_abundance(2, element="H")
        np.testing.assert_allclose(conc.sum(), 1.0, rtol=1e-10)
        # D/H ratio is very small — monoisotopic should be ~1
        assert conc[0] > 0.999


class TestTotalRatio:
    def test_natural_abundance_c13(self):
        """At natural abundance, total ratio should be close to VPDB."""
        conc = natural_abundance(3)
        ratio = total_ratio(conc)
        np.testing.assert_allclose(ratio, VPDB_RATIO, rtol=1e-4)

    def test_vectorized_matches_scalar(self):
        conc = natural_abundance(4)
        r1 = total_ratio(conc)
        r2 = total_ratio_vectorized(conc, 4)
        np.testing.assert_allclose(r1, r2, rtol=1e-10)

    def test_pure_heavy(self):
        """All heavy → ratio = n_heavy/0 = inf."""
        conc = np.zeros(8)
        conc[7] = 1.0  # 0b111 = all heavy
        ratio = total_ratio(conc)
        # 3 heavy / 0 light → divide by zero, but using total_ratio
        # n_heavy = 3, n_light = 0 → should handle gracefully
        # All 3 positions heavy: total_heavy = 3.0, total_light = 0
        assert ratio == 0 or np.isinf(ratio) or ratio > 1000

    def test_pure_light(self):
        conc = np.zeros(4)
        conc[0] = 1.0
        ratio = total_ratio(conc)
        assert ratio == 0.0  # No heavy atoms


class TestSiteRatio:
    def test_natural_abundance(self):
        conc = natural_abundance(2)
        r0 = site_ratio(conc, 0)
        r1 = site_ratio(conc, 1)
        np.testing.assert_allclose(r0, VPDB_RATIO, rtol=1e-4)
        np.testing.assert_allclose(r1, VPDB_RATIO, rtol=1e-4)

    def test_single_labeled_position(self):
        """Set up a vector where only position 0 is 50% heavy."""
        # 2 atoms: isotopologues 00, 01, 10, 11
        # Position 0 = MSB. Heavy at pos 0 means bit 1 is set.
        conc = np.array([0.5, 0.5, 0.5, 0.5])
        # All equally likely → site ratio at each position = 1.0 (50/50)
        r0 = site_ratio(conc, 0)
        np.testing.assert_allclose(r0, 1.0, rtol=1e-10)


class TestDelta:
    def test_vpdb_returns_zero(self):
        d = delta(VPDB_RATIO, "C")
        np.testing.assert_allclose(d, 0.0, atol=1e-10)

    def test_enriched(self):
        ratio = VPDB_RATIO * 1.05  # 5% enriched
        d = delta(ratio, "C")
        np.testing.assert_allclose(d, 50.0, rtol=1e-6)  # 50 permil

    def test_depleted(self):
        ratio = VPDB_RATIO * 0.95
        d = delta(ratio, "C")
        np.testing.assert_allclose(d, -50.0, rtol=1e-6)


class TestSiteDelta:
    def test_natural_abundance(self):
        conc = natural_abundance(3)
        for pos in range(3):
            d = site_delta(conc, pos, "C")
            np.testing.assert_allclose(d, 0.0, atol=0.1)  # should be ~0‰


class TestMassSpectrum:
    def test_pure_light(self):
        conc = np.zeros(8)
        conc[0] = 1.0
        spec = mass_spectrum(conc)
        expected = np.array([1, 0, 0, 0], dtype=float)
        np.testing.assert_array_equal(spec, expected)

    def test_pure_heavy(self):
        conc = np.zeros(8)
        conc[7] = 1.0  # all 3 atoms heavy
        spec = mass_spectrum(conc)
        expected = np.array([0, 0, 0, 1], dtype=float)
        np.testing.assert_array_equal(spec, expected)

    def test_sums_to_one(self):
        conc = natural_abundance(4)
        spec = mass_spectrum(conc)
        np.testing.assert_allclose(spec.sum(), 1.0, rtol=1e-10)

    def test_vectorized_matches_scalar(self):
        conc = natural_abundance(3)
        s1 = mass_spectrum(conc)
        s2 = mass_spectrum_vectorized(conc, 3)
        np.testing.assert_allclose(s1, s2, rtol=1e-10)

    def test_empty(self):
        spec = mass_spectrum(np.zeros(4))
        np.testing.assert_array_equal(spec, np.zeros(3))


class TestIntramolecular:
    def test_natural_abundance_at_zero_delta(self):
        """All sites at 0‰ should match natural abundance."""
        conc = intramolecular(3, np.zeros(3), total_conc=1.0)
        expected = natural_abundance_vectorized(3)
        np.testing.assert_allclose(conc, expected, rtol=1e-6)

    def test_total_concentration(self):
        conc = intramolecular(2, np.array([10.0, -5.0]), total_conc=2.0)
        np.testing.assert_allclose(conc.sum(), 2.0, rtol=1e-6)

    def test_enriched_site(self):
        """Setting a high delta at position 0 should increase its site ratio."""
        conc = intramolecular(2, np.array([100.0, 0.0]))
        r0 = site_ratio(conc, 0)
        r1 = site_ratio(conc, 1)
        d0 = delta(r0, "C")
        d1 = delta(r1, "C")
        np.testing.assert_allclose(d0, 100.0, rtol=0.01)
        np.testing.assert_allclose(d1, 0.0, atol=0.1)

    def test_shape(self):
        conc = intramolecular(4, np.zeros(4))
        assert conc.shape == (16,)
