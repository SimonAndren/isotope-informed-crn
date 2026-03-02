"""Unit tests for isotopologue.kie — kinetic isotope effect rate vectors."""

import numpy as np
import pytest

from isotopologue.kie import build_rate_vector, reduced_mass_kie


class TestBuildRateVector:
    def test_no_kie(self):
        """Without KIE, all rates equal base_rate."""
        rates = build_rate_vector(3, 0.5)
        assert rates.shape == (8,)
        np.testing.assert_array_equal(rates, np.full(8, 0.5))

    def test_no_kie_explicit_none(self):
        rates = build_rate_vector(2, 1.0, site_kies=None)
        np.testing.assert_array_equal(rates, np.ones(4))

    def test_monoisotopic_rate(self):
        """Index 0 (all light) should always equal base_rate."""
        rates = build_rate_vector(3, 2.0, {0: 0.99, 1: 0.98})
        assert rates[0] == 2.0

    def test_single_substitution(self):
        """Single-substituted isotopologue rate = base * kie."""
        rates = build_rate_vector(2, 1.0, {0: 0.95})
        # Index 0b10 (position 0 heavy) → rate = 1.0 * 0.95
        assert np.isclose(rates[0b10], 0.95)
        # Index 0b01 (position 1 heavy, no KIE specified) → rate = 1.0
        assert np.isclose(rates[0b01], 1.0)

    def test_double_substitution(self):
        """Doubly-substituted: compound KIE normalised by base_rate."""
        rates = build_rate_vector(2, 1.0, {0: 0.95, 1: 0.98})
        # Index 0b11 (both heavy):
        # k = base * kie[0] * kie[1] / base^(2-1) = 1.0 * 0.95 * 0.98 / 1.0
        expected = 0.95 * 0.98
        assert np.isclose(rates[0b11], expected)

    def test_vector_length(self):
        for n in range(1, 6):
            rates = build_rate_vector(n, 1.0)
            assert len(rates) == 2**n

    def test_zero_base_rate(self):
        rates = build_rate_vector(2, 0.0, {0: 0.99})
        np.testing.assert_array_equal(rates, np.zeros(4))

    def test_three_carbon_all_kie(self):
        """3-carbon molecule with KIE on all positions."""
        kies = {0: 0.99, 1: 0.98, 2: 0.97}
        rates = build_rate_vector(3, 1.0, kies)
        # Index 0: base = 1.0
        assert rates[0] == 1.0
        # Index 0b100 (position 0 heavy): 1.0 * 0.99
        assert np.isclose(rates[0b100], 0.99)
        # Index 0b010 (position 1 heavy): 1.0 * 0.98
        assert np.isclose(rates[0b010], 0.98)
        # Index 0b001 (position 2 heavy): 1.0 * 0.97
        assert np.isclose(rates[0b001], 0.97)
        # Index 0b111 (all heavy): 1.0 * 0.99 * 0.98 * 0.97 / 1.0^2
        expected = 0.99 * 0.98 * 0.97
        assert np.isclose(rates[0b111], expected)


class TestReducedMassKIE:
    def test_c13_kie(self):
        """C-12 → C-13 should give a modest KIE."""
        kie = reduced_mass_kie(12.0, 13.0)
        assert 0.95 < kie < 1.0  # normal (inverse) KIE for heavier isotope

    def test_identical_mass(self):
        """Same mass → KIE = 1."""
        kie = reduced_mass_kie(12.0, 12.0)
        assert np.isclose(kie, 1.0)

    def test_two_member_ts(self):
        """Two-member TS uses direct mass ratio."""
        kie = reduced_mass_kie(12.0, 13.0, three_member=False)
        expected = (12.0 / 13.0) ** 0.5
        assert np.isclose(kie, expected)

    def test_deuterium(self):
        """D/H substitution should give a large KIE."""
        kie = reduced_mass_kie(1.00794, 2.01410, three_member=False)
        assert kie < 0.8  # significant effect
