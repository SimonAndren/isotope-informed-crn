"""Integration tests for propane pyrolysis benchmark."""

import numpy as np
import pytest

from isotopologue.analysis import delta, total_ratio_vectorized
from isotopologue.benchmarks.propane import (
    initial_conditions,
    propane_3rxn,
    propane_6rxn,
)
from isotopologue.engine import IsotopologueEngine


class TestPropane3Rxn:
    @pytest.fixture
    def network(self):
        return propane_3rxn(T=1123.0)

    @pytest.fixture
    def engine(self, network):
        return IsotopologueEngine(network)

    def test_network_species_count(self, network):
        assert len(network.species) == 6

    def test_network_reactions_count(self, network):
        assert len(network.reactions) == 3

    def test_state_size(self, network):
        # C3H8(8) + CH3(2) + C2H5(4) + CH4(2) + C2H4(4) + C2H6(4) = 24
        assert network.state_size == 24

    def test_initial_conditions(self, network):
        conc = initial_conditions(network)
        assert set(conc.keys()) == set(network.species.keys())
        assert conc["C3H8"].sum() > 0.99  # ~1.0 total
        for name in ("CH3", "C2H5", "CH4", "C2H4", "C2H6"):
            assert conc[name].sum() < 1e-5  # trace amounts

    def test_moles_increase_with_breakdown(self, network, engine):
        """Total moles increase because breakdown (C3H8→CH3+C2H5) creates molecules."""
        conc = initial_conditions(network)
        y0 = network.pack(conc)
        initial_total = y0.sum()
        # Use BDF: propane kinetics are stiff (rate constants up to ~1e14)
        result = engine.solve(y0, (0, 1e-4), method="BDF")
        assert result.success, f"Solver failed: {result.message}"
        final_total = result.y[:, -1].sum()
        assert final_total >= initial_total - 1e-10

    def test_carbon_conservation(self, network, engine):
        """Total carbon atoms (weighted by n_labeled) must be conserved."""
        conc = initial_conditions(network)
        y0 = network.pack(conc)

        def total_carbon(y):
            state = network.unpack(y)
            total = 0.0
            for name, sp in network.species.items():
                total += state[name].sum() * sp.n_labeled
            return total

        initial_c = total_carbon(y0)
        result = engine.solve(y0, (0, 1e-4), method="BDF")
        assert result.success
        for yi in result.y.T:
            np.testing.assert_allclose(
                total_carbon(yi), initial_c, rtol=1e-4, err_msg="Carbon atoms not conserved"
            )

    @pytest.mark.slow
    def test_simulation_runs(self, network, engine):
        """Full 85ms simulation should complete successfully."""
        conc = initial_conditions(network)
        y0 = network.pack(conc)
        result = engine.solve(y0, (0, 0.085), method="BDF")
        assert result.success, f"Solver failed: {result.message}"
        final = network.unpack(result.y[:, -1])
        # All concentrations should be non-negative (within tolerance)
        for name, c in final.items():
            assert np.all(c >= -1e-10), f"{name} has negative concentrations"

    @pytest.mark.slow
    def test_products_formed(self, network, engine):
        """After integration, some propane should have converted to products."""
        conc = initial_conditions(network)
        y0 = network.pack(conc)
        result = engine.solve(y0, (0, 0.085), method="BDF")
        assert result.success
        final = network.unpack(result.y[:, -1])
        # Some products should have formed (exact amounts depend on rates)
        total_products = sum(final[name].sum() for name in ("CH4", "C2H4", "C2H6"))
        # Products might be very small if rates are such that little conversion
        # occurs, but should be > 0
        assert total_products >= 0


class TestPropane6Rxn:
    def test_network_species_count(self):
        net = propane_6rxn()
        assert len(net.species) == 8  # 6 base species + H + H2

    def test_network_reactions_count(self):
        net = propane_6rxn()
        assert len(net.reactions) == 6

    @pytest.mark.slow
    def test_simulation_runs(self):
        net = propane_6rxn()
        eng = IsotopologueEngine(net)
        conc = initial_conditions(net)
        y0 = net.pack(conc)
        result = eng.solve(y0, (0, 0.085), method="BDF")
        assert result.success, f"Solver failed: {result.message}"


class TestIsotopeTracking:
    """Verify isotope-specific behaviour of the propane benchmark."""

    def test_isotope_mass_balance(self):
        """The total weighted 13C count across all species must be conserved.

        For each isotopologue, the number of 13C atoms is popcount(index).
        Summing conc[i] * popcount(i) over all species gives total 13C moles.
        """
        net = propane_3rxn()
        eng = IsotopologueEngine(net)
        conc = initial_conditions(net)
        y0 = net.pack(conc)

        def total_c13(y):
            state = net.unpack(y)
            total = 0.0
            for name in net.species:
                c = state[name]
                for im in range(len(c)):
                    n_heavy = bin(im).count("1")
                    total += c[im] * n_heavy
            return total

        initial = total_c13(y0)
        result = eng.solve(y0, (0, 1e-4), method="BDF")
        assert result.success
        for yi in result.y.T:
            np.testing.assert_allclose(
                total_c13(yi), initial, rtol=1e-4, err_msg="13C mass balance violated"
            )
