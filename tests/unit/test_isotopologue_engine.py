"""Unit tests for isotopologue.engine — ODE RHS and solver."""

import tracemalloc

import numpy as np
import pytest

from isotopologue.analysis import natural_abundance_vectorized, total_ratio_vectorized
from isotopologue.engine import IsotopologueEngine
from isotopologue.species import AtomMap, Network, Reaction, Species


def _make_simple_network():
    """A → C, both 2-carbon species. No KIE (uniform rates)."""
    net = Network(
        species={
            "A": Species("A", 2),
            "C": Species("C", 2),
        },
        reactions=[
            Reaction(
                "R1",
                ("A",),
                ("C",),
                k_forward=np.full(4, 0.1),
                k_reverse=np.full(4, 0.05),
            ),
        ],
    )
    return net


def _make_synthesis_network():
    """A(1C) + B(1C) → C(2C)."""
    net = Network(
        species={
            "A": Species("A", 1),
            "B": Species("B", 1),
            "C": Species("C", 2),
        },
        reactions=[
            Reaction(
                "R1",
                ("A", "B"),
                ("C",),
                k_forward=np.full(4, 1.0),  # 2^(1+1)
                k_reverse=np.full(4, 0.5),
            ),
        ],
    )
    return net


def _make_breakdown_network():
    """A(2C) → C(1C) + D(1C)."""
    net = Network(
        species={
            "A": Species("A", 2),
            "C": Species("C", 1),
            "D": Species("D", 1),
        },
        reactions=[
            Reaction(
                "R1",
                ("A",),
                ("C", "D"),
                k_forward=np.full(4, 0.2),  # 2^2 for A
                k_reverse=np.full(4, 0.1),  # 2^(1+1) for C+D
            ),
        ],
    )
    return net


def _make_exchange_network():
    """A(1C) + B(1C) → C(1C) + D(1C)."""
    net = Network(
        species={
            "A": Species("A", 1),
            "B": Species("B", 1),
            "C": Species("C", 1),
            "D": Species("D", 1),
        },
        reactions=[
            Reaction(
                "R1",
                ("A", "B"),
                ("C", "D"),
                k_forward=np.full(4, 0.3),
                k_reverse=np.full(4, 0.15),
            ),
        ],
    )
    return net


class TestSimpleReaction:
    def test_rhs_shape(self):
        net = _make_simple_network()
        eng = IsotopologueEngine(net)
        y0 = np.zeros(net.state_size)
        dydt = eng.rhs(0.0, y0)
        assert dydt.shape == y0.shape

    def test_mass_conservation(self):
        """Total concentration of all isotopologues must be conserved."""
        net = _make_simple_network()
        eng = IsotopologueEngine(net)
        conc = {
            "A": natural_abundance_vectorized(2) * 1.0,
            "C": np.zeros(4),
        }
        y0 = net.pack(conc)
        result = eng.solve(y0, (0, 10.0), method="RK45", rtol=1e-10, atol=1e-14)
        assert result.success
        for yi in result.y.T:
            np.testing.assert_allclose(yi.sum(), y0.sum(), rtol=1e-8)

    def test_equilibrium(self):
        """System should reach equilibrium where forward = reverse flux."""
        net = _make_simple_network()
        eng = IsotopologueEngine(net)
        conc = {
            "A": natural_abundance_vectorized(2) * 1.0,
            "C": np.zeros(4),
        }
        y0 = net.pack(conc)
        result = eng.solve(y0, (0, 500.0), method="BDF")
        assert result.success
        final = net.unpack(result.y[:, -1])
        # At equilibrium: k_f * [A] = k_r * [C] for each isotopologue
        # k_f/k_r = 0.1/0.05 = 2, so [C]/[A] = 2
        ratio = final["C"].sum() / final["A"].sum()
        np.testing.assert_allclose(ratio, 2.0, rtol=0.01)

    def test_zero_concentrations(self):
        """Starting with zero should stay at zero."""
        net = _make_simple_network()
        eng = IsotopologueEngine(net)
        y0 = np.zeros(net.state_size)
        dydt = eng.rhs(0.0, y0)
        np.testing.assert_array_equal(dydt, np.zeros(net.state_size))


class TestSynthesisReaction:
    def test_carbon_conservation(self):
        """Carbon atoms conserved: 1*sum(A) + 1*sum(B) + 2*sum(C) = const.

        Total moles are NOT conserved in synthesis (2 mols → 1 mol).
        """
        net = _make_synthesis_network()
        eng = IsotopologueEngine(net)
        conc = {
            "A": natural_abundance_vectorized(1) * 0.5,
            "B": natural_abundance_vectorized(1) * 0.5,
            "C": np.zeros(4),
        }
        y0 = net.pack(conc)
        initial_carbon = 1 * 0.5 + 1 * 0.5 + 2 * 0.0  # = 1.0
        result = eng.solve(y0, (0, 10.0), method="RK45", rtol=1e-10, atol=1e-14)
        assert result.success
        for yi in result.y.T:
            state = net.unpack(yi)
            carbon = 1 * state["A"].sum() + 1 * state["B"].sum() + 2 * state["C"].sum()
            np.testing.assert_allclose(carbon, initial_carbon, rtol=1e-6)

    def test_product_formed(self):
        """Product C should be created over time."""
        net = _make_synthesis_network()
        eng = IsotopologueEngine(net)
        conc = {
            "A": natural_abundance_vectorized(1) * 1.0,
            "B": natural_abundance_vectorized(1) * 1.0,
            "C": np.zeros(4),
        }
        y0 = net.pack(conc)
        result = eng.solve(y0, (0, 5.0), method="RK45", rtol=1e-10, atol=1e-14)
        assert result.success
        final = net.unpack(result.y[:, -1])
        assert final["C"].sum() > 0


class TestBreakdownReaction:
    def test_carbon_conservation(self):
        """Carbon atoms conserved: 2*sum(A) + 1*sum(C) + 1*sum(D) = const.

        Total moles are NOT conserved in breakdown (1 mol → 2 mols).
        """
        net = _make_breakdown_network()
        eng = IsotopologueEngine(net)
        conc = {
            "A": natural_abundance_vectorized(2) * 1.0,
            "C": np.zeros(2),
            "D": np.zeros(2),
        }
        y0 = net.pack(conc)
        initial_carbon = 2 * 1.0  # A has 2 carbons
        result = eng.solve(y0, (0, 10.0), method="RK45", rtol=1e-10, atol=1e-14)
        assert result.success
        for yi in result.y.T:
            state = net.unpack(yi)
            carbon = 2 * state["A"].sum() + 1 * state["C"].sum() + 1 * state["D"].sum()
            np.testing.assert_allclose(carbon, initial_carbon, rtol=1e-6)

    def test_products_formed(self):
        net = _make_breakdown_network()
        eng = IsotopologueEngine(net)
        conc = {
            "A": natural_abundance_vectorized(2) * 1.0,
            "C": np.zeros(2),
            "D": np.zeros(2),
        }
        y0 = net.pack(conc)
        result = eng.solve(y0, (0, 5.0), method="RK45", rtol=1e-10, atol=1e-14)
        assert result.success
        final = net.unpack(result.y[:, -1])
        assert final["C"].sum() > 0
        assert final["D"].sum() > 0


class TestExchangeReaction:
    def test_mass_conservation(self):
        net = _make_exchange_network()
        eng = IsotopologueEngine(net)
        conc = {
            "A": natural_abundance_vectorized(1) * 1.0,
            "B": natural_abundance_vectorized(1) * 0.5,
            "C": np.zeros(2),
            "D": np.zeros(2),
        }
        y0 = net.pack(conc)
        result = eng.solve(y0, (0, 10.0), method="RK45", rtol=1e-10, atol=1e-14)
        assert result.success
        for yi in result.y.T:
            np.testing.assert_allclose(yi.sum(), y0.sum(), rtol=1e-6)

    def test_products_formed(self):
        net = _make_exchange_network()
        eng = IsotopologueEngine(net)
        conc = {
            "A": natural_abundance_vectorized(1) * 1.0,
            "B": natural_abundance_vectorized(1) * 0.5,
            "C": np.zeros(2),
            "D": np.zeros(2),
        }
        y0 = net.pack(conc)
        result = eng.solve(y0, (0, 5.0), method="RK45", rtol=1e-10, atol=1e-14)
        assert result.success
        final = net.unpack(result.y[:, -1])
        assert final["C"].sum() > 0
        assert final["D"].sum() > 0


class TestSolverOptions:
    def test_bdf_solver(self):
        net = _make_simple_network()
        eng = IsotopologueEngine(net)
        y0 = net.pack({"A": np.array([0.9, 0.03, 0.03, 0.04]), "C": np.zeros(4)})
        result = eng.solve(y0, (0, 10.0), method="BDF")
        assert result.success

    def test_radau_solver(self):
        net = _make_simple_network()
        eng = IsotopologueEngine(net)
        y0 = net.pack({"A": np.array([0.9, 0.03, 0.03, 0.04]), "C": np.zeros(4)})
        result = eng.solve(y0, (0, 10.0), method="Radau")
        assert result.success

    def test_t_eval(self):
        net = _make_simple_network()
        eng = IsotopologueEngine(net)
        y0 = net.pack({"A": natural_abundance_vectorized(2), "C": np.zeros(4)})
        t_eval = np.linspace(0, 10, 50)
        result = eng.solve(y0, (0, 10.0), t_eval=t_eval)
        assert result.success
        assert len(result.t) == 50


class TestRhsAllocations:
    """RHS must not allocate heap memory after the first call (warm path)."""

    def _run_rhs_check(self, net, y0):
        eng = IsotopologueEngine(net)
        # Warm up: first call may trigger internal allocations
        for _ in range(5):
            eng.rhs(0.0, y0)

        # Measure allocations on the warm path
        tracemalloc.start()
        for _ in range(100):
            eng.rhs(0.0, y0)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Allow at most 8 KB of peak allocation for 100 calls (< 80 bytes/call)
        # This catches accidental per-call list/array allocations.
        assert peak < 8 * 1024, f"RHS allocates too much: {peak} bytes over 100 calls"

    def test_simple_no_alloc(self):
        net = _make_simple_network()
        y0 = net.pack({"A": np.array([0.9, 0.03, 0.03, 0.04]), "C": np.zeros(4)})
        self._run_rhs_check(net, y0)

    def test_synthesis_no_alloc(self):
        # A(1C=2), B(1C=2) → C(2C=4)
        net = _make_synthesis_network()
        y0 = net.pack({
            "A": np.array([0.95, 0.05]),
            "B": np.array([0.95, 0.05]),
            "C": np.zeros(4),
        })
        self._run_rhs_check(net, y0)

    def test_exchange_no_alloc(self):
        # A(1C=2), B(1C=2), C(1C=2), D(1C=2)
        net = _make_exchange_network()
        y0 = net.pack({
            "A": np.array([0.95, 0.05]),
            "B": np.array([0.95, 0.05]),
            "C": np.zeros(2),
            "D": np.zeros(2),
        })
        self._run_rhs_check(net, y0)
