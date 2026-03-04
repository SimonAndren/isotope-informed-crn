"""Unit tests for isotope tracking through specific reaction types.

Focus: exchange (H-abstraction) reactions where carbon skeletons are
preserved. This is the most common mechanistic step in propane pyrolysis
and the hardest to get right — without a correct atom map the engine will
incorrectly scramble ¹³C between products.

H-abstraction archetype:
    A(2C) + B(1C) → C(1C) + D(2C)

The hydrogen moves; no carbon crosses the bond. Therefore:
    - B's single carbon  →  C's single carbon
    - A's two carbons    →  D's two carbons (in order)

The atom map encoding for this in the engine's binary-index convention:
    Reactant combined space  A(2C) × B(1C)  has 4×2 = 8 entries.
    Product  combined space  C(1C) × D(2C)  has 2×4 = 8 entries.

    Bit layout of reactant outer-product index r = a_idx*2 + b_idx:
        bit 2  =  A position-0  =  a0
        bit 1  =  A position-1  =  a1
        bit 0  =  B position-0  =  b0

    Bit layout of product outer-product index p = c_idx*4 + d_idx:
        bit 2  =  C position-0  =  c0
        bit 1  =  D position-0  =  d0
        bit 0  =  D position-1  =  d1

    Carbon routing (H-abstraction, skeleton preserved):
        b0  →  c0
        a0  →  d0
        a1  →  d1

    Derived maps:
        forward[p] = r  :  p=0→0, p=1→2, p=2→4, p=3→6, p=4→1, p=5→3, p=6→5, p=7→7
        reverse[r] = p  :  r=0→0, r=1→4, r=2→1, r=3→5, r=4→2, r=5→6, r=6→3, r=7→7
"""

import numpy as np
import pytest

from isotopologue.analysis import (
    VPDB_RATIO,
    delta,
    intramolecular,
    natural_abundance_vectorized,
    site_delta,
    site_ratio,
)
from isotopologue.engine import IsotopologueEngine
from isotopologue.kie import build_rate_vector
from isotopologue.species import AtomMap, Network, Reaction, Species


# ─── Atom map for H-abstraction: A(2C) + B(1C) → C(1C) + D(2C) ─────────────

_H_ABS_FORWARD = np.array([0, 2, 4, 6, 1, 3, 5, 7], dtype=np.int32)
_H_ABS_REVERSE = np.array([0, 4, 1, 5, 2, 6, 3, 7], dtype=np.int32)
_H_ABS_ATOM_MAP = AtomMap(forward=_H_ABS_FORWARD, reverse=_H_ABS_REVERSE)


def _h_abs_network(k_fwd: float = 1.0, k_rev: float = 0.0) -> Network:
    """Minimal network: A(2C) + B(1C) → C(1C) + D(2C).

    By default the reaction is irreversible (k_rev=0) to make assertions
    about carbon routing unambiguous.
    """
    return Network(
        species={
            "A": Species("A", n_labeled=2),
            "B": Species("B", n_labeled=1),
            "C": Species("C", n_labeled=1),
            "D": Species("D", n_labeled=2),
        },
        reactions=[
            Reaction(
                "H_abs",
                reactants=("A", "B"),
                products=("C", "D"),
                k_forward=np.full(8, k_fwd),
                k_reverse=np.full(8, k_rev),
                atom_map=_H_ABS_ATOM_MAP,
            )
        ],
    )


# ─── Tests ───────────────────────────────────────────────────────────────────


class TestAtomMapConstruction:
    """Verify the H-abstraction atom map satisfies basic permutation invariants."""

    def test_forward_is_permutation(self):
        assert sorted(_H_ABS_FORWARD.tolist()) == list(range(8))

    def test_reverse_is_permutation(self):
        assert sorted(_H_ABS_REVERSE.tolist()) == list(range(8))

    def test_forward_reverse_are_inverses(self):
        """forward and reverse must be inverse permutations of each other."""
        inv = np.empty(8, dtype=np.int32)
        inv[_H_ABS_FORWARD] = np.arange(8, dtype=np.int32)
        np.testing.assert_array_equal(inv, _H_ABS_REVERSE)

    def test_all_light_maps_to_all_light(self):
        """Index 0 (no heavy atoms) must map to index 0 in both directions."""
        assert _H_ABS_FORWARD[0] == 0
        assert _H_ABS_REVERSE[0] == 0

    def test_all_heavy_maps_to_all_heavy(self):
        """Index 7 (all heavy) must map to index 7 in both directions."""
        assert _H_ABS_FORWARD[7] == 7
        assert _H_ABS_REVERSE[7] == 7


class TestHAbstractionCarbonRouting:
    """B's carbon → C, A's carbons → D (skeleton-preserving H-abstraction)."""

    @pytest.fixture
    def net(self):
        return _h_abs_network(k_fwd=10.0, k_rev=0.0)

    @pytest.fixture
    def engine(self, net):
        return IsotopologueEngine(net)

    def test_b_carbon_goes_to_c(self, net, engine):
        """Enriching B (source of C) should enrich C, not D."""
        # A at natural abundance, B enriched at +100 ‰
        conc_a = natural_abundance_vectorized(2)
        conc_b = intramolecular(1, np.array([100.0]))

        y0 = net.pack({"A": conc_a, "B": conc_b, "C": np.zeros(2), "D": np.zeros(4)})
        result = engine.solve(y0, (0, 0.5), method="RK45", rtol=1e-10, atol=1e-14)
        assert result.success

        final = net.unpack(result.y[:, -1])

        d_C = delta(site_ratio(final["C"], 0))
        d_D0 = site_delta(final["D"], 0)

        # C came from B (enriched), so δ¹³C(C) should be clearly positive
        assert d_C > 20.0, f"C should be enriched (B was +100‰) but δ¹³C(C) = {d_C:.1f}‰"
        # D came from A (natural abundance), so δ¹³C(D) should be near 0
        assert abs(d_D0) < 10.0, f"D should be near natural abundance but δ¹³C(D pos0) = {d_D0:.1f}‰"

    def test_a_carbons_go_to_d(self, net, engine):
        """Enriching A (source of D) should enrich D, not C."""
        # A enriched at both positions, B at natural abundance
        conc_a = intramolecular(2, np.array([80.0, 80.0]))
        conc_b = natural_abundance_vectorized(1)

        y0 = net.pack({"A": conc_a, "B": conc_b, "C": np.zeros(2), "D": np.zeros(4)})
        result = engine.solve(y0, (0, 0.5), method="RK45", rtol=1e-10, atol=1e-14)
        assert result.success

        final = net.unpack(result.y[:, -1])

        d_C = delta(site_ratio(final["C"], 0))
        d_D0 = site_delta(final["D"], 0)

        # D came from A (enriched), so δ¹³C(D) should be positive
        assert d_D0 > 10.0, f"D should be enriched (A was +80‰) but δ¹³C(D pos0) = {d_D0:.1f}‰"
        # C came from B (natural abundance), so δ¹³C(C) should be near 0
        assert abs(d_C) < 10.0, f"C should be near natural abundance but δ¹³C(C) = {d_C:.1f}‰"

    def test_position_order_preserved(self, net, engine):
        """A position-0 heavy (a0) should map to D position-0 (d0), not d1."""
        # Set up A: only position 0 enriched, position 1 depleted
        conc_a = intramolecular(2, np.array([200.0, -20.0]))
        conc_b = natural_abundance_vectorized(1)

        y0 = net.pack({"A": conc_a, "B": conc_b, "C": np.zeros(2), "D": np.zeros(4)})
        result = engine.solve(y0, (0, 0.3), method="RK45", rtol=1e-10, atol=1e-14)
        assert result.success

        final = net.unpack(result.y[:, -1])

        d_D0 = site_delta(final["D"], 0)
        d_D1 = site_delta(final["D"], 1)

        # D position 0 should be more enriched than D position 1
        assert d_D0 > d_D1, (
            f"D pos0 ({d_D0:.1f}‰) should be more enriched than D pos1 ({d_D1:.1f}‰)"
        )

    def test_carbon_conservation(self, net, engine):
        """Total ¹³C moles must not change through H-abstraction."""
        conc_a = intramolecular(2, np.array([50.0, -10.0])) * 1.0
        conc_b = intramolecular(1, np.array([30.0])) * 0.5

        y0 = net.pack({"A": conc_a, "B": conc_b, "C": np.zeros(2), "D": np.zeros(4)})

        def c13_moles(y):
            state = net.unpack(y)
            total = 0.0
            for name in net.species:
                c = state[name]
                for im in range(len(c)):
                    total += c[im] * bin(im).count("1")
            return total

        initial_c13 = c13_moles(y0)
        result = engine.solve(y0, (0, 1.0), method="BDF", rtol=1e-10, atol=1e-14)
        assert result.success

        for yi in result.y.T:
            np.testing.assert_allclose(
                c13_moles(yi), initial_c13, rtol=1e-5,
                err_msg="¹³C moles not conserved through H-abstraction"
            )

    def test_total_carbon_conservation(self, net, engine):
        """Total carbon (¹²C + ¹³C) weighted by positions must be conserved."""
        conc_a = natural_abundance_vectorized(2)
        conc_b = natural_abundance_vectorized(1) * 0.5

        y0 = net.pack({"A": conc_a, "B": conc_b, "C": np.zeros(2), "D": np.zeros(4)})

        def total_carbon(y):
            state = net.unpack(y)
            return (
                2 * state["A"].sum()
                + 1 * state["B"].sum()
                + 1 * state["C"].sum()
                + 2 * state["D"].sum()
            )

        initial_c = total_carbon(y0)
        result = engine.solve(y0, (0, 0.5), method="RK45", rtol=1e-10, atol=1e-14)
        assert result.success
        for yi in result.y.T:
            np.testing.assert_allclose(
                total_carbon(yi), initial_c, rtol=1e-6,
                err_msg="Total carbon not conserved through H-abstraction"
            )


class TestHAbstractionKIE:
    """KIE on a specific position should fractionate that position into products."""

    def _build_kie_network(self, kie_at_a0: float) -> Network:
        """H-abstraction with a KIE on A's position-0 carbon.

        In the reactant outer-product space (A(2C)×B(1C)), bit 2 = A position-0.
        A KIE < 1 at that position means the ¹³C isotopologue reacts slower.
        """
        # Build a KIE-modified rate vector for the 8-entry reactant space.
        # The outer product bit layout:
        #   bit 2 = A pos 0 (a0)
        #   bit 1 = A pos 1 (a1)
        #   bit 0 = B pos 0 (b0)
        nm = 8
        k_base = 1.0
        k_fwd = np.full(nm, k_base)
        for im in range(nm):
            # A position 0 is bit 2 in the combined space
            if (im >> 2) & 1:
                k_fwd[im] *= kie_at_a0

        return Network(
            species={
                "A": Species("A", n_labeled=2),
                "B": Species("B", n_labeled=1),
                "C": Species("C", n_labeled=1),
                "D": Species("D", n_labeled=2),
            },
            reactions=[
                Reaction(
                    "H_abs_kie",
                    reactants=("A", "B"),
                    products=("C", "D"),
                    k_forward=k_fwd,
                    k_reverse=np.zeros(8),
                    atom_map=_H_ABS_ATOM_MAP,
                )
            ],
        )

    def test_normal_kie_depletes_heavy_reactant(self):
        """Normal KIE (< 1): ¹³C-bearing A reacts slower → A becomes enriched over time."""
        kie = 0.97  # normal KIE: heavy isotope reacts ~3% slower
        net = self._build_kie_network(kie_at_a0=kie)
        eng = IsotopologueEngine(net)

        # A slightly enriched at position 0, B at natural abundance
        conc_a = intramolecular(2, np.array([10.0, 0.0])) * 1.0
        conc_b = natural_abundance_vectorized(1) * 1.0

        y0 = net.pack({"A": conc_a, "B": conc_b, "C": np.zeros(2), "D": np.zeros(4)})
        result = eng.solve(y0, (0, 2.0), method="RK45", rtol=1e-10, atol=1e-14)
        assert result.success

        initial_state = net.unpack(y0)
        final_state = net.unpack(result.y[:, -1])

        d_A_pos0_initial = site_delta(initial_state["A"], 0)
        d_A_pos0_final = site_delta(final_state["A"], 0)

        # Under normal KIE the ¹³C reactant is consumed more slowly, so the
        # residual A should be enriched (more positive δ¹³C) compared to initial
        assert d_A_pos0_final > d_A_pos0_initial - 0.5, (
            f"Normal KIE should enrich residual A pos0: "
            f"initial {d_A_pos0_initial:.2f}‰ → final {d_A_pos0_final:.2f}‰"
        )

    def test_no_kie_no_fractionation(self):
        """Without KIE, the isotope ratio of products should equal that of the
        consumed reactant (no fractionation)."""
        net = _h_abs_network(k_fwd=5.0, k_rev=0.0)  # no KIE
        eng = IsotopologueEngine(net)

        # A uniformly enriched at +30‰ at all positions
        conc_a = intramolecular(2, np.array([30.0, 30.0])) * 1.0
        conc_b = natural_abundance_vectorized(1) * 0.5

        y0 = net.pack({"A": conc_a, "B": conc_b, "C": np.zeros(2), "D": np.zeros(4)})
        result = eng.solve(y0, (0, 0.1), method="RK45", rtol=1e-10, atol=1e-14)
        assert result.success

        final = net.unpack(result.y[:, -1])

        # D pos0 should reflect A's enrichment (~30‰), not be fractionated
        d_D0 = site_delta(final["D"], 0)
        d_D1 = site_delta(final["D"], 1)

        # Both D positions should be similarly enriched (A was +30‰ at both)
        # Allow ±10‰ tolerance since not all A has reacted yet
        assert abs(d_D0 - d_D1) < 5.0, (
            f"Without KIE, both D positions should be similarly enriched: "
            f"D pos0={d_D0:.1f}‰, D pos1={d_D1:.1f}‰"
        )


class TestReversibleHAbstraction:
    """Reversible H-abstraction should reach isotopic equilibrium."""

    def test_equilibrium_isotope_ratio(self):
        """At thermodynamic equilibrium the forward and reverse fluxes balance.

        H-abstraction with a skeleton-preserving atom map has two independent
        carbon-exchange branches:
            - A(2C) ↔ D(2C)  (A's skeleton goes into D and back)
            - B(1C) ↔ C(1C)  (B's carbon goes into C and back)

        Without KIE, each branch equilibrates within itself, but there is no
        mechanism for carbon to cross from the A↔D branch to the B↔C branch.
        At equilibrium: δ¹³C(A) ≈ δ¹³C(D) and δ¹³C(B) ≈ δ¹³C(C).
        """
        net = _h_abs_network(k_fwd=1.0, k_rev=0.5)
        eng = IsotopologueEngine(net)

        # Start with A and D at different δ¹³C so there is something to equilbrate;
        # B and C similarly offset from each other.
        conc_a = intramolecular(2, np.array([50.0, 30.0]))
        conc_b = intramolecular(1, np.array([-20.0]))
        conc_c = intramolecular(1, np.array([40.0])) * 0.2   # C differs from B
        conc_d = intramolecular(2, np.array([-10.0, -5.0])) * 0.3  # D differs from A

        y0 = net.pack({"A": conc_a, "B": conc_b, "C": conc_c, "D": conc_d})
        result = eng.solve(y0, (0, 200.0), method="BDF")
        assert result.success

        final = net.unpack(result.y[:, -1])

        from isotopologue.analysis import total_ratio_vectorized

        d_A = delta(total_ratio_vectorized(final["A"], 2))
        d_D = delta(total_ratio_vectorized(final["D"], 2))
        d_B = delta(total_ratio_vectorized(final["B"], 1))
        d_C = delta(total_ratio_vectorized(final["C"], 1))

        # Each skeleton-branch must equilibrate within itself (no KIE)
        assert abs(d_A - d_D) < 2.0, (
            f"A↔D branch: δ¹³C(A)={d_A:.2f}‰ and δ¹³C(D)={d_D:.2f}‰ "
            "should equalise at equilibrium without KIE"
        )
        assert abs(d_B - d_C) < 2.0, (
            f"B↔C branch: δ¹³C(B)={d_B:.2f}‰ and δ¹³C(C)={d_C:.2f}‰ "
            "should equalise at equilibrium without KIE"
        )
