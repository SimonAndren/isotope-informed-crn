"""Unit tests for the RMG-Py → vector engine bridge.

Tests run without RMG-Py by mocking lightweight stand-ins for the key
RMG objects (Species, Molecule, Atom, Reaction, Arrhenius). This keeps
the unit tests fast and CI-friendly.

Integration tests that require an actual RMG-Py environment are in
tests/integration/test_rmg_vs_vector.py.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from isotopologue.rmg_bridge import (
    HAS_RMG,
    _n_combined,
    _reaction_type,
    rmg_network_to_vector,
    rmg_reaction_to_vector,
    rmg_species_to_vector,
)
from isotopologue.species import Network, Reaction, Species


# ─── Lightweight RMG stand-ins ────────────────────────────────────────────────

def _fake_atom(symbol: str):
    a = SimpleNamespace(symbol=symbol)
    return a


def _fake_molecule(symbols: list[str]):
    m = SimpleNamespace(atoms=[_fake_atom(s) for s in symbols])
    return m


def _fake_rmg_species(label: str, symbols: list[str]):
    """Minimal RMG Species stand-in."""
    mol = _fake_molecule(symbols)
    return SimpleNamespace(label=label, molecule=[mol])


def _fake_kinetics(kf: float = 1.0):
    k = MagicMock()
    k.get_rate_coefficient.return_value = kf
    return k


def _fake_rmg_reaction(
    label: str,
    reactant_specs,
    product_specs,
    kf: float = 1.0,
    Keq: float = 10.0,
):
    rxn = SimpleNamespace(
        label=label,
        reactants=reactant_specs,
        products=product_specs,
        kinetics=_fake_kinetics(kf),
        pairs=None,
    )
    rxn.get_equilibrium_constant = MagicMock(return_value=Keq)
    return rxn


# ─── Species conversion tests ─────────────────────────────────────────────────

class TestRmgSpeciesToVector:
    def test_propane_n_labeled(self):
        rmg_sp = _fake_rmg_species("C3H8", ["C", "C", "C", "H", "H", "H", "H", "H", "H", "H", "H"])
        sp = rmg_species_to_vector(rmg_sp)
        assert sp.n_labeled == 3

    def test_methyl_n_labeled(self):
        rmg_sp = _fake_rmg_species("CH3", ["C", "H", "H", "H"])
        sp = rmg_species_to_vector(rmg_sp)
        assert sp.n_labeled == 1

    def test_ethylene_n_labeled(self):
        rmg_sp = _fake_rmg_species("C2H4", ["C", "C", "H", "H", "H", "H"])
        sp = rmg_species_to_vector(rmg_sp)
        assert sp.n_labeled == 2

    def test_hydrogen_no_carbon(self):
        rmg_sp = _fake_rmg_species("H2", ["H", "H"])
        sp = rmg_species_to_vector(rmg_sp)
        assert sp.n_labeled == 0

    def test_label_preserved(self):
        rmg_sp = _fake_rmg_species("MyMolecule", ["C", "C", "H"])
        sp = rmg_species_to_vector(rmg_sp)
        assert sp.name == "MyMolecule"

    def test_element_tracking(self):
        # Track nitrogen instead of carbon
        rmg_sp = _fake_rmg_species("NH3", ["N", "H", "H", "H"])
        sp = rmg_species_to_vector(rmg_sp, element="N")
        assert sp.n_labeled == 1
        assert sp.element == "N"

    def test_n_isotopologues(self):
        rmg_sp = _fake_rmg_species("C3H8", ["C", "C", "C", "H", "H", "H", "H", "H", "H", "H", "H"])
        sp = rmg_species_to_vector(rmg_sp)
        assert sp.n_isotopologues == 8  # 2^3


# ─── Reaction type tests ──────────────────────────────────────────────────────

class TestReactionType:
    def test_simple(self):
        assert _reaction_type(1, 1) == "simple"

    def test_breakdown(self):
        assert _reaction_type(1, 2) == "breakdown"

    def test_synthesis(self):
        assert _reaction_type(2, 1) == "synthesis"

    def test_exchange(self):
        assert _reaction_type(2, 2) == "exchange"

    def test_unsupported_raises(self):
        with pytest.raises(ValueError):
            _reaction_type(3, 1)


# ─── Rate vector length tests ─────────────────────────────────────────────────

class TestNCombined:
    def test_synthesis_length(self):
        # CH3(2) + C2H5(4) → C3H8(8): forward space = 2*4=8
        species_map = {
            "CH3": Species("CH3", n_labeled=1),
            "C2H5": Species("C2H5", n_labeled=2),
        }
        n = _n_combined(("CH3", "C2H5"), species_map)
        assert n == 8  # 2^1 * 2^2

    def test_simple_length(self):
        species_map = {"C2H5": Species("C2H5", n_labeled=2)}
        n = _n_combined(("C2H5",), species_map)
        assert n == 4  # 2^2

    def test_exchange_reactant_space(self):
        species_map = {
            "CH3": Species("CH3", n_labeled=1),
            "C2H5": Species("C2H5", n_labeled=2),
        }
        n = _n_combined(("CH3", "C2H5"), species_map)
        assert n == 8


# ─── Full reaction conversion tests ──────────────────────────────────────────

def _make_propane_species_map():
    return {
        "C3H8": Species("C3H8", n_labeled=3),
        "CH3": Species("CH3", n_labeled=1),
        "C2H5": Species("C2H5", n_labeled=2),
        "CH4": Species("CH4", n_labeled=1),
        "C2H4": Species("C2H4", n_labeled=2),
        "C2H6": Species("C2H6", n_labeled=2),
    }


class TestRmgReactionToVector:
    def setup_method(self):
        self.smap = _make_propane_species_map()

    def _make_spc(self, label):
        """Return a fake RMG Species with the right label."""
        return _fake_rmg_species(label, ["C"])  # content doesn't matter for bridge

    def test_synthesis_type(self):
        ch3 = self._make_spc("CH3")
        c2h5 = self._make_spc("C2H5")
        c3h8 = self._make_spc("C3H8")
        rxn = _fake_rmg_reaction("R1", [ch3, c2h5], [c3h8], kf=1e7)
        vec_rxn = rmg_reaction_to_vector(rxn, self.smap, T=1123.0)
        assert vec_rxn.reaction_type == "synthesis"

    def test_exchange_type(self):
        ch3 = self._make_spc("CH3")
        c2h5 = self._make_spc("C2H5")
        ch4 = self._make_spc("CH4")
        c2h4 = self._make_spc("C2H4")
        rxn = _fake_rmg_reaction("R2", [ch3, c2h5], [ch4, c2h4], kf=1e4)
        vec_rxn = rmg_reaction_to_vector(rxn, self.smap, T=1123.0)
        assert vec_rxn.reaction_type == "exchange"

    def test_k_forward_vector_length_synthesis(self):
        # CH3(2) + C2H5(4) → C3H8(8): k_forward length = 8
        ch3 = self._make_spc("CH3")
        c2h5 = self._make_spc("C2H5")
        c3h8 = self._make_spc("C3H8")
        rxn = _fake_rmg_reaction("R1", [ch3, c2h5], [c3h8], kf=1e7)
        vec_rxn = rmg_reaction_to_vector(rxn, self.smap, T=1123.0)
        assert len(vec_rxn.k_forward) == 8  # 2^(1+2)

    def test_k_reverse_vector_length_synthesis(self):
        # C3H8(8): k_reverse length = 8
        ch3 = self._make_spc("CH3")
        c2h5 = self._make_spc("C2H5")
        c3h8 = self._make_spc("C3H8")
        rxn = _fake_rmg_reaction("R1", [ch3, c2h5], [c3h8], kf=1e7, Keq=1e10)
        vec_rxn = rmg_reaction_to_vector(rxn, self.smap, T=1123.0)
        assert len(vec_rxn.k_reverse) == 8  # 2^3

    def test_kf_value_uniform(self):
        ch3 = self._make_spc("CH3")
        c2h5 = self._make_spc("C2H5")
        c3h8 = self._make_spc("C3H8")
        kf_expected = 1.23e7
        rxn = _fake_rmg_reaction("R1", [ch3, c2h5], [c3h8], kf=kf_expected, Keq=1e10)
        vec_rxn = rmg_reaction_to_vector(rxn, self.smap, T=1123.0)
        np.testing.assert_allclose(vec_rxn.k_forward, kf_expected)

    def test_kr_value_from_keq(self):
        ch3 = self._make_spc("CH3")
        c2h5 = self._make_spc("C2H5")
        c3h8 = self._make_spc("C3H8")
        kf, Keq = 1e7, 1e5
        rxn = _fake_rmg_reaction("R1", [ch3, c2h5], [c3h8], kf=kf, Keq=Keq)
        vec_rxn = rmg_reaction_to_vector(rxn, self.smap, T=1123.0)
        np.testing.assert_allclose(vec_rxn.k_reverse, kf / Keq)

    def test_label_preserved(self):
        ch3 = self._make_spc("CH3")
        c2h5 = self._make_spc("C2H5")
        c3h8 = self._make_spc("C3H8")
        rxn = _fake_rmg_reaction("MyRxn", [ch3, c2h5], [c3h8])
        vec_rxn = rmg_reaction_to_vector(rxn, self.smap, T=1123.0)
        assert vec_rxn.name == "MyRxn"


# ─── Full network conversion tests ────────────────────────────────────────────

class TestRmgNetworkToVector:
    def _make_full_propane_3rxn(self):
        # Fake RMG species
        spc = {
            name: _fake_rmg_species(name, ["C"] * nc)
            for name, nc in [
                ("C3H8", 3), ("CH3", 1), ("C2H5", 2),
                ("CH4", 1), ("C2H4", 2), ("C2H6", 2),
            ]
        }
        rxns = [
            _fake_rmg_reaction("R1", [spc["CH3"], spc["C2H5"]], [spc["C3H8"]]),
            _fake_rmg_reaction("R2", [spc["CH3"], spc["C2H5"]], [spc["CH4"], spc["C2H4"]]),
            _fake_rmg_reaction("R3", [spc["CH3"], spc["CH3"]], [spc["C2H6"]]),
        ]
        return list(spc.values()), rxns

    def test_species_count(self):
        species, rxns = self._make_full_propane_3rxn()
        net = rmg_network_to_vector(species, rxns, T=1123.0)
        assert len(net.species) == 6

    def test_reactions_count(self):
        species, rxns = self._make_full_propane_3rxn()
        net = rmg_network_to_vector(species, rxns, T=1123.0)
        assert len(net.reactions) == 3

    def test_state_size_matches_propane(self):
        # Should match propane_3rxn().state_size = 24
        from isotopologue.benchmarks.propane import propane_3rxn
        ref_net = propane_3rxn()

        species, rxns = self._make_full_propane_3rxn()
        net = rmg_network_to_vector(species, rxns, T=1123.0)

        assert net.state_size == ref_net.state_size

    def test_pack_unpack_roundtrip(self):
        species, rxns = self._make_full_propane_3rxn()
        net = rmg_network_to_vector(species, rxns, T=1123.0)

        rng = np.random.default_rng(42)
        original = {name: rng.random(sp.n_isotopologues) for name, sp in net.species.items()}
        y = net.pack(original)
        recovered = net.unpack(y)

        for name in original:
            np.testing.assert_array_equal(recovered[name], original[name])

    def test_reaction_types_correct(self):
        species, rxns = self._make_full_propane_3rxn()
        net = rmg_network_to_vector(species, rxns, T=1123.0)

        types = {rxn.name: rxn.reaction_type for rxn in net.reactions}
        assert types["R1"] == "synthesis"
        assert types["R2"] == "exchange"
        assert types["R3"] == "synthesis"

    def test_k_vector_lengths(self):
        species, rxns = self._make_full_propane_3rxn()
        net = rmg_network_to_vector(species, rxns, T=1123.0)

        rxn_map = {rxn.name: rxn for rxn in net.reactions}
        # R1: CH3(2) + C2H5(4) → C3H8(8): k_fwd len=8, k_rev len=8
        assert len(rxn_map["R1"].k_forward) == 8
        assert len(rxn_map["R1"].k_reverse) == 8
        # R2: CH3(2) + C2H5(4) → CH4(2) + C2H4(4): k_fwd len=8, k_rev len=8
        assert len(rxn_map["R2"].k_forward) == 8
        assert len(rxn_map["R2"].k_reverse) == 8
        # R3: CH3(2) + CH3(2) → C2H6(4): k_fwd len=4, k_rev len=4
        assert len(rxn_map["R3"].k_forward) == 4
        assert len(rxn_map["R3"].k_reverse) == 4
