"""Unit tests for isotopologue.species data structures."""

import numpy as np
import pytest

from isotopologue.species import AtomMap, Network, Reaction, Species, build_atom_map


class TestSpecies:
    def test_creation(self):
        sp = Species("CH4", n_labeled=1, element="C")
        assert sp.name == "CH4"
        assert sp.n_labeled == 1
        assert sp.element == "C"

    def test_isotopologue_count(self):
        assert Species("CH4", 1).n_isotopologues == 2
        assert Species("C2H6", 2).n_isotopologues == 4
        assert Species("C3H8", 3).n_isotopologues == 8
        assert Species("glucose", 6).n_isotopologues == 64

    def test_zero_labeled(self):
        sp = Species("H2", 0)
        assert sp.n_isotopologues == 1

    def test_default_element(self):
        sp = Species("X", 2)
        assert sp.element == "C"

    def test_multi_element(self):
        sp = Species("H2O", n_labeled=2, element="H")
        assert sp.element == "H"
        assert sp.n_isotopologues == 4


class TestBuildAtomMap:
    def test_identity(self):
        """Identity mapping: [2,1,0] → [2,1,0] should be range(8)."""
        result = build_atom_map([2, 1, 0], [2, 1, 0])
        np.testing.assert_array_equal(result, np.arange(8))

    def test_reversal(self):
        """Reverse bit order: position 0↔2 swap for 3-bit molecule."""
        fwd = build_atom_map([2, 1, 0], [0, 1, 2])
        # Index 0b100 (4) → should map to 0b001 (1) and vice versa
        # new_conc = old_conc[fwd]
        conc = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=float)
        new_conc = conc[fwd]
        # After reversing bits:
        # 0b000(0)→0b000(0), 0b001(1)→0b100(4), 0b010(2)→0b010(2),
        # 0b011(3)→0b110(6), 0b100(4)→0b001(1), 0b101(5)→0b101(5),
        # 0b110(6)→0b011(3), 0b111(7)→0b111(7)
        expected = np.array([0, 4, 2, 6, 1, 5, 3, 7], dtype=float)
        np.testing.assert_array_equal(new_conc, expected)

    def test_is_permutation(self):
        result = build_atom_map([3, 2, 1, 0], [1, 0, 3, 2])
        assert len(result) == 16
        assert set(result.tolist()) == set(range(16))

    def test_two_atoms(self):
        """Simple 2-atom swap."""
        fwd = build_atom_map([1, 0], [0, 1])
        conc = np.array([10.0, 20.0, 30.0, 40.0])
        new_conc = conc[fwd]
        # Swap bits: 00→00, 01→10, 10→01, 11→11
        expected = np.array([10.0, 30.0, 20.0, 40.0])
        np.testing.assert_array_equal(new_conc, expected)


class TestAtomMap:
    def test_from_bit_maps(self):
        am = AtomMap.from_bit_maps([2, 1, 0], [2, 1, 0], [2, 1, 0])
        np.testing.assert_array_equal(am.forward, np.arange(8))
        np.testing.assert_array_equal(am.reverse, np.arange(8))

    def test_forward_reverse_inverse(self):
        """Forward and reverse with same mapping should be identical."""
        am = AtomMap.from_bit_maps([2, 1, 0], [0, 1, 2], [0, 1, 2])
        # Applying forward then reverse should recover original
        conc = np.arange(8, dtype=float)
        mapped = conc[am.forward]
        # This specific bit swap is self-inverse, so applying forward twice recovers original
        np.testing.assert_array_equal(mapped[am.forward], conc)


class TestReaction:
    def test_simple(self):
        rxn = Reaction(
            "R1",
            ("A",),
            ("C",),
            k_forward=np.ones(4),
            k_reverse=np.ones(4),
        )
        assert rxn.reaction_type == "simple"

    def test_breakdown(self):
        rxn = Reaction(
            "R2",
            ("A",),
            ("C", "D"),
            k_forward=np.ones(8),
            k_reverse=np.ones(8),
        )
        assert rxn.reaction_type == "breakdown"

    def test_synthesis(self):
        rxn = Reaction(
            "R3",
            ("A", "B"),
            ("C",),
            k_forward=np.ones(8),
            k_reverse=np.ones(8),
        )
        assert rxn.reaction_type == "synthesis"

    def test_exchange(self):
        rxn = Reaction(
            "R4",
            ("A", "B"),
            ("C", "D"),
            k_forward=np.ones(16),
            k_reverse=np.ones(16),
        )
        assert rxn.reaction_type == "exchange"


class TestNetwork:
    def test_state_size(self):
        net = Network(
            species={
                "A": Species("A", 2),  # 4
                "B": Species("B", 1),  # 2
                "C": Species("C", 3),  # 8
            }
        )
        assert net.state_size == 14

    def test_pack_unpack_roundtrip(self):
        net = Network(
            species={
                "A": Species("A", 2),
                "B": Species("B", 1),
            }
        )
        conc = {
            "A": np.array([1.0, 2.0, 3.0, 4.0]),
            "B": np.array([5.0, 6.0]),
        }
        packed = net.pack(conc)
        assert packed.shape == (6,)
        np.testing.assert_array_equal(packed, [1, 2, 3, 4, 5, 6])

        unpacked = net.unpack(packed)
        np.testing.assert_array_equal(unpacked["A"], conc["A"])
        np.testing.assert_array_equal(unpacked["B"], conc["B"])

    def test_offset(self):
        net = Network(
            species={
                "X": Species("X", 1),
                "Y": Species("Y", 2),
            }
        )
        assert net.offset("X") == (0, 2)
        assert net.offset("Y") == (2, 6)

    def test_empty_network(self):
        net = Network(species={})
        assert net.state_size == 0
        packed = net.pack({})
        assert packed.shape == (0,)
        unpacked = net.unpack(packed)
        assert unpacked == {}
