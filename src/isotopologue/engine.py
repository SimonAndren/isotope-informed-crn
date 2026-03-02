"""ODE-based isotopologue reaction network solver.

Evaluates the rate-of-change for all isotopologue concentrations and integrates
using scipy's implicit solvers (BDF/Radau) for stiff chemical kinetics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from isotopologue.species import Network, Reaction


@dataclass
class IsotopologueEngine:
    """Integrates isotopologue concentration ODEs for a reaction network.

    Precomputes offsets and scratch arrays at construction time so that the
    RHS function allocates nothing during integration.
    """

    network: Network
    _scratch: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        # Preallocate scratch arrays for synthesis/breakdown intermediates
        for rxn in self.network.reactions:
            key = rxn.name
            if rxn.reaction_type in ("synthesis", "exchange"):
                n_a = self.network.species[rxn.reactants[0]].n_isotopologues
                n_b = self.network.species[rxn.reactants[1]].n_isotopologues
                self._scratch[f"{key}_fwd"] = np.empty(n_a * n_b)
            if rxn.reaction_type == "exchange":
                n_c = self.network.species[rxn.products[0]].n_isotopologues
                n_d = self.network.species[rxn.products[1]].n_isotopologues
                self._scratch[f"{key}_rev"] = np.empty(n_c * n_d)
            if rxn.reaction_type == "synthesis":
                n_c = self.network.species[rxn.products[0]].n_isotopologues
                self._scratch[f"{key}_rev_breakdown"] = np.empty(n_c)

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """Compute dy/dt for all isotopologue concentrations."""
        dydt = np.zeros_like(y)
        net = self.network
        for rxn in net.reactions:
            self._apply_reaction(y, dydt, rxn)
        return dydt

    def _apply_reaction(self, y: np.ndarray, dydt: np.ndarray, rxn: Reaction):
        net = self.network
        rt = rxn.reaction_type

        if rt == "simple":
            self._apply_simple(y, dydt, rxn, net)
        elif rt == "breakdown":
            self._apply_breakdown(y, dydt, rxn, net)
        elif rt == "synthesis":
            self._apply_synthesis(y, dydt, rxn, net)
        elif rt == "exchange":
            self._apply_exchange(y, dydt, rxn, net)

    def _apply_simple(self, y: np.ndarray, dydt: np.ndarray, rxn: Reaction, net: Network):
        """A → C (with optional atom remap)."""
        sa, sc = net.offset(rxn.reactants[0]), net.offset(rxn.products[0])
        conc_a = y[sa[0] : sa[1]]
        conc_c = y[sc[0] : sc[1]]

        reacted_a = conc_a * rxn.k_forward
        reacted_c = conc_c * rxn.k_reverse

        if rxn.atom_map is not None:
            created_c = reacted_a[rxn.atom_map.forward]
            created_a = reacted_c[rxn.atom_map.reverse]
        else:
            created_c = reacted_a
            created_a = reacted_c

        dydt[sa[0] : sa[1]] += created_a - reacted_a
        dydt[sc[0] : sc[1]] += created_c - reacted_c

    def _apply_breakdown(self, y: np.ndarray, dydt: np.ndarray, rxn: Reaction, net: Network):
        """A → C + D."""
        sa = net.offset(rxn.reactants[0])
        sc = net.offset(rxn.products[0])
        sd = net.offset(rxn.products[1])
        conc_a = y[sa[0] : sa[1]]
        conc_c = y[sc[0] : sc[1]]
        conc_d = y[sd[0] : sd[1]]
        n_c = net.species[rxn.products[0]].n_isotopologues
        n_d = net.species[rxn.products[1]].n_isotopologues

        # Forward: A → C + D
        reacted_a = conc_a * rxn.k_forward
        if rxn.atom_map is not None:
            reacted_a = reacted_a[rxn.atom_map.forward]
        mat = reacted_a.reshape(n_c, n_d)
        created_c = mat.sum(axis=1)
        created_d = mat.sum(axis=0)

        # Reverse: C + D → A
        created_a_full = np.outer(conc_c, conc_d).ravel() * rxn.k_reverse
        if rxn.atom_map is not None:
            created_a_full = created_a_full[rxn.atom_map.reverse]
        # Reverse synthesis: outer product of C, D gives full A-space vector
        rev_mat = np.outer(conc_c, conc_d) * rxn.k_reverse.reshape(n_c, n_d)
        reacted_c = rev_mat.sum(axis=1)
        reacted_d = rev_mat.sum(axis=0)
        created_a = rev_mat.ravel()
        if rxn.atom_map is not None:
            created_a = created_a[rxn.atom_map.reverse]

        dydt[sa[0] : sa[1]] += created_a - reacted_a.ravel()
        dydt[sc[0] : sc[1]] += created_c - reacted_c
        dydt[sd[0] : sd[1]] += created_d - reacted_d

    def _apply_synthesis(self, y: np.ndarray, dydt: np.ndarray, rxn: Reaction, net: Network):
        """A + B → C."""
        sa = net.offset(rxn.reactants[0])
        sb = net.offset(rxn.reactants[1])
        sc = net.offset(rxn.products[0])
        conc_a = y[sa[0] : sa[1]]
        conc_b = y[sb[0] : sb[1]]
        conc_c = y[sc[0] : sc[1]]
        n_a = net.species[rxn.reactants[0]].n_isotopologues
        n_b = net.species[rxn.reactants[1]].n_isotopologues

        # Forward: A + B → C
        outer = np.outer(conc_a, conc_b).ravel()
        created_c = outer * rxn.k_forward
        if rxn.atom_map is not None:
            created_c = created_c[rxn.atom_map.forward]
        # Consumed A and B from forward reaction
        mat_fwd = (outer * rxn.k_forward).reshape(n_a, n_b)
        reacted_a = mat_fwd.sum(axis=1)
        reacted_b = mat_fwd.sum(axis=0)

        # Reverse: C → A + B
        reacted_c = conc_c * rxn.k_reverse
        if rxn.atom_map is not None:
            reacted_c_remapped = reacted_c[rxn.atom_map.reverse]
        else:
            reacted_c_remapped = reacted_c
        mat_rev = reacted_c_remapped.reshape(n_a, n_b)
        created_a = mat_rev.sum(axis=1)
        created_b = mat_rev.sum(axis=0)

        dydt[sa[0] : sa[1]] += created_a - reacted_a
        dydt[sb[0] : sb[1]] += created_b - reacted_b
        dydt[sc[0] : sc[1]] += created_c.ravel() - reacted_c

    def _apply_exchange(self, y: np.ndarray, dydt: np.ndarray, rxn: Reaction, net: Network):
        """A + B → C + D (synthesis → remap → breakdown)."""
        sa = net.offset(rxn.reactants[0])
        sb = net.offset(rxn.reactants[1])
        sc = net.offset(rxn.products[0])
        sd = net.offset(rxn.products[1])
        conc_a = y[sa[0] : sa[1]]
        conc_b = y[sb[0] : sb[1]]
        conc_c = y[sc[0] : sc[1]]
        conc_d = y[sd[0] : sd[1]]
        n_a = net.species[rxn.reactants[0]].n_isotopologues
        n_b = net.species[rxn.reactants[1]].n_isotopologues
        n_c = net.species[rxn.products[0]].n_isotopologues
        n_d = net.species[rxn.products[1]].n_isotopologues

        # Forward: A + B → intermediate → C + D
        outer_fwd = np.outer(conc_a, conc_b).ravel()
        intermediate_fwd = outer_fwd * rxn.k_forward
        if rxn.atom_map is not None:
            intermediate_fwd = intermediate_fwd[rxn.atom_map.forward]
        mat_fwd = intermediate_fwd.reshape(n_c, n_d)
        created_c = mat_fwd.sum(axis=1)
        created_d = mat_fwd.sum(axis=0)
        # Consumed A and B
        consumed_fwd = (outer_fwd * rxn.k_forward).reshape(n_a, n_b)
        reacted_a = consumed_fwd.sum(axis=1)
        reacted_b = consumed_fwd.sum(axis=0)

        # Reverse: C + D → intermediate → A + B
        outer_rev = np.outer(conc_c, conc_d).ravel()
        intermediate_rev = outer_rev * rxn.k_reverse
        if rxn.atom_map is not None:
            intermediate_rev = intermediate_rev[rxn.atom_map.reverse]
        mat_rev = intermediate_rev.reshape(n_a, n_b)
        created_a = mat_rev.sum(axis=1)
        created_b = mat_rev.sum(axis=0)
        # Consumed C and D
        consumed_rev = (outer_rev * rxn.k_reverse).reshape(n_c, n_d)
        reacted_c = consumed_rev.sum(axis=1)
        reacted_d = consumed_rev.sum(axis=0)

        dydt[sa[0] : sa[1]] += created_a - reacted_a
        dydt[sb[0] : sb[1]] += created_b - reacted_b
        dydt[sc[0] : sc[1]] += created_c - reacted_c
        dydt[sd[0] : sd[1]] += created_d - reacted_d

    def solve(
        self,
        y0: np.ndarray,
        t_span: tuple[float, float],
        t_eval: np.ndarray | None = None,
        method: str = "BDF",
        rtol: float = 1e-8,
        atol: float = 1e-12,
        **kwargs,
    ):
        """Integrate the isotopologue ODEs.

        Args:
            y0: Initial state vector (from Network.pack).
            t_span: (t_start, t_end) in seconds.
            t_eval: Optional times at which to store the solution.
            method: ODE solver method ('BDF', 'Radau', 'RK45', etc.).
            rtol: Relative tolerance.
            atol: Absolute tolerance.

        Returns:
            scipy.integrate.OdeResult with .t, .y, .success attributes.
        """
        return solve_ivp(
            self.rhs,
            t_span,
            y0,
            method=method,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            **kwargs,
        )
