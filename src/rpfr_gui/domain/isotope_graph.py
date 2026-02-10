"""Isotope exchange network representation using NetworkX.

This module implements the core domain logic for representing isotope
exchange systems as graphs, where nodes are atoms (sites) and edges
represent valid exchange pathways.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd


class IsotopeGraph:
    """Represents an isotope exchange network as a graph.

    The graph models the chemical system where:
    - Nodes: Specific atoms (sites) capable of isotope exchange
    - Edges: Valid exchange pathways (reactions)
    - Node attributes: RPFR values, atom symbols, molecule IDs

    Attributes
    ----------
    graph : nx.Graph
        The underlying NetworkX graph.
    anchor_node : str or None
        The reference site for relative calculations.
    mass_law_enabled : bool
        Whether mass-dependent scaling is applied.
    """

    def __init__(
        self,
        *,
        connectivity: str = "full",
        mass_law_enabled: bool = False,
    ):
        """Initialize an empty isotope exchange network.

        Parameters
        ----------
        connectivity : str, optional
            Graph connectivity type: "full" (all atoms connected) or
            "custom" (user-defined edges). Default is "full".
        mass_law_enabled : bool, optional
            Enable mass-dependent scaling for isotope conversions.
            Default is False.
        """
        self.graph = nx.Graph()
        self.connectivity = connectivity
        self.mass_law_enabled = mass_law_enabled
        self.anchor_node: str | None = None
        self.anchor_element: str | None = None  # element of the current anchor

        # Mapping from node ID to (molecule_id, atom_index)
        self._node_to_atom: dict[str, tuple[str, int]] = {}

    def add_molecule(
        self,
        molecule_id: str,
        atom_data: pd.DataFrame,
        *,
        rpfr_column: str = "RPFR_300K",
        symbol_column: str = "Atom_Symbol",
    ) -> list[str]:
        """Add a molecule's atoms as nodes to the graph.

        Parameters
        ----------
        molecule_id : str
            Unique identifier for the molecule.
        atom_data : pd.DataFrame
            Per-atom data with columns: Atom_Index, <symbol_column>, <rpfr_column>
        rpfr_column : str, optional
            Name of the RPFR data column. Default is "RPFR_300K".
        symbol_column : str, optional
            Name of the atom symbol column. Default is "Atom_Symbol".

        Returns
        -------
        list of str
            Node IDs added to the graph.
        """
        node_ids = []

        for _, row in atom_data.iterrows():
            atom_idx = int(row["Atom_Index"])
            node_id = f"{molecule_id}_{atom_idx}"

            # Add node with attributes
            self.graph.add_node(
                node_id,
                molecule_id=molecule_id,
                atom_index=atom_idx,
                atom_symbol=row[symbol_column],
                rpfr=float(row[rpfr_column]),
            )

            self._node_to_atom[node_id] = (molecule_id, atom_idx)
            node_ids.append(node_id)

        return node_ids

    def set_connectivity(
        self, *, mode: str = "full", custom_edges: list[tuple[str, str]] | None = None
    ):
        """Define the exchange connectivity between atoms.

        Parameters
        ----------
        mode : str, optional
            "full" for fully connected graph (all atoms can exchange),
            "custom" for user-defined edges. Default is "full".
        custom_edges : list of tuples, optional
            List of (node_id_1, node_id_2) pairs for custom connectivity.
        """
        if mode == "full":
            self._build_full_connectivity()
        elif mode == "custom":
            if custom_edges is None:
                raise ValueError("custom_edges must be provided for 'custom' mode")
            self._build_custom_connectivity(custom_edges)
        else:
            raise ValueError(f"Unknown connectivity mode: {mode}")

    def _build_full_connectivity(self):
        """Create a fully connected graph (all atoms can exchange)."""
        nodes = list(self.graph.nodes())

        # Group by atom symbol for element-specific connectivity
        symbol_groups: dict[str, list[str]] = {}
        for node in nodes:
            symbol = self.graph.nodes[node]["atom_symbol"]
            if symbol not in symbol_groups:
                symbol_groups[symbol] = []
            symbol_groups[symbol].append(node)

        # Connect all atoms of the same element
        for symbol, node_list in symbol_groups.items():
            for i, node1 in enumerate(node_list):
                for node2 in node_list[i + 1 :]:
                    self.graph.add_edge(node1, node2, exchange_type=f"{symbol}_exchange")

    def _build_custom_connectivity(self, edges: list[tuple[str, str]]):
        """Add custom edges to the graph."""
        for node1, node2 in edges:
            if node1 not in self.graph or node2 not in self.graph:
                raise ValueError(f"Edge ({node1}, {node2}) references unknown nodes")
            self.graph.add_edge(node1, node2, exchange_type="custom")

    def set_anchor(self, node_id: str):
        """Set the reference (anchor) site for relative RPFR calculations.

        The anchor defines the element context for normalization: only nodes
        with the same element as the anchor will have valid relative RPFR values,
        since isotope exchange only occurs within the same element.

        Parameters
        ----------
        node_id : str
            Node ID to use as the anchor.

        Raises
        ------
        ValueError
            If node_id is not in the graph.
        """
        if node_id not in self.graph:
            raise ValueError(f"Node {node_id} not found in graph")
        self.anchor_node = node_id
        self.anchor_element = self.graph.nodes[node_id]["atom_symbol"]

    def get_relative_rpfr(self, node_id: str) -> float | None:
        """Calculate RPFR relative to the anchor node.

        Isotope exchange only occurs within the same element, so this method
        returns None when the queried node's element differs from the anchor's
        element (e.g., querying an H atom when the anchor is C is undefined).

        Parameters
        ----------
        node_id : str
            Node ID to calculate relative RPFR for.

        Returns
        -------
        float or None
            Relative RPFR (node_rpfr / anchor_rpfr), or None if no anchor is
            set or if the node's element differs from the anchor element.
        """
        if self.anchor_node is None:
            return None

        node_element = self.graph.nodes[node_id]["atom_symbol"]
        if node_element != self.anchor_element:
            return None  # Cross-element comparison is physically undefined

        node_rpfr = self.graph.nodes[node_id]["rpfr"]
        anchor_rpfr = self.graph.nodes[self.anchor_node]["rpfr"]

        if anchor_rpfr == 0:
            raise ValueError("Anchor RPFR is zero, cannot compute relative values")

        return node_rpfr / anchor_rpfr

    def get_element_normalized_rpfr(self, method: str = "mean") -> pd.DataFrame:
        """Return RPFR values normalized independently within each element.

        Because RPFR values are not comparable across elements (H RPFR ~10-15,
        C/N/O RPFR ~1.1-1.2), each element is normalized on its own scale.
        This is useful for visualization where all atoms must share a color map.

        Parameters
        ----------
        method : str, optional
            Normalization method per element:
            - "mean"   : (rpfr - mean) / mean  (fractional deviation from mean)
            - "minmax" : (rpfr - min) / (max - min)  (0 to 1 within element)
            Default is "mean".

        Returns
        -------
        pd.DataFrame
            Columns: node_id, molecule_id, atom_index, atom_symbol, rpfr,
            normalized_rpfr, element_mean, element_std
        """
        rows = []
        for node_id in self.graph.nodes():
            attrs = self.graph.nodes[node_id]
            rows.append(
                {
                    "node_id": node_id,
                    "molecule_id": attrs["molecule_id"],
                    "atom_index": attrs["atom_index"],
                    "atom_symbol": attrs["atom_symbol"],
                    "rpfr": attrs["rpfr"],
                }
            )
        df = pd.DataFrame(rows)

        # Compute per-element statistics
        stats = df.groupby("atom_symbol")["rpfr"].agg(["mean", "std", "min", "max"])
        df = df.join(
            stats.rename(
                columns={
                    "mean": "element_mean",
                    "std": "element_std",
                    "min": "element_min",
                    "max": "element_max",
                }
            ),
            on="atom_symbol",
        )

        # Apply normalization
        if method == "mean":
            df["normalized_rpfr"] = (df["rpfr"] - df["element_mean"]) / df["element_mean"]
        elif method == "minmax":
            rng = (df["element_max"] - df["element_min"]).replace(0, 1.0)
            df["normalized_rpfr"] = (df["rpfr"] - df["element_min"]) / rng
        else:
            raise ValueError(f"Unknown normalization method: {method!r}. Use 'mean' or 'minmax'.")

        return df.drop(columns=["element_min", "element_max"])

    def get_rpfr_dataframe(self, *, relative: bool = False) -> pd.DataFrame:
        """Export RPFR values as a DataFrame.

        When ``relative=True`` the column is ``relative_rpfr`` and nodes whose
        element differs from the anchor element will have ``NaN`` (since
        cross-element isotope comparison is undefined).

        Parameters
        ----------
        relative : bool, optional
            If True, return anchor-relative RPFR values for the anchor's element;
            other elements receive NaN. Default is False.

        Returns
        -------
        pd.DataFrame
            Columns: node_id, molecule_id, atom_index, atom_symbol,
            rpfr or relative_rpfr.
        """
        rows = []
        for node_id in self.graph.nodes():
            attrs = self.graph.nodes[node_id]
            rpfr_val = self.get_relative_rpfr(node_id) if relative else attrs["rpfr"]

            rows.append(
                {
                    "node_id": node_id,
                    "molecule_id": attrs["molecule_id"],
                    "atom_index": attrs["atom_index"],
                    "atom_symbol": attrs["atom_symbol"],
                    "rpfr" if not relative else "relative_rpfr": rpfr_val,
                }
            )

        return pd.DataFrame(rows)

    def get_subgraph_by_element(self, element: str) -> nx.Graph:
        """Extract a subgraph containing only atoms of a specific element.

        Parameters
        ----------
        element : str
            Atomic symbol (e.g., "H", "C", "O").

        Returns
        -------
        nx.Graph
            Subgraph containing only the specified element.
        """
        nodes = [n for n in self.graph.nodes() if self.graph.nodes[n]["atom_symbol"] == element]
        return self.graph.subgraph(nodes).copy()

    def get_connected_components(self) -> list[set[str]]:
        """Identify disconnected subgraphs (separate reaction systems).

        Returns
        -------
        list of sets
            Each set contains node IDs in a connected component.
        """
        return [set(comp) for comp in nx.connected_components(self.graph)]

    def apply_mass_law_scaling(
        self,
        *,
        source_isotope: str = "17O",
        target_isotope: str = "18O",
        scaling_factor: float | None = None,
    ):
        """Apply mass-law scaling to convert RPFR values between isotopes.

        Parameters
        ----------
        source_isotope : str, optional
            Source isotope label. Default is "17O".
        target_isotope : str, optional
            Target isotope label. Default is "18O".
        scaling_factor : float, optional
            Custom scaling factor. If None, uses theoretical mass law.
        """
        if not self.mass_law_enabled:
            raise RuntimeError("Mass law is disabled. Enable it during initialization.")

        # Placeholder for mass law logic
        # In a real implementation, this would:
        # 1. Calculate mass-dependent fractionation factors
        # 2. Update node RPFR values accordingly
        # 3. Store metadata about the transformation

        raise NotImplementedError("Mass law scaling not yet implemented")

    def summary(self) -> dict:
        """Generate a summary of the isotope graph.

        Returns
        -------
        dict
            Summary statistics including number of nodes, edges, elements, etc.
        """
        elements = {self.graph.nodes[n]["atom_symbol"] for n in self.graph.nodes()}
        components = self.get_connected_components()

        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "elements": sorted(elements),
            "num_molecules": len({self._node_to_atom[n][0] for n in self.graph.nodes()}),
            "num_connected_components": len(components),
            "connectivity_mode": self.connectivity,
            "anchor_set": self.anchor_node is not None,
            "mass_law_enabled": self.mass_law_enabled,
        }

    def __repr__(self) -> str:
        """Return a string representation of the graph."""
        summary = self.summary()
        return (
            f"IsotopeGraph(nodes={summary['num_nodes']}, "
            f"edges={summary['num_edges']}, "
            f"elements={summary['elements']}, "
            f"connectivity={self.connectivity})"
        )
