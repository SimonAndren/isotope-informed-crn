# RPFR-GUI Architecture Documentation

## Overview

RPFR-GUI is a modular, high-performance Python library for analyzing isotope fractionation in molecular systems. The architecture prioritizes **lazy loading**, **strict typing**, and **graph-based modeling**.

---

## Design Principles

1. **Lazy Loading**: Never load the full >10GB HDF5 dataset into memory
2. **Strategy Pattern**: Pluggable data providers for different sources
3. **Graph Theory**: Represent isotope exchange as NetworkX graphs
4. **Type Safety**: Full type hints throughout the codebase
5. **Testability**: Comprehensive unit and integration tests

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interface                          │
│                   (Jupyter Notebooks)                        │
│                      [Phase 2]                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Domain Layer                            │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  IsotopeGraph (NetworkX)                             │  │
│  │  • Nodes: Atoms with RPFR values                     │  │
│  │  • Edges: Exchange pathways                          │  │
│  │  • Operations: Relative RPFR, subgraphs, analysis    │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Data Layer                              │
│                                                              │
│  ┌────────────────────┐         ┌──────────────────────┐   │
│  │ ChemistryResolver  │         │  Data Providers      │   │
│  │ • Canonicalize     │         │  (Strategy Pattern)  │   │
│  │   SMILES/InChI     │         │                      │   │
│  │ • Generate index   │         │  ┌────────────────┐  │   │
│  │ • Fast lookup      │         │  │  H5Provider    │  │   │
│  └────────────────────┘         │  │  (Lazy HDF5)   │  │   │
│                                  │  └────────────────┘  │   │
│                                  │  ┌────────────────┐  │   │
│                                  │  │  GNNProvider   │  │   │
│                                  │  │  [Future]      │  │   │
│                                  │  └────────────────┘  │   │
│                                  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Storage Layer                           │
│                                                              │
│  ┌──────────────────┐         ┌──────────────────────┐     │
│  │  HDF5 Store      │         │  Parquet Index       │     │
│  │  (qm9s.h5)       │         │  (Fast Lookup)       │     │
│  │  >10GB           │         │  <10MB               │     │
│  └──────────────────┘         └──────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Data Layer

#### ChemistryResolver
**Purpose**: Maps chemical identifiers to database indices

**Responsibilities**:
- Canonicalize SMILES/InChI using RDKit
- Generate lightweight lookup indices (Parquet/CSV)
- Fast molecule ID resolution

**Key Methods**:
```python
canonicalize_smiles(smiles: str) -> Optional[str]
resolve(identifier: str, *, id_type: str) -> Optional[str]
batch_resolve(identifiers: list[str]) -> Dict[str, Optional[str]]
build_index(h5_path: Path, output_path: Path)  # Static method
```

**Design Notes**:
- Index generation is a one-time operation
- Uses RDKit's canonical SMILES for consistency
- Supports both SMILES and InChI input

---

#### AbstractProvider
**Purpose**: Interface for data access strategies

**Contract**:
```python
@abstractmethod
def get_rpfr(molecule_id: str, temperature: float) -> Optional[pd.DataFrame]
def get_structure(molecule_id: str) -> Optional[str]
def has_molecule(molecule_id: str) -> bool
```

**Implementations**:
1. **H5Provider**: Lazy HDF5 access
2. **GNNProvider**: Future ML predictions

---

#### H5Provider
**Purpose**: Lazy loading from HDF5 files

**Responsibilities**:
- Open HDF5 file only when needed
- Load individual molecules on demand
- Decode HDF5 data types to Python types
- Support batch operations with progress tracking

**Key Methods**:
```python
get_rpfr(molecule_id: str, temperature: float) -> Optional[pd.DataFrame]
get_full_atom_data(molecule_id: str, datasets: Dict) -> Optional[pd.DataFrame]
load_batch(molecule_ids: list[str]) -> pd.DataFrame
```

**Performance Characteristics**:
- Memory: O(1) per molecule lookup
- I/O: Minimal seeks due to HDF5 structure
- No file locking (read-only access)

---

### 2. Domain Layer

#### IsotopeGraph
**Purpose**: Model isotope exchange networks as graphs

**Graph Structure**:
- **Nodes**: Atoms with attributes (RPFR, symbol, molecule_id)
- **Edges**: Exchange pathways between compatible atoms
- **Components**: Disconnected reaction systems

**Connectivity Modes**:
1. **Full**: All atoms of the same element can exchange
2. **Custom**: User-defined edge list

**Key Methods**:
```python
add_molecule(molecule_id: str, atom_data: pd.DataFrame) -> List[str]
set_connectivity(mode: str, custom_edges: Optional[List]) -> None
set_anchor(node_id: str) -> None
get_relative_rpfr(node_id: str) -> Optional[float]
get_subgraph_by_element(element: str) -> nx.Graph
get_connected_components() -> List[Set[str]]
```

**Future Features**:
- Mass law scaling (O¹⁷ → O¹⁸ conversion)
- Clumping effects (combinatorial logic)
- Graph-based equilibrium calculations

---

### 3. UI Layer (Phase 2)

**Planned Components**:
- Molecule selection widget (SMILES input)
- 2D structure viewer with RDKit
- RPFR heatmap overlay (Viridis colormap)
- Graph visualization with ipycytoscape
- Configuration panel (anchor, mass laws, clumping)

---

## Data Flow Examples

### Example 1: Basic Workflow

```python
# 1. Resolve molecule
resolver = ChemistryResolver("index.parquet")
mol_id = resolver.resolve("CC")  # Ethane

# 2. Load data
provider = H5Provider("qm9s.h5")
rpfr_data = provider.get_rpfr(mol_id, temperature=300.0)

# 3. Build graph
graph = IsotopeGraph(connectivity="full")
graph.add_molecule(mol_id, rpfr_data)
graph.set_connectivity(mode="full")

# 4. Analyze
graph.set_anchor("mol_001_0")  # First carbon
df = graph.get_rpfr_dataframe(relative=True)
```

### Example 2: Batch Processing

```python
# Resolve multiple molecules
smiles_list = ["C", "CC", "CCO", "CO"]
mol_ids = resolver.batch_resolve(smiles_list)

# Load and graph
graph = IsotopeGraph()
for mol_id in mol_ids.values():
    if mol_id:
        data = provider.get_rpfr(mol_id)
        graph.add_molecule(mol_id, data)

graph.set_connectivity(mode="full")
summary = graph.summary()
```

---

## Testing Strategy

### Unit Tests
- **ChemistryResolver**: Canonicalization, index building
- **H5Provider**: Data loading, decoding
- **IsotopeGraph**: Node/edge operations, connectivity

### Integration Tests
- End-to-end workflows
- Batch processing
- Multi-molecule graphs

### Fixtures (tests/conftest.py)
- Sample HDF5 files
- Sample SMILES data
- Temporary indices

---

## Performance Considerations

### Memory
- **Index**: 10MB for 130K molecules
- **Per-molecule**: ~1KB (methane) to ~10KB (larger molecules)
- **Graph**: O(N) where N = number of atoms

### I/O
- **Index lookup**: O(1) with Parquet
- **HDF5 read**: O(1) per molecule (chunked storage)
- **Batch loading**: Progress tracking with tqdm

### Scalability
- **Molecules**: Tested with 130K molecules
- **Atoms per graph**: Supports thousands of nodes
- **Parallelization**: Batch operations can be parallelized

---

## Dependencies

### Core
- `h5py`: HDF5 file access
- `pandas`: DataFrames
- `numpy`: Numerical operations
- `rdkit`: Chemistry toolkit
- `networkx`: Graph operations

### UI (Phase 2)
- `ipywidgets`: Jupyter widgets
- `matplotlib`/`plotly`: Visualization
- `ipycytoscape`: Graph visualization

### Development
- `pytest`: Testing framework
- `ruff`: Linting and formatting
- `mypy`: Type checking
- `uv`: Package management

---

## Future Enhancements

### Phase 2: UI Layer
- Interactive Jupyter widgets
- Real-time visualization
- Configuration panel

### Phase 3: Advanced Features
- GNN-based predictions
- Mass law transformations
- Clumping calculations
- Multi-temperature analysis

### Phase 4: Performance
- Parallel processing
- GPU acceleration (via CuGraph)
- Distributed computing (Dask)

---

## References

- [NetworkX Documentation](https://networkx.org/)
- [RDKit Documentation](https://www.rdkit.org/docs/)
- [HDF5 Best Practices](https://www.hdfgroup.org/)
- [uv Package Manager](https://github.com/astral-sh/uv)
