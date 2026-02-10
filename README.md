# RPFR-GUI: Isotope Fractionation Analysis Tool

[![Tests](https://github.com/simonandren/isotope-informed-crn/actions/workflows/test.yml/badge.svg)](https://github.com/simonandren/isotope-informed-crn/actions/workflows/test.yml)
[![Lint](https://github.com/simonandren/isotope-informed-crn/actions/workflows/lint.yml/badge.svg)](https://github.com/simonandren/isotope-informed-crn/actions/workflows/lint.yml)
[![Coverage](https://github.com/simonandren/isotope-informed-crn/actions/workflows/coverage.yml/badge.svg)](https://github.com/simonandren/isotope-informed-crn/actions/workflows/coverage.yml)
[![codecov](https://codecov.io/gh/simonandren/isotope-informed-crn/graph/badge.svg)](https://codecov.io/gh/simonandren/isotope-informed-crn)

A high-performance, modular Python library for analyzing Reduced Partition Function Ratios (RPFR) in molecular systems. Designed for site-specific isotope fractionation analysis with lazy loading for large (>10GB) datasets.

---

## Features

- **Lazy Loading**: Efficiently handle >10GB HDF5 datasets without loading everything into memory
- **Chemistry Resolution**: Automatic canonicalization and lookup of SMILES/InChI strings using RDKit
- **Graph-Based Models**: Represent isotope exchange networks using NetworkX
- **Jupyter Integration**: Interactive widgets for visualization and analysis
- **Strict Typing**: Type-safe interfaces throughout the codebase
- **Modular Architecture**: Clean separation between data, domain logic, and UI layers

---

## Quick Start

### 1. Install `uv`

`uv` is a fast Python package manager that we use for dependency management.

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Or via pip:**
```bash
pip install uv
```

For more installation options, see the [uv documentation](https://github.com/astral-sh/uv).

---

### 2. Clone the Repository

```bash
git clone https://github.com/simonandren/isotope-informed-crn.git
cd isotope-informed-crn
```

---

### 3. Install Dependencies

```bash
# Install all dependencies (including dev tools)
uv sync --all-extras

# Or install only core dependencies
uv sync
```

This will:
- Create a virtual environment
- Install all required packages
- Set up the development environment

---

### 4. Verify Installation

```bash
# Run the test suite
uv run pytest tests/ -v

# Check code formatting
uv run ruff check src/

# Run a quick type check
uv run mypy src/rpfr_gui --ignore-missing-imports
```

---

## Usage

### Basic Workflow

```python
from pathlib import Path
from rpfr_gui.data import ChemistryResolver, H5Provider
from rpfr_gui.domain import IsotopeGraph

# 1. Set up data access
h5_path = Path("data/raw/qm9s.h5")
provider = H5Provider(h5_path)

# 2. Resolve a molecule by SMILES
index_path = Path("data/processed/index.csv")
resolver = ChemistryResolver(index_path)
mol_id = resolver.resolve("CC", id_type="smiles")  # Ethane

# 3. Load RPFR data
rpfr_data = provider.get_rpfr(mol_id, temperature=300.0)

# 4. Build isotope exchange network
graph = IsotopeGraph(connectivity="full")
graph.add_molecule(mol_id, rpfr_data)
graph.set_connectivity(mode="full")

# 5. Analyze
summary = graph.summary()
print(summary)
```

---

### Building the Index

Before using the `ChemistryResolver`, you need to generate an index from your HDF5 file:

```python
from pathlib import Path
from rpfr_gui.data import ChemistryResolver

ChemistryResolver.build_index(
    h5_path=Path("data/raw/qm9s.h5"),
    output_path=Path("data/processed/index.parquet"),
    smiles_dataset="SMILES",
)
```

This creates a lightweight lookup table for fast molecule resolution.

---

## Project Structure

```
isotope-informed-crn/
├── src/
│   └── rpfr_gui/
│       ├── data/              # Data layer (lazy loading, chemistry resolution)
│       │   ├── chemistry_resolver.py
│       │   └── providers.py
│       ├── domain/            # Domain logic (isotope graphs)
│       │   └── isotope_graph.py
│       └── ui/                # User interface (Jupyter widgets)
├── tests/
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── conftest.py            # Shared test fixtures
├── notebooks/
│   └── exploration/           # Exploratory notebooks
├── data/
│   ├── raw/                   # Original HDF5 files (gitignored)
│   └── processed/             # Generated indices (gitignored)
├── .github/
│   └── workflows/             # CI/CD configurations
├── pyproject.toml             # Project metadata and dependencies
└── README.md
```

---

## Development

### Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run with coverage report
uv run pytest tests/ --cov=rpfr_gui --cov-report=html

# Run only unit tests
uv run pytest tests/unit/ -v
```

### Code Formatting

```bash
# Check formatting
uv run ruff check src/ tests/

# Auto-fix issues
uv run ruff check src/ tests/ --fix

# Format code
uv run ruff format src/ tests/
```

### Type Checking

```bash
uv run mypy src/rpfr_gui
```

---

## Architecture

### Data Layer

- **`ChemistryResolver`**: Maps SMILES/InChI → Molecule IDs using RDKit canonicalization
- **`AbstractProvider`**: Strategy pattern interface for data access
- **`H5Provider`**: Lazy HDF5 reader (only loads requested molecules)
- **`GNNProvider`**: Placeholder for future GNN-based predictions

### Domain Layer

- **`IsotopeGraph`**: NetworkX-based representation of isotope exchange networks
  - Nodes: Atoms (sites) with RPFR values
  - Edges: Valid exchange pathways
  - Supports fully connected or custom connectivity

### UI Layer (Coming in Phase 2)

- Jupyter widgets for interactive molecule selection
- 2D structure visualization with RDKit
- Heatmap overlays for RPFR values
- Graph visualization with ipycytoscape

---

## Data Requirements

This tool expects HDF5 files with the following structure:

```
qm9s.h5
├── 000001/                    # Molecule ID
│   ├── SMILES                 # Molecular structure
│   ├── Atom_Symbol            # Per-atom element symbols
│   ├── RPFR_300K              # RPFR values at 300K
│   └── ...                    # Additional datasets
├── 000002/
└── ...
```

For optimal performance with >10GB files:
- Use Parquet or CSV indices for fast lookups
- Enable HDF5 chunking and compression
- Access molecules individually (never load the full dataset)

---

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{rpfr_gui,
  author = {Andren, Simon},
  title = {RPFR-GUI: Isotope Fractionation Analysis Tool},
  year = {2024},
  url = {https://github.com/simonandren/isotope-informed-crn}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests and linting (`uv run pytest && uv run ruff check`)
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

---

## Support

For bugs, feature requests, or questions:
- **Issues**: [GitHub Issues](https://github.com/simonandren/isotope-informed-crn/issues)
- **Discussions**: [GitHub Discussions](https://github.com/simonandren/isotope-informed-crn/discussions)

---

## Acknowledgments

- Built with [RDKit](https://www.rdkit.org/) for chemoinformatics
- Graph analysis powered by [NetworkX](https://networkx.org/)
- Fast dependency management by [uv](https://github.com/astral-sh/uv)
