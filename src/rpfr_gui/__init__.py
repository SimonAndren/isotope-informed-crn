"""RPFR-GUI: Isotope Fractionation Analysis Tool.

A high-performance, modular Python library for analyzing Reduced Partition
Function Ratios (RPFR) in molecular systems with lazy loading for large datasets.
"""

__version__ = "0.1.0"

from rpfr_gui.data import AbstractProvider, ChemistryResolver, H5Provider
from rpfr_gui.domain import IsotopeGraph

__all__ = [
    "ChemistryResolver",
    "AbstractProvider",
    "H5Provider",
    "IsotopeGraph",
]
