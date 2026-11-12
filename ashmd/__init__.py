"""ASH-MD: Active-Shell Symplectic Hamiltonian Molecular Dynamics Engine for PepFind.

Original multi-layer CPU molecular dynamics algorithm & C++17/Python package:
  1. Pillar 1 (FC-BAOAB): Deterministic Resonance-Free Filtered-Coordinate Split Langevin Integrator
     with Q31.32 associative fixed-point force reduction & Philox-4x32 counter PRNG.
  2. Pillar 2 (Dual-Mode ASH / SPME): Unified AMBER99SB-ILDN all-atom Hamiltonian supporting both
     gold-standard Periodic Explicit TIP3P + SPME and Active-Shell Explicit Interface Water + OBC2/ACE.
  3. Pillar 3 (IHRT-MBAR): 4-Replica Synchronous Interface Hamiltonian Replica Tempering + Self-Consistent
     MBAR unbiased ensemble & binding funnel free-energy estimation, with SHA-256 atomic checkpoints.
"""

__version__ = "2.0.0"

from .bridge import AshmdCoreLibrary, simd_backend
from .topology_builder import BuiltSystem, TopologyBuildError, build_complex_system, build_protein_system
from .simulator import SimulationConfig, SimulationResult, run_simulation
from .pepfind_adapter import run_pepfind_ashmd

__all__ = [
    "__version__",
    "AshmdCoreLibrary",
    "simd_backend",
    "BuiltSystem",
    "TopologyBuildError",
    "build_complex_system",
    "build_protein_system",
    "SimulationConfig",
    "SimulationResult",
    "run_simulation",
    "run_pepfind_ashmd",
]
