"""Command-line interface for ASH-MD (`python -m ashmd`).

Supports ultra-long multi-day CPU simulations via atomic SHA-256 epoch checkpointing
(`--checkpoint` + `--resume` + `--max-wall-seconds`), tailored for 4-vCPU Instabox nodes.
"""

from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import sys

from .pepfind_adapter import PROFILES, run_pepfind_ashmd
from .simulator import PHYSIOLOGICAL_TEMP_K


def _read_pdb_text(path: pathlib.Path) -> str:
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    return path.read_text(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ashmd",
        description=(
            "ASH-MD: Active-Shell Symplectic Hamiltonian Molecular Dynamics "
            "(CPU-native C++17 AMBER99SB-ILDN + FC-BAOAB + IHRT-MBAR)"
        ),
    )
    parser.add_argument("--pdb", required=True, help="Input PDB or PDB.GZ file path")
    parser.add_argument(
        "--receptor-chains",
        required=True,
        help="Comma-separated receptor chain IDs (e.g. A or A,B)",
    )
    parser.add_argument(
        "--peptide-chains",
        default="",
        help="Comma-separated peptide chain IDs (omit for apo protein / metalloenzyme MD)",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES.keys()),
        default="fast",
        help="Simulation length & sampling profile (default: fast)",
    )
    parser.add_argument(
        "--solvent-mode",
        choices=["active-shell", "explicit-pme"],
        default="active-shell",
        help="Hamiltonian solvent mode (default: active-shell)",
    )
    parser.add_argument(
        "--temperature-k",
        type=float,
        default=PHYSIOLOGICAL_TEMP_K,
        help=f"Target temperature in Kelvin (default physiological: {PHYSIOLOGICAL_TEMP_K} K / 37.00 °C)",
    )
    parser.add_argument(
        "--metal-handling",
        choices=["strip", "retain"],
        default="strip",
        help="Whether to retain structural ZN/NA/CL ions (default: strip)",
    )
    parser.add_argument(
        "--strict-hetatm",
        action="store_true",
        help="Fail immediately on any unapproved HETATM instead of stripping non-retained ligands",
    )
    parser.add_argument(
        "--strict-chains",
        action="store_true",
        help="Fail immediately on internal backbone breaks (>2.30 A) instead of capping termini",
    )
    parser.add_argument("--threads", type=int, default=4, help="CPU worker threads (default: 4)")
    parser.add_argument("--replicas", type=int, default=None, help="Override IHRT replica count")
    parser.add_argument("--outer-steps", type=int, default=None, help="Override total FC-BAOAB outer steps")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic Philox-4x32 global seed")
    parser.add_argument("--checkpoint", default=None, help="Atomic SHA-256 checkpoint file path (.chk)")
    parser.add_argument("--resume", action="store_true", help="Resume seamlessly from existing --checkpoint")
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=None,
        help="Gracefully checkpoint & exit after wall budget (e.g. 27000s for 8h Instabox cooldown)",
    )
    parser.add_argument("--out-json", default=None, help="Write structured PepFind JSON report to path")
    parser.add_argument("--out-pdb", default=None, help="Write final physical replica-0 PDB coordinates to path")

    args = parser.parse_args(argv)
    pdb_path = pathlib.Path(args.pdb)
    if not pdb_path.is_file():
        print(f"[ashmd] ERROR: Input PDB not found: {pdb_path}", file=sys.stderr)
        return 2

    rec_chains = [c.strip() for c in args.receptor_chains.split(",") if c.strip()]
    pep_chains = [c.strip() for c in args.peptide_chains.split(",") if c.strip()] or None

    pdb_text = _read_pdb_text(pdb_path)
    result = run_pepfind_ashmd(
        pdb_text=pdb_text,
        receptor_chains=rec_chains,
        peptide_chains=pep_chains,
        profile=args.profile,
        solvent_mode=args.solvent_mode,
        temperature_k=args.temperature_k,
        metal_handling=args.metal_handling,
        allow_strip_hetatm=not args.strict_hetatm,
        allow_chain_breaks=not args.strict_chains,
        num_threads=args.threads,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
        resume=args.resume,
        max_wall_seconds=args.max_wall_seconds,
        n_outer_steps_override=args.outer_steps,
        n_replicas_override=args.replicas,
    )

    if args.out_pdb:
        pathlib.Path(args.out_pdb).write_text(result["finalComplexPdb"], encoding="utf-8")

    if args.out_json:
        pathlib.Path(args.out_json).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        summary_view = {
            k: v for k, v in result.items() if k not in ("finalComplexPdb", "rmsfPerResidue")
        }
        print(json.dumps(summary_view, ensure_ascii=False, indent=2))

    return 0 if result["completed"] else 10


if __name__ == "__main__":
    raise SystemExit(main())
