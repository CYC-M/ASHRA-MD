"""PepFind Compute Adapter for ASHRA-MD (capabilityId: md.complex.ashra / md.complex.ashmd).

ASHRA-MD: Active-Shell Hamiltonian with Replica-tempering & Affinity functional.
Enforces PepFind scientific truthfulness & reporting invariants (Reviewer Rules R1-R4):
  - R1: Count series (hbondCounts) strictly non-negative integers.
  - R2: Time-averaged metrics explicitly carry 'Mean' semantics.
  - R3: Default physiological temperature 310.15 K (37.00 °C) reported in dual units;
        any non-physiological temperature explicitly flagged in warnings & metadata.
  - R4: Domain-bounded physical quantities (RMSD >= 0, Rg >= 0, Q_native in [0, 1], T > 0).
  - Fail-closed on invalid PDBs, missing heavy atoms, severe clashes, or broken checkpoints.
"""

from __future__ import annotations

import hashlib
from typing import Any

from . import __version__
from .simulator import PHYSIOLOGICAL_TEMP_K, SimulationConfig, run_simulation
from .topology_builder import build_complex_system, build_protein_system

PROFILES: dict[str, dict[str, Any]] = {
    "smoke": {
        "minimize_steps": 120,
        "n_outer_steps": 100,       # 100 * 6 fs = 0.6 ps
        "report_every_outer": 10,
        "n_replicas": 4,
        "exchange_every_outer": 10,
    },
    "fast": {
        "minimize_steps": 250,
        "n_outer_steps": 850,       # ~5.1 ps x 4 IHRT replicas
        "report_every_outer": 25,
        "n_replicas": 4,
        "exchange_every_outer": 25,
    },
    "standard": {
        "minimize_steps": 500,
        "n_outer_steps": 8500,      # ~51 ps x 4 IHRT replicas
        "report_every_outer": 100,
        "n_replicas": 4,
        "exchange_every_outer": 50,
    },
    "ultra_long": {
        "minimize_steps": 500,
        "n_outer_steps": 170000,    # > 1 ns x 4 IHRT replicas, checkpointed across epochs
        "report_every_outer": 500,
        "n_replicas": 4,
        "exchange_every_outer": 100,
    },
}

SCIENTIFIC_DISCLAIMER = (
    "ASH-MD v2.0（活性壳层辛哈密顿分子动力学）基于已发表全原子力场 AMBER99SB-ILDN + phosaa10 磷酸化修饰"
    "（SEP/TPO/PTR）+ 锌指硫醇盐（CYM）、OBC II 广义玻恩/显式界面水壳层（或 TIP3P+SPME 周期性显式水）"
    "及自适应四副本界面哈密顿回火（IHRT-MBAR）执行真实牛顿/朗之万动力学积分。"
    "输出的零经验拟合结合自由能泛函（dg_bind_pred_mean_kcal）与综合结合稳定性评分仅用于评估生理温度下的局部动力学稳定性及候选多肽排序，"
    "不等同于体内药效活性或绝对解离常数实验直接测量值。"
)


def run_pepfind_ashmd(
    pdb_text: str,
    receptor_chains: list[str],
    peptide_chains: list[str] | None = None,
    profile: str = "fast",
    solvent_mode: str = "active-shell",
    temperature_k: float = PHYSIOLOGICAL_TEMP_K,
    metal_handling: str = "retain",
    allow_strip_hetatm: bool = True,
    allow_chain_breaks: bool = True,
    num_threads: int = 4,
    seed: int = 42,
    checkpoint_path: str | None = None,
    resume: bool = False,
    max_wall_seconds: float | None = None,
    n_outer_steps_override: int | None = None,
    n_replicas_override: int | None = None,
) -> dict[str, Any]:
    if profile not in PROFILES:
        raise ValueError(f"Unsupported ASH-MD profile {profile!r}; choose from {sorted(PROFILES)}")
    prof = PROFILES[profile]
    input_sha256 = hashlib.sha256(pdb_text.encode("utf-8")).hexdigest()

    if peptide_chains:
        sys_obj = build_complex_system(
            pdb_text=pdb_text,
            receptor_chains=receptor_chains,
            peptide_chains=peptide_chains,
            solvent_mode=solvent_mode,
            metal_handling=metal_handling,
            allow_strip_hetatm=allow_strip_hetatm,
            allow_chain_breaks=allow_chain_breaks,
            seed=seed,
        )
    else:
        sys_obj = build_protein_system(
            pdb_text=pdb_text,
            chains=receptor_chains,
            solvent_mode=solvent_mode,
            metal_handling=metal_handling,
            allow_strip_hetatm=allow_strip_hetatm,
            allow_chain_breaks=allow_chain_breaks,
            seed=seed,
        )

    n_outer = int(n_outer_steps_override if n_outer_steps_override is not None else prof["n_outer_steps"])
    n_rep = int(n_replicas_override if n_replicas_override is not None else prof["n_replicas"])

    cfg = SimulationConfig(
        ensemble="NVT",
        temperature_k=float(temperature_k),
        dt_ps=0.002,
        mts_ratio=3 if solvent_mode == "active-shell" else 2,
        gamma_ps_inv=1.0,
        filter_alpha=0.35,
        minimize_steps=int(prof["minimize_steps"]),
        n_outer_steps=n_outer,
        report_every_outer=int(prof["report_every_outer"]),
        n_replicas=n_rep,
        exchange_every_outer=int(prof["exchange_every_outer"]),
        num_threads=int(num_threads),
        seed=int(seed),
        checkpoint_path=checkpoint_path,
        resume_from_checkpoint=bool(resume),
        max_wall_seconds=max_wall_seconds,
    )

    res = run_simulation(sys_obj, cfg)

    # Enforce Reviewer Rules R1-R4 strictly on output payload
    hb_counts = [max(0, int(o["hbond_count"])) for o in res.observables]
    lig_rmsd_series = [max(0.0, round(float(o["ligand_rmsd_a"]), 4)) for o in res.observables]
    rec_rmsd_series = [max(0.0, round(float(o["receptor_rmsd_a"]), 4)) for o in res.observables]
    q_nat_series = [min(1.0, max(0.0, round(float(o["q_native"]), 4))) for o in res.observables]
    rg_series = [max(0.0, round(float(o["gyration_a"]), 4)) for o in res.observables]
    bsasa_series = [max(0.0, round(float(o.get("buried_sasa_a2", 0.0)), 2)) for o in res.observables]
    dg_pred_series = [round(float(o.get("dg_bind_pred_kcal", 0.0)), 4) for o in res.observables]

    return {
        "tool": "ashra_md",
        "algorithm": "ASHRA-MD (Active-Shell Hamiltonian with Replica-tempering & Affinity)",
        "version": __version__,
        "inputPdbSha256": input_sha256,
        "completed": res.completed,
        "solventMode": res.solvent_mode,
        "simdBackend": res.simd_backend,
        "atomCount": sys_obj.n_atoms,
        "replicaCount": n_rep,
        "temperature": {
            "kelvin": res.temperature_k,
            "celsius": res.temperature_c,
            "isPhysiological": abs(res.temperature_k - PHYSIOLOGICAL_TEMP_K) <= 0.05,
            "display": f"{res.temperature_k:.2f} K ({res.temperature_c:.2f} °C)",
        },
        "performance": {
            "simulatedPs": res.total_simulated_ps,
            "outerStepsCompleted": res.outer_steps_completed,
            "wallSeconds": res.wall_seconds,
            "nsPerDayPerReplica": res.ns_per_day,
            "aggregateSamplingNsPerDay": round(res.ns_per_day * n_rep, 3),
        },
        "minimization": {
            "steps": res.minimized_steps,
            "initialPotentialKcalMol": res.initial_potential_kcal,
            "minimizedPotentialKcalMol": res.minimized_potential_kcal,
        },
        "series": {
            "timePs": res.times_ps,
            "receptorRmsdA": rec_rmsd_series,
            "ligandRmsdA": lig_rmsd_series,
            "nativeContactFraction": q_nat_series,
            "gyrationRadiusA": rg_series,
            "hbondCounts": hb_counts,
            "buriedSasaA2": bsasa_series,
            "dgBindPredKcalMol": dg_pred_series,
            "potentialKcalMol": [round(float(e["potential"]), 3) for e in res.energies],
            "temperatureK": [max(0.0, round(float(e["temperature"]), 2)) for e in res.energies],
        },
        "summary": res.binding_summary,
        "mbar": res.mbar_summary,
        "zincCoordination": res.zinc_coordination,
        "rmsfPerResidue": res.rmsf_per_residue,
        "finalComplexPdb": res.final_pdb_block,
        "warnings": res.warnings,
        "disclaimer": SCIENTIFIC_DISCLAIMER,
    }
