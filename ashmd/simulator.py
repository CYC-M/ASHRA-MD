"""High-Level Dual-Mode & Ultra-Long Epoch ASH-MD Simulator (v2.0)."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
import pathlib
import time
from typing import Any

import numpy as np

from .analysis import TrajectoryAnalyzer, compute_ca_rmsf, summarize_binding_and_mbar
from .bridge import AshmdCoreLibrary, simd_backend
from .topology_builder import BuiltSystem, ROLE_PEPTIDE

PHYSIOLOGICAL_TEMP_K = 310.15


@dataclass
class SimulationConfig:
    ensemble: str = "NVT"                 # 'NVT' or 'NVE'
    temperature_k: float = PHYSIOLOGICAL_TEMP_K
    dt_ps: float = 0.002                  # 2.0 fs inner step (with SETTLE + X-H RATTLE)
    mts_ratio: int = 3                    # FC-BAOAB outer step = 3 * 2.0 = 6.0 fs for slow GB/PME-recip!
    gamma_ps_inv: float = 1.0             # Langevin friction [1/ps] (ignored if ensemble == 'NVE')
    filter_alpha: float = 0.35            # FC-BAOAB topological covalent filter strength (0.0 disables)

    minimize_steps: int = 250             # Pre-MD clash relaxation steps (0 to skip)
    minimize_fmax: float = 12.0           # Force tolerance [kcal/(mol*A)]

    n_outer_steps: int = 500              # Total outer steps to integrate
    report_every_outer: int = 10          # Trajectory frame / observable sampling interval

    # Original Pillar 3: IHRT 4-Replica Synchronous Tempering (designed for Instabox 4-vCPU!)
    n_replicas: int = 1
    lambda_schedule: list[float] | None = None
    mu_schedule: list[float] | None = None
    softcore_schedule: list[float] | None = None
    exchange_every_outer: int = 20

    num_threads: int = 4                  # Matches Instabox 4 vCPUs per node
    seed: int = 42

    # Ultra-Long CPU Epoch & Atomic Checkpointing (handles Instabox 8h cooldown seamlessly!)
    checkpoint_path: str | os.PathLike | None = None
    checkpoint_every_frames: int = 25
    resume_from_checkpoint: bool = False
    max_wall_seconds: float | None = None

    def resolved_schedules(self) -> tuple[list[float], list[float], list[float]]:
        M = max(1, int(self.n_replicas))
        if M == 1:
            return [1.0], [1.0], [0.0]
        if self.lambda_schedule is not None:
            lam = list(self.lambda_schedule)
        elif M == 4:
            lam = [1.00, 0.93, 0.86, 0.78]
        else:
            lam = np.geomspace(1.0, 0.78, M).tolist()

        mu = list(self.mu_schedule) if self.mu_schedule is not None else [math.sqrt(x) for x in lam]
        sc = (
            list(self.softcore_schedule)
            if self.softcore_schedule is not None
            else [round(0.05 * (1.0 - x), 6) for x in lam]
        )
        return lam, mu, sc


@dataclass
class SimulationResult:
    completed: bool
    outer_steps_completed: int
    total_simulated_ps: float
    wall_seconds: float
    ns_per_day: float
    simd_backend: str
    solvent_mode: str
    temperature_k: float
    temperature_c: float

    minimized_steps: int
    initial_potential_kcal: float
    minimized_potential_kcal: float

    times_ps: list[float]
    energies: list[dict[str, float]]
    observables: list[dict[str, Any]]
    rmsf_per_residue: list[dict[str, Any]]
    zinc_coordination: list[dict[str, Any]]
    mbar_summary: dict[str, Any]
    binding_summary: dict[str, Any]

    final_coords: np.ndarray
    final_velocities: np.ndarray
    final_pdb_block: str
    warnings: list[str] = field(default_factory=list)


def _trajectory_sidecar_path(chk_path: str | os.PathLike) -> pathlib.Path:
    return pathlib.Path(f"{chk_path}.history.json")


def _populate_peptide_energetics(
    core: AshmdCoreLibrary,
    analyzer: TrajectoryAnalyzer,
    coords: np.ndarray,
    temp_k: float,
    obs_dict: dict[str, Any],
) -> None:
    d2_legacy = core.eval_binding_dg(coords, temp_k, eps_in_gbsa=2.0)
    d4_pol = core.eval_binding_dg(coords, temp_k, eps_in_gbsa=4.0)
    sasa_terms, sf, sb = core.eval_binding_rd_gbsa_sasa(
        coords, temp_k, cav_gamma=0.0072, disp_scale=1.0, return_atom_sasa=True
    )
    obs_dict.update(
        analyzer.eval_physical_binding_frame(coords, obs_dict, d2_legacy, d4_pol, sasa_terms, sf, sb)
    )


def run_simulation(sys_obj: BuiltSystem, cfg: SimulationConfig) -> SimulationResult:
    """Execute ASH-MD simulation (from scratch or seamlessly resumed from atomic SHA-256 checkpoint)."""
    if cfg.temperature_k <= 0.0:
        raise ValueError(f"Temperature must be strictly positive (got {cfg.temperature_k})")
    if cfg.dt_ps <= 0.0 or cfg.mts_ratio < 1:
        raise ValueError("Invalid timestep or mts_ratio")

    t0_wall = time.perf_counter()
    gamma = 0.0 if cfg.ensemble.upper() == "NVE" else float(cfg.gamma_ps_inv)
    lam_sched, mu_sched, sc_sched = cfg.resolved_schedules()
    has_peptide = any(am.role == ROLE_PEPTIDE for am in sys_obj.atoms)

    warnings = list(sys_obj.warnings)
    if abs(cfg.temperature_k - PHYSIOLOGICAL_TEMP_K) > 0.05:
        warnings.append(
            f"Non-physiological simulation temperature explicitly requested: "
            f"{cfg.temperature_k:.2f} K ({cfg.temperature_k - 273.15:.2f} °C); "
            f"physiological baseline is 310.15 K (37.00 °C)."
        )

    with AshmdCoreLibrary(num_threads=cfg.num_threads) as core:
        core.upload_system(sys_obj, filter_alpha=cfg.filter_alpha)

        times_ps: list[float] = []
        energies_log: list[dict[str, float]] = []
        obs_log: list[dict[str, Any]] = []
        saved_solute_frames: list[np.ndarray] = []

        start_outer = 0
        min_steps_done = 0
        init_pot = 0.0
        min_pot = 0.0
        ref_coords_for_analysis = sys_obj.coords.copy()

        if cfg.resume_from_checkpoint and cfg.checkpoint_path and pathlib.Path(cfg.checkpoint_path).is_file():
            start_outer = core.load_checkpoint(cfg.checkpoint_path)
            sidecar = _trajectory_sidecar_path(cfg.checkpoint_path)
            if sidecar.is_file():
                hist = json.loads(sidecar.read_text(encoding="utf-8"))
                times_ps = hist["times_ps"]
                energies_log = hist["energies"]
                obs_log = hist["observables"]
                min_steps_done = hist["minimized_steps"]
                init_pot = hist["initial_potential_kcal"]
                min_pot = hist["minimized_potential_kcal"]
                ref_coords_for_analysis = np.array(hist["ref_coords"], dtype=np.float64)
                if "lambda_schedule" in hist and isinstance(hist["lambda_schedule"], list):
                    lam_sched = [float(x) for x in hist["lambda_schedule"]]
            analyzer = TrajectoryAnalyzer(sys_obj, ref_coords=ref_coords_for_analysis)
        else:
            init_t = cfg.temperature_k if cfg.minimize_steps <= 0 else 0.0
            core.init_replicas(
                sys_obj.coords,
                n_replicas=cfg.n_replicas,
                lambda_inter=lam_sched,
                mu_torsion=mu_sched,
                softcore_alpha=sc_sched,
                seed=cfg.seed,
                init_temp_k=init_t,
            )
            _, _, e0 = core.get_state(0)
            init_pot = e0["potential"]

            if cfg.minimize_steps > 0:
                min_steps_done = core.minimize(
                    max_steps=cfg.minimize_steps,
                    fmax_tol=cfg.minimize_fmax,
                    init_temp_after_k=cfg.temperature_k,
                )
            min_coords, _, emin = core.get_state(0)
            min_pot = emin["potential"]

            # Adaptive IHRT lambda-spacing for large multi-domain peptide interfaces (|E_inter| > 125 kcal/mol)
            if has_peptide and cfg.n_replicas > 1 and cfg.lambda_schedule is None:
                e_int_mag = max(20.0, abs(float(emin["raw_interface_nonb"])))
                if e_int_mag > 125.0:
                    d_lam = max(0.035, min(0.0733, 9.0 / e_int_mag))
                    lam_min = max(0.78, 1.0 - d_lam * (cfg.n_replicas - 1))
                    lam_sched = [round(float(x), 5) for x in np.linspace(1.0, lam_min, cfg.n_replicas)]
                    mu_sched = (
                        list(cfg.mu_schedule)
                        if cfg.mu_schedule is not None
                        else [round(math.sqrt(x), 6) for x in lam_sched]
                    )
                    sc_sched = (
                        list(cfg.softcore_schedule)
                        if cfg.softcore_schedule is not None
                        else [round(0.05 * (1.0 - x), 6) for x in lam_sched]
                    )
                    core.set_replica_tempering(lam_sched, mu_sched, sc_sched, cfg.temperature_k)

            ref_coords_for_analysis = min_coords.copy()
            analyzer = TrajectoryAnalyzer(sys_obj, ref_coords=ref_coords_for_analysis)

            # Record t = 0 frame
            times_ps.append(0.0)
            energies_log.append(emin)
            f0_obs = analyzer.analyze_frame(min_coords)
            if has_peptide:
                _populate_peptide_energetics(core, analyzer, min_coords, cfg.temperature_k, f0_obs)
            obs_log.append(f0_obs)
            saved_solute_frames.append(min_coords)

        dt_outer_ps = cfg.dt_ps * cfg.mts_ratio
        chunk = max(1, int(cfg.report_every_outer))
        cur_outer = start_outer
        frames_since_chk = 0
        completed = True

        while cur_outer < cfg.n_outer_steps:
            if cfg.max_wall_seconds is not None and (time.perf_counter() - t0_wall) >= cfg.max_wall_seconds:
                completed = False
                break

            step_batch = min(chunk, cfg.n_outer_steps - cur_outer)
            core.run_steps(
                n_outer_steps=step_batch,
                dt_ps=cfg.dt_ps,
                mts_ratio=cfg.mts_ratio,
                gamma_ps_inv=gamma,
                target_temp_k=cfg.temperature_k,
                exchange_every_outer=cfg.exchange_every_outer,
                record_mbar=True,
            )
            cur_outer += step_batch

            c_now, _, e_now = core.get_state(0)
            if (
                not math.isfinite(e_now["total"])
                or not np.all(np.isfinite(c_now))
                or float(np.max(np.abs(c_now))) > 1.0e5
                or e_now["temperature"] > 1.0e5
            ):
                raise RuntimeError(f"Numerical instability detected at outer step {cur_outer}: non-physical state")

            times_ps.append(round(cur_outer * dt_outer_ps, 6))
            energies_log.append(e_now)
            fn_obs = analyzer.analyze_frame(c_now)
            if has_peptide:
                _populate_peptide_energetics(core, analyzer, c_now, cfg.temperature_k, fn_obs)
            obs_log.append(fn_obs)
            saved_solute_frames.append(c_now)
            frames_since_chk += 1

            if cfg.checkpoint_path and frames_since_chk >= max(1, cfg.checkpoint_every_frames):
                core.save_checkpoint(cfg.checkpoint_path)
                _save_history_sidecar(
                    cfg.checkpoint_path, times_ps, energies_log, obs_log,
                    min_steps_done, init_pot, min_pot, ref_coords_for_analysis, lam_sched
                )
                frames_since_chk = 0

        if cfg.checkpoint_path:
            core.save_checkpoint(cfg.checkpoint_path)
            _save_history_sidecar(
                cfg.checkpoint_path, times_ps, energies_log, obs_log,
                min_steps_done, init_pot, min_pot, ref_coords_for_analysis, lam_sched
            )

        final_coords, final_vel, _ = core.get_state(0)
        mbar_raw = core.get_mbar_summary()

    wall_s = max(1e-9, time.perf_counter() - t0_wall)
    sim_ps_this_epoch = max(0.0, (cur_outer - start_outer) * dt_outer_ps)
    ns_per_day = (sim_ps_this_epoch * 86.4) / wall_s

    rmsf_table = compute_ca_rmsf(sys_obj, saved_solute_frames, ref_coords_for_analysis)

    # Summarize Zn2+ donor coordination distances across trajectory
    zn_summary: list[dict[str, Any]] = []
    if sys_obj.zinc_sites:
        donor_flat_idx = 0
        for zs in sys_obj.zinc_sites:
            site_donors_out = []
            for d in zs["donors"]:
                series = [
                    frame_o["zn_donor_dists_a"][donor_flat_idx]
                    for frame_o in obs_log
                    if donor_flat_idx < len(frame_o["zn_donor_dists_a"])
                ]
                arr = np.array(series, dtype=np.float64) if series else np.array([d["init_dist"]])
                site_donors_out.append({
                    "donor": f"{d['res']}:{d['atom']}",
                    "initialA": round(float(d["init_dist"]), 4),
                    "meanA": round(float(np.mean(arr)), 4),
                    "stdA": round(float(np.std(arr)), 4),
                    "minA": round(float(np.min(arr)), 4),
                    "maxA": round(float(np.max(arr)), 4),
                })
                donor_flat_idx += 1
            zn_summary.append({
                "chainId": zs["chain_id"],
                "resSeq": zs["res_seq"],
                "donors": site_donors_out,
            })

    mbar_clean = {
        "n_samples": mbar_raw["n_samples"],
        "lambda_schedule": [round(float(x), 5) for x in lam_sched],
        "f_dimensionless": [round(float(x), 6) for x in mbar_raw["f_dimensionless"]],
        "swap_attempts": mbar_raw["swap_attempts"],
        "swap_accepts": mbar_raw["swap_accepts"],
        "acceptance_rates": [
            round(acc / att, 4) if att > 0 else 0.0
            for att, acc in zip(mbar_raw["swap_attempts"], mbar_raw["swap_accepts"])
        ],
    }

    binding_summary = summarize_binding_and_mbar(
        obs_log, energies_log, mbar_clean, lam_sched, cfg.temperature_k, has_peptide
    )

    return SimulationResult(
        completed=completed,
        outer_steps_completed=cur_outer,
        total_simulated_ps=round(cur_outer * dt_outer_ps, 6),
        wall_seconds=round(wall_s, 4),
        ns_per_day=round(ns_per_day, 3),
        simd_backend=simd_backend(),
        solvent_mode=sys_obj.solvent_mode,
        temperature_k=round(cfg.temperature_k, 2),
        temperature_c=round(cfg.temperature_k - 273.15, 2),
        minimized_steps=min_steps_done,
        initial_potential_kcal=round(init_pot, 4),
        minimized_potential_kcal=round(min_pot, 4),
        times_ps=times_ps,
        energies=energies_log,
        observables=obs_log,
        rmsf_per_residue=rmsf_table,
        zinc_coordination=zn_summary,
        mbar_summary=mbar_clean,
        binding_summary=binding_summary,
        final_coords=final_coords,
        final_velocities=final_vel,
        final_pdb_block=sys_obj.to_pdb_block(final_coords, include_solvent=False),
        warnings=warnings,
    )


def _save_history_sidecar(
    chk_path: str | os.PathLike,
    times_ps: list[float],
    energies: list[dict[str, float]],
    observables: list[dict[str, Any]],
    minimized_steps: int,
    init_pot: float,
    min_pot: float,
    ref_coords: np.ndarray,
    lambda_schedule: list[float],
) -> None:
    sidecar = _trajectory_sidecar_path(chk_path)
    tmp = sidecar.with_suffix(".tmp")
    payload = {
        "times_ps": times_ps,
        "energies": energies,
        "observables": observables,
        "minimized_steps": minimized_steps,
        "initial_potential_kcal": init_pot,
        "minimized_potential_kcal": min_pot,
        "ref_coords": ref_coords.tolist(),
        "lambda_schedule": [float(x) for x in lambda_schedule],
    }
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, sidecar)
