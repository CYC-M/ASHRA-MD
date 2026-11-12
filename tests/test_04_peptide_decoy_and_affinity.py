"""Test Suite 04: Multi-Class Decoy & 0.00 A Hot-Spot Alanine Mutant Discrimination (Criterion 1),
4-Replica IHRT-MBAR Sampling & Funnel Ablation (Original Pillar 3),
and 100% (14/14) Experimental Kd / dG Zero-Fitted Physical Affinity Benchmark (Criterion 3).
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ashmd.simulator import PHYSIOLOGICAL_TEMP_K, SimulationConfig, run_simulation
from ashmd.topology_builder import TopologyBuildError, build_complex_system

CALIB_DIR = ROOT / "tests" / "fixtures" / "calib"

# Exact biological receptor-peptide contacting chains across all 14 real RCSB PDBs in calib_dg.json
CALIB_CHAIN_MAP: dict[str, tuple[str, str]] = {
    "2BBM": ("A", "B"),
    "1CDL": ("A", "E"),
    "1MXE": ("A", "E"),
    "2F3Y": ("A", "B"),
    "2VAY": ("A", "B"),
    "2FLU": ("X", "P"),
    "1YCR": ("A", "B"),
    "1BXL": ("A", "B"),
    "1BE9": ("A", "B"),
    "1N7F": ("A", "C"),
    "2I0L": ("A", "C"),
    "1QJA": ("B", "Q"),
    "1JYR": ("A", "L"),
    "1G3F": ("A", "B"),
}


def make_inpocket_decoy_pdb(pdb_text: str, rec_chain: str, pep_chain: str, angle_deg: float = 180.0) -> str:
    """Class A Decoy: Clash-free in-pocket false-positive pose rotated 180 deg around pocket outward normal."""
    rec_heavy_xyz: list[list[float]] = []
    pep_lines_idx: list[int] = []
    pep_heavy_xyz: list[list[float]] = []
    lines: list[str] = []

    for ln in pdb_text.splitlines():
        if ln.startswith("ENDMDL"):
            break
        if ln.startswith(("ATOM  ", "HETATM")):
            ch = ln[21].strip() or "A"
            if ch not in (rec_chain, pep_chain):
                continue
            rname = ln[17:20].strip()
            if ln.startswith("HETATM") and rname in ("HOH", "WAT"):
                continue
            aname = ln[12:16].strip()
            elem = ln[76:78].strip() if len(ln) >= 78 else ""
            is_h = (elem == "H") or aname.startswith("H") or (len(aname) >= 2 and aname[0].isdigit() and aname[1] == "H")
            if is_h:
                continue
            x, y, z = float(ln[30:38]), float(ln[38:46]), float(ln[46:54])
            idx = len(lines)
            lines.append(ln)
            if ch == rec_chain:
                rec_heavy_xyz.append([x, y, z])
            else:
                pep_lines_idx.append(idx)
                pep_heavy_xyz.append([x, y, z])

    R_arr = np.array(rec_heavy_xyz, dtype=np.float64)
    P_arr = np.array(pep_heavy_xyz, dtype=np.float64)
    cP = P_arr.mean(axis=0)

    dmat = np.linalg.norm(R_arr[:, None, :] - P_arr[None, :, :], axis=-1)
    pocket_mask = np.min(dmat, axis=1) <= 6.5
    c_pock = R_arr[pocket_mask].mean(axis=0) if np.any(pocket_mask) else R_arr.mean(axis=0)
    n_vec = cP - c_pock
    n_vec /= max(float(np.linalg.norm(n_vec)), 1e-12)

    th = math.radians(angle_deg)
    K = np.array([
        [0.0, -n_vec[2], n_vec[1]],
        [n_vec[2], 0.0, -n_vec[0]],
        [-n_vec[1], n_vec[0], 0.0],
    ])
    Rot = np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)
    P_rot = (P_arr - cP) @ Rot.T + cP

    best_s, best_min_d = 0.0, -1.0
    P_final = None
    for s in np.arange(0.0, 14.0, 0.35):
        P_cand = P_rot + s * n_vec
        min_d = float(np.min(np.linalg.norm(R_arr[:, None, :] - P_cand[None, :, :], axis=-1)))
        if min_d > best_min_d:
            best_min_d = min_d
            best_s = s
        if min_d >= 2.40:
            P_final = P_cand
            break
    if P_final is None:
        P_final = P_rot + best_s * n_vec

    for k, l_idx in enumerate(pep_lines_idx):
        ln = lines[l_idx]
        x, y, z = P_final[k]
        lines[l_idx] = f"{ln[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{ln[54:]}"
    lines.append("END")
    return "\n".join(lines) + "\n"


def make_axial_shift_decoy_pdb(
    pdb_text: str, rec_chain: str, pep_chain: str, shift_a: float = 3.5, tilt_deg: float = 35.0
) -> str:
    """Class B Decoy: In-pocket 1-residue axial groove register slip (3.5 A) + 35 deg pocket-normal tilt."""
    rec_heavy_xyz: list[list[float]] = []
    pep_lines_idx: list[int] = []
    pep_heavy_xyz: list[list[float]] = []
    pep_ca_xyz: list[tuple[int, np.ndarray]] = []
    lines: list[str] = []

    for ln in pdb_text.splitlines():
        if ln.startswith("ENDMDL"):
            break
        if ln.startswith(("ATOM  ", "HETATM")):
            ch = ln[21].strip() or "A"
            if ch not in (rec_chain, pep_chain):
                continue
            rname = ln[17:20].strip()
            if ln.startswith("HETATM") and rname in ("HOH", "WAT"):
                continue
            aname = ln[12:16].strip()
            elem = ln[76:78].strip() if len(ln) >= 78 else ""
            is_h = (elem == "H") or aname.startswith("H") or (len(aname) >= 2 and aname[0].isdigit() and aname[1] == "H")
            if is_h:
                continue
            x, y, z = float(ln[30:38]), float(ln[38:46]), float(ln[46:54])
            idx = len(lines)
            lines.append(ln)
            if ch == rec_chain:
                rec_heavy_xyz.append([x, y, z])
            else:
                pep_lines_idx.append(idx)
                pep_heavy_xyz.append([x, y, z])
                if aname == "CA":
                    pep_ca_xyz.append((int(ln[22:26]), np.array([x, y, z], dtype=np.float64)))

    R_arr = np.array(rec_heavy_xyz, dtype=np.float64)
    P_arr = np.array(pep_heavy_xyz, dtype=np.float64)
    cP = P_arr.mean(axis=0)

    _, _, Vt = np.linalg.svd(P_arr - cP, full_matrices=False)
    u_axis = Vt[0].copy()
    if len(pep_ca_xyz) >= 2:
        pep_ca_xyz.sort(key=lambda item: item[0])
        if float(np.dot(pep_ca_xyz[-1][1] - pep_ca_xyz[0][1], u_axis)) < 0.0:
            u_axis = -u_axis
    u_axis /= max(float(np.linalg.norm(u_axis)), 1e-12)

    dmat = np.linalg.norm(R_arr[:, None, :] - P_arr[None, :, :], axis=-1)
    pocket_mask = np.min(dmat, axis=1) <= 6.5
    c_pock = R_arr[pocket_mask].mean(axis=0) if np.any(pocket_mask) else R_arr.mean(axis=0)
    n_vec = cP - c_pock
    n_vec /= max(float(np.linalg.norm(n_vec)), 1e-12)

    th = math.radians(tilt_deg)
    K = np.array([
        [0.0, -n_vec[2], n_vec[1]],
        [n_vec[2], 0.0, -n_vec[0]],
        [-n_vec[1], n_vec[0], 0.0],
    ])
    Rot = np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)
    P_shifted = (P_arr - cP) @ Rot.T + cP + shift_a * u_axis

    best_s, best_min_d = 0.0, -1.0
    P_final = None
    for s in np.arange(0.0, 12.0, 0.30):
        P_cand = P_shifted + s * n_vec
        min_d = float(np.min(np.linalg.norm(R_arr[:, None, :] - P_cand[None, :, :], axis=-1)))
        if min_d > best_min_d:
            best_min_d = min_d
            best_s = s
        if min_d >= 2.25:
            P_final = P_cand
            break
    if P_final is None:
        P_final = P_shifted + best_s * n_vec

    for k, l_idx in enumerate(pep_lines_idx):
        ln = lines[l_idx]
        x, y, z = P_final[k]
        lines[l_idx] = f"{ln[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{ln[54:]}"
    lines.append("END")
    return "\n".join(lines) + "\n"


def make_hotspot_alanine_mutant_pdb(
    pdb_text: str, rec_chain: str, pep_chain: str, mutate_res_seqs: list[int]
) -> str:
    """Class C Decoy: Exact 0.00 A crystal-backbone Hot-Spot Alanine Loss-of-Function Mutant."""
    mut_set = set(mutate_res_seqs)
    keep_atoms = {"N", "CA", "C", "O", "OXT", "CB"}
    out_lines: list[str] = []
    for ln in pdb_text.splitlines():
        if ln.startswith("ENDMDL"):
            break
        if ln.startswith(("ATOM  ", "HETATM")):
            ch = ln[21].strip() or "A"
            if ch not in (rec_chain, pep_chain):
                continue
            rname = ln[17:20].strip()
            if ln.startswith("HETATM") and rname in ("HOH", "WAT"):
                continue
            aname = ln[12:16].strip()
            elem = ln[76:78].strip() if len(ln) >= 78 else ""
            is_h = (elem == "H") or aname.startswith("H") or (len(aname) >= 2 and aname[0].isdigit() and aname[1] == "H")
            if is_h:
                continue
            rseq = int(ln[22:26])
            if ch == pep_chain and rseq in mut_set:
                if aname not in keep_atoms:
                    continue
                ln = "ATOM  " + ln[6:17] + "ALA" + ln[20:]
            out_lines.append(ln)
    out_lines.append("END")
    return "\n".join(out_lines) + "\n"


def spearman_exact_tie_aware(x: list[float], y: list[float]) -> float:
    """Exact Spearman rank correlation handling tied experimental dG values via midranks."""
    def midranks(vals: list[float]) -> np.ndarray:
        arr = np.asarray(vals, dtype=np.float64)
        order = np.argsort(arr)
        ranks = np.empty_like(arr)
        i = 0
        n = len(arr)
        while i < n:
            j = i
            while j + 1 < n and abs(arr[order[j + 1]] - arr[order[i]]) < 1e-12:
                j += 1
            avg_rank = 0.5 * (i + j) + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks

    rx = midranks(x)
    ry = midranks(y)
    return float(np.corrcoef(rx, ry)[0, 1])


class TestPeptideDecoyAndAffinity(unittest.TestCase):
    def test_01_multi_class_decoy_and_hotspot_mutant_discrimination_310_15k(self) -> None:
        """Criterion 1: 100% discrimination across Class A (180-deg), Class B (axial shift), and Class C (0.00 A Ala mutants)."""
        # 1. Class A (180-deg flip) & Class B (3.5 A register slip + 35-deg tilt) with 4-replica IHRT-MBAR
        pose_panel = [
            ("1YCR", "A", "B"),
            ("1G3F", "A", "B"),
            ("1N7F", "A", "C"),
            ("1BXL", "A", "B"),
        ]
        cfg_4rep = SimulationConfig(
            ensemble="NVT",
            temperature_k=PHYSIOLOGICAL_TEMP_K,
            dt_ps=0.002,
            mts_ratio=3,
            minimize_steps=110,
            n_outer_steps=100,
            report_every_outer=20,
            n_replicas=4,
            exchange_every_outer=20,
            num_threads=4,
            seed=42,
        )

        wt_cache: dict[str, dict] = {}
        for pid, rc, pc in pose_panel:
            pdb_txt = (CALIB_DIR / f"{pid}.pdb").read_text(encoding="utf-8")
            dec_a_txt = make_inpocket_decoy_pdb(pdb_txt, rc, pc, 180.0)
            dec_b_txt = make_axial_shift_decoy_pdb(pdb_txt, rc, pc, shift_a=3.5, tilt_deg=35.0)

            sys_nat = build_complex_system(
                pdb_txt, [rc], [pc], solvent_mode="active-shell",
                max_shell_waters=48, active_radius=11.0, buffer_radius=18.0, seed=42,
            )
            sys_a = build_complex_system(
                dec_a_txt, [rc], [pc], solvent_mode="active-shell",
                max_shell_waters=48, active_radius=11.0, buffer_radius=18.0, seed=42,
            )
            sys_b = build_complex_system(
                dec_b_txt, [rc], [pc], solvent_mode="active-shell",
                max_shell_waters=48, active_radius=11.0, buffer_radius=18.0, seed=42,
            )

            bn = run_simulation(sys_nat, cfg_4rep).binding_summary
            ba = run_simulation(sys_a, cfg_4rep).binding_summary
            bb = run_simulation(sys_b, cfg_4rep).binding_summary
            wt_cache[pid] = bn

            self.assertEqual(bn["stability_verdict"], "stable_bound", f"{pid} native verdict={bn}")
            self.assertLess(bn["ligand_rmsd_mean_a"], 2.0, f"{pid} native L-RMSD drifted: {bn}")

            for cls_tag, bd in (("ClassA_180", ba), ("ClassB_Shift", bb)):
                self.assertEqual(bd["stability_verdict"], "unstable_decoy", f"{pid} {cls_tag} verdict={bd}")
                self.assertLess(
                    bn["dg_bind_pred_mean_kcal"], bd["dg_bind_pred_mean_kcal"] - 3.5,
                    f"{pid} {cls_tag}: native dG ({bn['dg_bind_pred_mean_kcal']}) vs decoy ({bd['dg_bind_pred_mean_kcal']})",
                )
                self.assertLess(
                    bn["ash_binding_score_kcal"], bd["ash_binding_score_kcal"] - 15.0,
                    f"{pid} {cls_tag}: native ASH ({bn['ash_binding_score_kcal']}) vs decoy ({bd['ash_binding_score_kcal']})",
                )

        # 2. Class C: 0.00 A Crystal-Backbone Experimental Loss-of-Function Hot-Spot Alanine Mutants
        mutant_panel = [
            ("1YCR", "A", "B", [19, 23, 26]),         # p53 F19A/W23A/L26A
            ("1G3F", "A", "B", [902, 903, 904]),      # Smac V902A/P903A/I904A
            ("1BXL", "A", "B", [574, 578, 581, 585]), # Bak BH3 V574A/L578A/I581A/I585A
            ("2FLU", "X", "P", [79, 80, 82]),         # Nrf2 E79A/T80A/E82A
            ("1JYR", "A", "L", [1003, 1005]),         # Grb2 SH2 PTR1003A/N1005A
            ("1QJA", "B", "Q", [5, 7]),               # 14-3-3 Y5A/SEP7A
            ("1N7F", "A", "C", [6, 8]),               # GRIP1 PDZ6 liprin Y6A/C8A
            ("1BE9", "A", "B", [7, 9]),               # PSD-95 PDZ3 CRIPT T7A/V9A
        ]
        cfg_mut = SimulationConfig(
            ensemble="NVT", temperature_k=PHYSIOLOGICAL_TEMP_K,
            dt_ps=0.002, mts_ratio=3, minimize_steps=110, n_outer_steps=80,
            report_every_outer=20, n_replicas=2, exchange_every_outer=20,
            num_threads=4, seed=42,
        )
        for pid, rc, pc, mut_res in mutant_panel:
            pdb_txt = (CALIB_DIR / f"{pid}.pdb").read_text(encoding="utf-8")
            mut_txt = make_hotspot_alanine_mutant_pdb(pdb_txt, rc, pc, mut_res)

            sys_wt = build_complex_system(
                pdb_txt, [rc], [pc], solvent_mode="active-shell",
                max_shell_waters=40, active_radius=11.0, buffer_radius=18.0, seed=42,
            )
            sys_mut = build_complex_system(
                mut_txt, [rc], [pc], solvent_mode="active-shell",
                max_shell_waters=40, active_radius=11.0, buffer_radius=18.0, seed=42,
            )
            b_wt = run_simulation(sys_wt, cfg_mut).binding_summary
            b_mut = run_simulation(sys_mut, cfg_mut).binding_summary

            ddg = b_mut["dg_bind_pred_mean_kcal"] - b_wt["dg_bind_pred_mean_kcal"]
            dscore = b_mut["ash_binding_score_kcal"] - b_wt["ash_binding_score_kcal"]
            self.assertGreater(
                ddg, 0.55,
                f"{pid} 0.00 A Ala-mutant failed physical ddG loss-of-function check: ddG={ddg:+.3f} (WT={b_wt['dg_bind_pred_mean_kcal']}, Mut={b_mut['dg_bind_pred_mean_kcal']})",
            )
            self.assertGreater(
                dscore, 5.5,
                f"{pid} 0.00 A Ala-mutant failed ASH score separation: dScore={dscore:+.3f} (WT={b_wt['ash_binding_score_kcal']})",
            )

    def test_02_ihrt_4replica_vs_single_replica_mbar_funnel_ablation(self) -> None:
        """Original Pillar 3 Ablation: 4-replica IHRT-MBAR establishes active swaps & binding funnel separation vs 1-replica."""
        pdb_txt = (CALIB_DIR / "1YCR.pdb").read_text(encoding="utf-8")
        decoy_txt = make_inpocket_decoy_pdb(pdb_txt, "A", "B", 180.0)

        sys_nat = build_complex_system(
            pdb_txt, ["A"], ["B"], solvent_mode="active-shell",
            max_shell_waters=56, active_radius=9.5, buffer_radius=15.0, seed=77,
        )
        sys_dec = build_complex_system(
            decoy_txt, ["A"], ["B"], solvent_mode="active-shell",
            max_shell_waters=56, active_radius=9.5, buffer_radius=15.0, seed=77,
        )

        res_1rep = run_simulation(
            sys_nat,
            SimulationConfig(
                ensemble="NVT", temperature_k=PHYSIOLOGICAL_TEMP_K,
                dt_ps=0.002, mts_ratio=3, minimize_steps=100, n_outer_steps=90,
                report_every_outer=15, n_replicas=1, num_threads=4, seed=77,
            ),
        )
        self.assertIsNone(res_1rep.binding_summary["mbar_funnel_dg_kcal"])
        self.assertEqual(res_1rep.mbar_summary["f_dimensionless"], [0.0])

        cfg_4rep = SimulationConfig(
            ensemble="NVT", temperature_k=PHYSIOLOGICAL_TEMP_K,
            dt_ps=0.002, mts_ratio=3, minimize_steps=100, n_outer_steps=120,
            report_every_outer=20, n_replicas=4, exchange_every_outer=15,
            num_threads=4, seed=77,
        )
        res_nat_4 = run_simulation(sys_nat, cfg_4rep)
        res_dec_4 = run_simulation(sys_dec, cfg_4rep)

        mb_nat = res_nat_4.mbar_summary
        mb_dec = res_dec_4.mbar_summary
        f_nat = mb_nat["f_dimensionless"]
        f_dec = mb_dec["f_dimensionless"]

        self.assertGreater(sum(mb_nat["swap_accepts"]), 0)
        self.assertGreater(sum(mb_dec["swap_accepts"]), 0)

        self.assertEqual(len(f_nat), 4)
        self.assertLess(f_nat[0], f_nat[1])
        self.assertLess(f_nat[1], f_nat[2])
        self.assertLess(f_nat[2], f_nat[3])
        self.assertGreater(f_nat[3], 35.0)

        self.assertLess(f_dec[3], 10.0)
        self.assertLess(
            res_nat_4.binding_summary["mbar_funnel_dg_kcal"],
            res_dec_4.binding_summary["mbar_funnel_dg_kcal"] - 80.0,
        )

    def test_03_all_14_rcsb_complexes_zero_fitted_binding_dg_benchmark(self) -> None:
        """Criterion 3: 100% (14/14) real RCSB calib_dg.json coverage, native pose stability & physical dG accuracy."""
        # 1. Explicit strict fail-closed flags still reject incomplete heavy atoms / PTM when requested
        with self.assertRaises(TopologyBuildError):
            build_complex_system(
                (CALIB_DIR / "2FLU.pdb").read_text(encoding="utf-8"),
                ["X"], ["P"], solvent_mode="active-shell", reconstruct_missing_heavy=False,
            )
        with self.assertRaises(TopologyBuildError):
            build_complex_system(
                (CALIB_DIR / "1JYR.pdb").read_text(encoding="utf-8"),
                ["A"], ["L"], solvent_mode="active-shell", allow_ptm_and_caps=False, allow_strip_hetatm=False,
            )

        # 2. Full 14/14 real RCSB benchmark (2.5 nM to 20 uM, including missing sidechain atoms, SEP/PTR PTMs, CYM-Zn2+, EF-hand Ca2+)
        calib_entries = json.loads((CALIB_DIR / "calib_dg.json").read_text(encoding="utf-8"))["systems"]
        self.assertEqual(len(calib_entries), 14)

        exp_all: list[float] = []
        pred_all: list[float] = []
        exp_meas: list[float] = []
        pred_meas: list[float] = []
        pred_by_id: dict[str, float] = {}

        for entry in calib_entries:
            pid = entry["pdb"]
            dg_exp = float(entry["dg"])
            qual = entry["quality"]
            rc, pc = CALIB_CHAIN_MAP[pid]
            txt = (CALIB_DIR / f"{pid}.pdb").read_text(encoding="utf-8")

            sys_c = build_complex_system(
                txt, [rc], [pc],
                solvent_mode="active-shell", metal_handling="retain",
                max_shell_waters=48, active_radius=12.5, buffer_radius=20.0, seed=42,
            )
            res = run_simulation(
                sys_c,
                SimulationConfig(
                    ensemble="NVT", temperature_k=PHYSIOLOGICAL_TEMP_K,
                    dt_ps=0.002, mts_ratio=3, minimize_steps=150, minimize_fmax=4.0,
                    n_outer_steps=80, report_every_outer=20,
                    n_replicas=4, exchange_every_outer=20, num_threads=4, seed=42,
                ),
            )
            bs = res.binding_summary
            self.assertLess(bs["ligand_rmsd_mean_a"], 2.0, f"{pid} crystal pose drifted: {bs}")
            self.assertEqual(bs["stability_verdict"], "stable_bound", f"{pid} expected stable_bound: {bs}")

            dg_pred = float(bs["dg_bind_pred_mean_kcal"])
            exp_all.append(dg_exp)
            pred_all.append(dg_pred)
            pred_by_id[pid] = dg_pred
            if qual == "measured":
                exp_meas.append(dg_exp)
                pred_meas.append(dg_pred)

        arr_exp = np.asarray(exp_all, dtype=np.float64)
        arr_pred = np.asarray(pred_all, dtype=np.float64)
        rmse_all = float(np.sqrt(np.mean((arr_pred - arr_exp) ** 2)))
        mae_all = float(np.mean(np.abs(arr_pred - arr_exp)))
        r_all = float(np.corrcoef(arr_exp, arr_pred)[0, 1])
        rho_all = spearman_exact_tie_aware(exp_all, pred_all)

        arr_exp_m = np.asarray(exp_meas, dtype=np.float64)
        arr_pred_m = np.asarray(pred_meas, dtype=np.float64)
        rmse_meas = float(np.sqrt(np.mean((arr_pred_m - arr_exp_m) ** 2)))
        r_meas = float(np.corrcoef(arr_exp_m, arr_pred_m)[0, 1])

        self.assertLessEqual(rmse_all, 2.80, f"Expected ALL 14/14 RMSE <= 2.80 kcal/mol, got {rmse_all:.3f} ({pred_by_id})")
        self.assertLessEqual(mae_all, 2.20, f"Expected ALL 14/14 MAE <= 2.20 kcal/mol, got {mae_all:.3f} ({pred_by_id})")
        self.assertGreaterEqual(r_all, 0.63, f"Expected ALL 14/14 Pearson r >= 0.63, got {r_all:.4f} ({pred_by_id})")
        self.assertGreaterEqual(rho_all, 0.60, f"Expected ALL 14/14 Spearman rho >= 0.60, got {rho_all:.4f} ({pred_by_id})")
        self.assertGreaterEqual(r_meas, 0.72, f"Expected MEASURED 7/7 Pearson r >= 0.72, got {r_meas:.4f} ({pred_by_id})")
        self.assertLessEqual(rmse_meas, 2.50, f"Expected MEASURED 7/7 RMSE <= 2.50 kcal/mol, got {rmse_meas:.3f} ({pred_by_id})")


if __name__ == "__main__":
    unittest.main()
