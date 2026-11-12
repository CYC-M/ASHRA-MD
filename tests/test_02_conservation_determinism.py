"""Test Suite 02: NVE Conservation, FC-BAOAB Resonance Suppression, 310.15K NVT & Bit-Exact Determinism."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ashmd.bridge import AshmdCoreLibrary
from ashmd.simulator import PHYSIOLOGICAL_TEMP_K, SimulationConfig, run_simulation
from ashmd.topology_builder import ATOM_TYPE_PROPERTIES, build_complex_system, build_protein_system
from ashmd.ff_amber99sb_ildn import lookup_bond_param


def _load_crambin_fragment(n_res: int = 12) -> str:
    crn_path = ROOT / "tests" / "fixtures" / "1crn-chainA.pdb"
    lines = [
        line for line in crn_path.read_text().splitlines()
        if line.startswith("ATOM") and int(line[22:26].strip()) <= n_res
    ]
    return "\n".join(lines + ["END"]) + "\n"


class TestConservationAndDeterminism(unittest.TestCase):
    def test_01_nve_symplectic_energy_conservation(self) -> None:
        """Verify symplectic NVE dynamics conserves total Hamiltonian without secular drift."""
        pdb = _load_crambin_fragment(10)
        sys_obj = build_protein_system(
            pdb,
            chains=["A"],
            solvent_mode="active-shell",
            shell_waters_cutoff=4.2,
            max_shell_waters=12,
            seed=42,
        )
        cfg = SimulationConfig(
            ensemble="NVE",
            temperature_k=PHYSIOLOGICAL_TEMP_K,
            dt_ps=0.001,          # 1.0 fs symplectic step
            mts_ratio=1,
            gamma_ps_inv=0.0,
            filter_alpha=0.35,
            minimize_steps=150,
            minimize_fmax=5.0,
            n_outer_steps=400,    # 400 steps NVE
            report_every_outer=20,
            n_replicas=1,
            num_threads=2,
            seed=42,
        )
        res = run_simulation(sys_obj, cfg)
        totals = np.array([e["total"] for e in res.energies[1:]], dtype=np.float64)
        kinetics = np.array([e["kinetic"] for e in res.energies[1:]], dtype=np.float64)
        mean_ek = float(np.mean(kinetics))
        std_etot = float(np.std(totals))
        rel_fluct = std_etot / max(mean_ek, 1.0)
        self.assertLess(
            rel_fluct, 0.025,
            f"NVE total energy fluctuation too large: std(E_tot)={std_etot:.4f} kcal/mol, <Ek>={mean_ek:.2f}, ratio={rel_fluct:.4f}"
        )

    def test_02_fc_baoab_resonance_suppression_ablation(self) -> None:
        """Ablation of Pillar 1: Covalent-Centroid Filter suppresses outer-step MTS resonance on flexible X-H bonds."""
        pdb = _load_crambin_fragment(6)
        sys_unconstrained = build_protein_system(
            pdb, chains=["A"], solvent_mode="active-shell", max_shell_waters=0, seed=101
        )
        # Convert RATTLE X-H constraints into stiff harmonic AMBER X-H bonds (period ~9.8 fs -> resonance at dt_outer ~ 5 fs!)
        add_bi = list(sys_unconstrained.bond_i)
        add_bj = list(sys_unconstrained.bond_j)
        add_kr = list(sys_unconstrained.bond_kr)
        add_req = list(sys_unconstrained.bond_req)
        for ci, cj, cd2 in zip(sys_unconstrained.con_i, sys_unconstrained.con_j, sys_unconstrained.con_d2):
            t1 = sys_unconstrained.atoms[int(ci)].atype
            t2 = sys_unconstrained.atoms[int(cj)].atype
            kr, req = lookup_bond_param(t1, t2, float(np.sqrt(cd2)))
            add_bi.append(int(ci))
            add_bj.append(int(cj))
            add_kr.append(kr)
            add_req.append(req)
        sys_unconstrained.bond_i = np.array(add_bi, dtype=np.int32)
        sys_unconstrained.bond_j = np.array(add_bj, dtype=np.int32)
        sys_unconstrained.bond_kr = np.array(add_kr, dtype=np.float64)
        sys_unconstrained.bond_req = np.array(add_req, dtype=np.float64)
        sys_unconstrained.con_i = np.zeros(0, dtype=np.int32)
        sys_unconstrained.con_j = np.zeros(0, dtype=np.int32)
        sys_unconstrained.con_d2 = np.zeros(0, dtype=np.float64)

        # Probe NVE at the classic half-period parametric resonance dt_outer = 10 * 0.48 fs = 4.8 fs!
        energies_std = []
        energies_filtered = []
        for alpha_val, bucket in [(0.0, energies_std), (0.50, energies_filtered)]:
            cfg = SimulationConfig(
                ensemble="NVE",
                temperature_k=PHYSIOLOGICAL_TEMP_K,
                dt_ps=0.00048,
                mts_ratio=10,     # dt_outer = 4.80 fs (right on X-H half-period parametric resonance!)
                gamma_ps_inv=0.0,
                filter_alpha=alpha_val,
                minimize_steps=120,
                minimize_fmax=5.0,
                n_outer_steps=250,
                report_every_outer=10,
                num_threads=1,
                seed=101,
            )
            res = run_simulation(sys_unconstrained, cfg)
            etot = np.array([e["total"] for e in res.energies[1:]], dtype=np.float64)
            bucket.append(float(np.max(np.abs(etot - etot[0]))))

        drift_unfiltered = energies_std[0]
        drift_filtered = energies_filtered[0]
        # Filtered-coordinate MTS must exhibit strictly smaller energy excursion / better stability!
        self.assertLess(
            drift_filtered, drift_unfiltered,
            f"Expected FC-BAOAB filter to reduce resonant NVE excursion: filtered={drift_filtered:.4f} vs unfiltered={drift_unfiltered:.4f}"
        )

    def test_03_nvt_physiological_310_15k_canonical_temperature(self) -> None:
        """Verify geodesic BAOAB Langevin thermostat maintains physiological 310.15 K (37.00 °C)."""
        pdb = _load_crambin_fragment(10)
        sys_obj = build_protein_system(
            pdb, chains=["A"], solvent_mode="active-shell", max_shell_waters=10, seed=7
        )
        cfg = SimulationConfig(
            ensemble="NVT",
            temperature_k=PHYSIOLOGICAL_TEMP_K,
            dt_ps=0.002,
            mts_ratio=3,
            gamma_ps_inv=2.0,
            minimize_steps=100,
            n_outer_steps=350,
            report_every_outer=5,
            num_threads=2,
            seed=7,
        )
        res = run_simulation(sys_obj, cfg)
        temps = np.array([e["temperature"] for e in res.energies[15:]], dtype=np.float64)
        mean_t = float(np.mean(temps))
        self.assertAlmostEqual(
            mean_t, PHYSIOLOGICAL_TEMP_K, delta=15.0,
            msg=f"Mean NVT temperature {mean_t:.2f} K deviated excessively from 310.15 K"
        )
        self.assertEqual(res.temperature_k, 310.15)
        self.assertEqual(res.temperature_c, 37.0)

    def test_04_cross_thread_bit_exact_determinism(self) -> None:
        """Verify 1 thread vs 4 threads produce IEEE-754 BIT-IDENTICAL coordinates & velocities."""
        pdb = _load_crambin_fragment(12)
        sys_obj = build_protein_system(
            pdb, chains=["A"], solvent_mode="active-shell", max_shell_waters=16, seed=99
        )
        results = []
        for nt in (1, 4):
            cfg = SimulationConfig(
                ensemble="NVT",
                temperature_k=PHYSIOLOGICAL_TEMP_K,
                dt_ps=0.002,
                mts_ratio=3,
                gamma_ps_inv=1.0,
                minimize_steps=60,
                n_outer_steps=120,
                report_every_outer=20,
                n_replicas=1,
                num_threads=nt,
                seed=20261107,
            )
            results.append(run_simulation(sys_obj, cfg))

        max_dx = float(np.max(np.abs(results[0].final_coords - results[1].final_coords)))
        max_dv = float(np.max(np.abs(results[0].final_velocities - results[1].final_velocities)))
        self.assertEqual(max_dx, 0.0, f"Cross-thread coordinate divergence: max|dx| = {max_dx:.3e} != 0.0")
        self.assertEqual(max_dv, 0.0, f"Cross-thread velocity divergence: max|dv| = {max_dv:.3e} != 0.0")

    def test_05_checkpoint_epoch_continuation_bit_exactness(self) -> None:
        """Verify 4-replica IHRT-MBAR split across a checkpoint epoch matches uninterrupted execution BIT-FOR-BIT."""
        crn = pathlib.Path(__file__).parent / "fixtures" / "1crn-chainA.pdb".read_text()
        lines_a = [l for l in crn.splitlines() if l.startswith("ATOM") and 1 <= int(l[22:26]) <= 8]
        lines_b = [l[:21] + "B" + l[22:] for l in crn.splitlines() if l.startswith("ATOM") and 12 <= int(l[22:26]) <= 16]
        pdb = "\n".join(lines_a + ["TER"] + lines_b + ["END"]) + "\n"

        sys_obj = build_complex_system(
            pdb, receptor_chains=["A"], peptide_chains=["B"],
            solvent_mode="active-shell", max_shell_waters=10, seed=55
        )

        # Reference uninterrupted run (80 outer steps, 4 IHRT replicas, swaps every 10 outer steps)
        cfg_full = SimulationConfig(
            ensemble="NVT",
            temperature_k=PHYSIOLOGICAL_TEMP_K,
            dt_ps=0.002,
            mts_ratio=2,
            minimize_steps=40,
            n_outer_steps=80,
            report_every_outer=10,
            n_replicas=4,
            exchange_every_outer=10,
            num_threads=4,
            seed=8888,
        )
        res_full = run_simulation(sys_obj, cfg_full)

        # Split two-epoch run (Epoch 1: 40 steps -> save .chk -> Epoch 2: load .chk -> finish to 80 steps!)
        with tempfile.TemporaryDirectory() as tmpdir:
            chk = pathlib.Path(tmpdir) / "epoch.chk"
            cfg_ep1 = SimulationConfig(
                ensemble="NVT",
                temperature_k=PHYSIOLOGICAL_TEMP_K,
                dt_ps=0.002,
                mts_ratio=2,
                minimize_steps=40,
                n_outer_steps=40,
                report_every_outer=10,
                n_replicas=4,
                exchange_every_outer=10,
                num_threads=4,
                seed=8888,
                checkpoint_path=chk,
                checkpoint_every_frames=2,
            )
            run_simulation(sys_obj, cfg_ep1)

            cfg_ep2 = SimulationConfig(
                ensemble="NVT",
                temperature_k=PHYSIOLOGICAL_TEMP_K,
                dt_ps=0.002,
                mts_ratio=2,
                minimize_steps=40,
                n_outer_steps=80,
                report_every_outer=10,
                n_replicas=4,
                exchange_every_outer=10,
                num_threads=4,
                seed=8888,
                checkpoint_path=chk,
                resume_from_checkpoint=True,
            )
            res_resumed = run_simulation(sys_obj, cfg_ep2)

        max_dx = float(np.max(np.abs(res_full.final_coords - res_resumed.final_coords)))
        self.assertEqual(max_dx, 0.0, f"Epoch checkpoint resume drifted: max|dx| = {max_dx:.3e}")
        self.assertEqual(res_full.mbar_summary["f_dimensionless"], res_resumed.mbar_summary["f_dimensionless"])
        self.assertEqual(res_full.mbar_summary["swap_accepts"], res_resumed.mbar_summary["swap_accepts"])


if __name__ == "__main__":
    unittest.main()
