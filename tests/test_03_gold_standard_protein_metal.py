"""Test Suite 03: Gold-Standard Protein Fold Stability, Explicit-PME vs Active-Shell Cross-Validation,
and Metalloenzyme Zn2+ Coordination Retention (Criterion 2).
"""

from __future__ import annotations

import gzip
import json
import pathlib
import sys
import tempfile
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ashmd.__main__ import main as cli_main
from ashmd.pepfind_adapter import run_pepfind_ashmd
from ashmd.simulator import PHYSIOLOGICAL_TEMP_K, SimulationConfig, run_simulation
from ashmd.topology_builder import ROLE_SOLVENT, build_protein_system

FIXTURES = ROOT / "tests" / "fixtures"
CRAMBIN_PDB = FIXTURES / "1crn-chainA.pdb"
CA2_6HD2_GZ = FIXTURES / "6HD2.pdb.gz"
CPA1_1M4L_GZ = FIXTURES / "1M4L.pdb.gz"


class TestGoldStandardProteinAndMetalloenzyme(unittest.TestCase):
    def test_01_crambin_explicit_pme_vs_active_shell_and_disulfides(self) -> None:
        """Cross-validate 1CRN Crambin (3 CYX disulfides) across Explicit-PME and Active-Shell at 310.15 K."""
        pdb_txt = CRAMBIN_PDB.read_text(encoding="utf-8")

        # 1. Active-Shell mode
        sys_ash = build_protein_system(
            pdb_txt,
            chains=["A"],
            solvent_mode="active-shell",
            max_shell_waters=48,
            seed=101,
        )
        cyx_residues = {am.res_seq for am in sys_ash.atoms if am.res_name == "CYX"}
        self.assertEqual(cyx_residues, {3, 4, 16, 26, 32, 40})

        res_ash = run_simulation(
            sys_ash,
            SimulationConfig(
                ensemble="NVT",
                temperature_k=PHYSIOLOGICAL_TEMP_K,
                dt_ps=0.002,
                mts_ratio=3,
                minimize_steps=120,
                n_outer_steps=160,
                report_every_outer=20,
                num_threads=4,
                seed=101,
            ),
        )

        # 2. Explicit-PME periodic TIP3P box + 4th-order B-spline SPME mode
        sys_pme = build_protein_system(
            pdb_txt,
            chains=["A"],
            solvent_mode="explicit-pme",
            box_padding=6.5,
            seed=101,
        )
        res_pme = run_simulation(
            sys_pme,
            SimulationConfig(
                ensemble="NVT",
                temperature_k=PHYSIOLOGICAL_TEMP_K,
                dt_ps=0.002,
                mts_ratio=2,
                minimize_steps=80,
                n_outer_steps=80,
                report_every_outer=20,
                num_threads=4,
                seed=101,
            ),
        )

        # Both modes preserve native Crambin fold (< 1.8 A backbone RMSD) and agree on Rg within 0.40 A
        rmsd_ash = res_ash.binding_summary["receptor_rmsd_mean_a"]
        rmsd_pme = res_pme.binding_summary["receptor_rmsd_mean_a"]
        rg_ash = float(np.mean([o["gyration_a"] for o in res_ash.observables[1:]]))
        rg_pme = float(np.mean([o["gyration_a"] for o in res_pme.observables[1:]]))

        self.assertLess(rmsd_ash, 1.80, f"Crambin Active-Shell RMSD too high: {rmsd_ash}")
        self.assertLess(rmsd_pme, 1.80, f"Crambin Explicit-PME RMSD too high: {rmsd_pme}")
        self.assertLess(abs(rg_ash - rg_pme), 0.40, f"Rg mismatch: ASH={rg_ash:.3f} vs PME={rg_pme:.3f}")

        # Verify all 3 CYX-CYX SG-SG disulfide bonds remain intact within canonical +/-3.5 sigma of req=2.038 A
        sg_by_res = {
            am.res_seq: am.index
            for am in sys_ash.atoms
            if am.res_name == "CYX" and am.atom_name == "SG"
        }
        for r1, r2 in [(3, 40), (4, 32), (16, 26)]:
            d_sg = float(np.linalg.norm(res_ash.final_coords[sg_by_res[r1]] - res_ash.final_coords[sg_by_res[r2]]))
            self.assertGreaterEqual(d_sg, 1.88)
            self.assertLessEqual(d_sg, 2.20)

    def test_02_carbonic_anhydrase_ii_zn2_tetrahedral_coordination_6hd2(self) -> None:
        """Verify His tautomer assignment (HID94/HID96/HIE119) + 4th-site catalytic water coordination in CA-II (6HD2)."""
        txt = gzip.open(CA2_6HD2_GZ, "rt", encoding="utf-8").read()
        sys_ca2 = build_protein_system(
            txt,
            chains=["A"],
            solvent_mode="active-shell",
            metal_handling="retain",
            zinc_coord_mode="harmonic_restraint",
            allow_strip_hetatm=True,
            allow_chain_breaks=True,
            max_shell_waters=56,
            seed=202,
        )
        # Verify exact biological tautomer assignment: NE2->HID (proton on ND1), ND1->HIE (proton on NE2)
        his_tautomers = {
            am.res_seq: am.res_name
            for am in sys_ca2.atoms
            if am.res_seq in (94, 96, 119) and am.atom_name == "CA"
        }
        self.assertEqual(his_tautomers, {94: "HID", 96: "HID", 119: "HIE"})

        res = run_simulation(
            sys_ca2,
            SimulationConfig(
                ensemble="NVT",
                temperature_k=PHYSIOLOGICAL_TEMP_K,
                dt_ps=0.002,
                mts_ratio=3,
                minimize_steps=140,
                n_outer_steps=180,
                report_every_outer=20,
                num_threads=4,
                seed=202,
            ),
        )
        self.assertEqual(len(res.zinc_coordination), 1)
        donors = res.zinc_coordination[0]["donors"]
        self.assertEqual(len(donors), 3)
        for d in donors:
            self.assertGreaterEqual(d["meanA"], 1.92, f"Zn-His compressed: {d}")
            self.assertLessEqual(d["meanA"], 2.28, f"Zn-His lost coordination: {d}")

        # Open 4th tetrahedral site must recruit an explicit active-shell TIP3P water oxygen (d <= 2.45 A)
        zn_idx = sys_ca2.zinc_sites[0]["zn_atom_index"]
        wat_o_indices = [am.index for am in sys_ca2.atoms if am.role == ROLE_SOLVENT and am.atom_name == "O"]
        d_zn_wat = np.linalg.norm(res.final_coords[wat_o_indices] - res.final_coords[zn_idx], axis=1)
        min_wat_dist = float(np.min(d_zn_wat))
        self.assertLessEqual(min_wat_dist, 2.45, f"Expected catalytic 4th-ligand water near Zn2+, got {min_wat_dist:.3f} A")

    def test_03_carboxypeptidase_a1_zn2_coordination_and_ablation_1m4l(self) -> None:
        """Verify CPA1 (1M4L) His69/Glu72/His196 coordination retention vs bare nonbonded ablation at 310.15 K."""
        txt = gzip.open(CPA1_1M4L_GZ, "rt", encoding="utf-8").read()

        sys_restrained = build_protein_system(
            txt,
            chains=["A"],
            solvent_mode="active-shell",
            metal_handling="retain",
            zinc_coord_mode="harmonic_restraint",
            allow_strip_hetatm=True,
            allow_chain_breaks=True,
            max_shell_waters=56,
            seed=303,
        )
        res_restrained = run_simulation(
            sys_restrained,
            SimulationConfig(
                ensemble="NVT",
                temperature_k=PHYSIOLOGICAL_TEMP_K,
                dt_ps=0.002,
                mts_ratio=3,
                minimize_steps=140,
                n_outer_steps=160,
                report_every_outer=20,
                num_threads=4,
                seed=303,
            ),
        )
        max_mean_restrained = max(d["meanA"] for d in res_restrained.zinc_coordination[0]["donors"])
        self.assertLessEqual(max_mean_restrained, 2.38)

        # Ablation: bare nonbonded fixed-charge Zn2+ allows TIP3P water ligand displacement
        sys_unrestrained = build_protein_system(
            txt,
            chains=["A"],
            solvent_mode="active-shell",
            metal_handling="retain",
            zinc_coord_mode="nonbonded_only",
            allow_strip_hetatm=True,
            allow_chain_breaks=True,
            max_shell_waters=56,
            seed=303,
        )
        res_unrestrained = run_simulation(
            sys_unrestrained,
            SimulationConfig(
                ensemble="NVT",
                temperature_k=PHYSIOLOGICAL_TEMP_K,
                dt_ps=0.002,
                mts_ratio=3,
                minimize_steps=140,
                n_outer_steps=160,
                report_every_outer=20,
                num_threads=4,
                seed=303,
            ),
        )
        max_mean_unrestrained = max(d["meanA"] for d in res_unrestrained.zinc_coordination[0]["donors"])
        self.assertGreater(max_mean_unrestrained, max_mean_restrained + 0.50)

    def test_04_pepfind_adapter_reviewer_invariants_and_cli_epoch(self) -> None:
        """Verify PepFind adapter R1-R4 invariants, non-physiological T warning, and CLI epoch resume."""
        pdb_txt = CRAMBIN_PDB.read_text(encoding="utf-8")
        out_300 = run_pepfind_ashmd(
            pdb_text=pdb_txt,
            receptor_chains=["A"],
            profile="smoke",
            temperature_k=300.0,
            n_replicas_override=1,
            num_threads=2,
        )
        self.assertFalse(out_300["temperature"]["isPhysiological"])
        self.assertEqual(out_300["temperature"]["display"], "300.00 K (26.85 °C)")
        self.assertTrue(any("310.15 K" in w for w in out_300["warnings"]))
        for c in out_300["series"]["hbondCounts"]:
            self.assertIsInstance(c, int)
            self.assertGreaterEqual(c, 0)

        # Test CLI (`python -m ashmd`) with checkpoint save + resume
        with tempfile.TemporaryDirectory() as tmp:
            chk = pathlib.Path(tmp) / "cli_epoch.chk"
            j1 = pathlib.Path(tmp) / "out1.json"
            j2 = pathlib.Path(tmp) / "out2.json"
            rc1 = cli_main([
                "--pdb", str(CRAMBIN_PDB),
                "--receptor-chains", "A",
                "--profile", "smoke",
                "--replicas", "1",
                "--outer-steps", "30",
                "--checkpoint", str(chk),
                "--out-json", str(j1),
            ])
            self.assertEqual(rc1, 0)
            rc2 = cli_main([
                "--pdb", str(CRAMBIN_PDB),
                "--receptor-chains", "A",
                "--profile", "smoke",
                "--replicas", "1",
                "--outer-steps", "60",
                "--checkpoint", str(chk),
                "--resume",
                "--out-json", str(j2),
            ])
            self.assertEqual(rc2, 0)
            d2 = json.loads(j2.read_text(encoding="utf-8"))
            self.assertTrue(d2["completed"])
            self.assertEqual(d2["performance"]["outerStepsCompleted"], 60)
            self.assertTrue(d2["temperature"]["isPhysiological"])


if __name__ == "__main__":
    unittest.main()
