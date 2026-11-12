"""Test Suite 01: Analytical Force Gradients, SPME Exactness & Fail-Closed Guards (unittest style)."""

from __future__ import annotations

import math
import pathlib
import sys
import tempfile
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ashmd.bridge import AshmdCoreLibrary
from ashmd.topology_builder import (
    TopologyBuildError,
    build_complex_system,
    build_protein_system,
)


def _make_mini_peptide_pdb() -> str:
    """Construct a clean, clash-free 2-chain mini complex (3-aa receptor tripeptide A + 2-aa ligand B) for exact all-atom FD checks."""
    # Standard alpha/extended ideal coordinates from real PDB geometry
    crn_path = ROOT / "tests" / "fixtures" / "1crn-chainA.pdb"
    lines_a = []
    lines_b = []
    for line in crn_path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        rseq = int(line[22:26].strip())
        if 1 <= rseq <= 4:
            lines_a.append(line)
        elif 10 <= rseq <= 12:
            # Re-label chain B as interacting peptide segment
            lines_b.append(line[:21] + "B" + line[22:])
    return "\n".join(lines_a + ["TER"] + lines_b + ["END"]) + "\n"


class TestGradientsAndGuards(unittest.TestCase):
    def test_01_active_shell_all_terms_analytical_gradient(self) -> None:
        """Verify analytical forces match central finite differences across ALL Active-Shell + OBC2 + Cap + Filter + Softcore terms."""
        pdb = _make_mini_peptide_pdb()
        sys_obj = build_complex_system(
            pdb,
            receptor_chains=["A"],
            peptide_chains=["B"],
            solvent_mode="active-shell",
            shell_waters_cutoff=4.5,
            max_shell_waters=8,
            active_radius=6.0,
            buffer_radius=14.0,
            seed=123,
        )
        # Ensure cap is actively engaged on at least one water by tightening cap_radius slightly
        sys_obj.cap_radius = 2.5
        sys_obj.cap_k = 5.0
        sys_obj.filter_alpha = 0.35

        with AshmdCoreLibrary(num_threads=2) as core:
            core.upload_system(sys_obj, filter_alpha=0.35)
            # Test with tempered soft-core replica parameters active simultaneously!
            core.init_replicas(
                sys_obj.coords,
                n_replicas=1,
                lambda_inter=[0.75],
                mu_torsion=[0.85],
                softcore_alpha=[0.08],
                seed=123,
                init_temp_k=0.0,
            )
            # Relax extreme H clashes briefly so energies are smooth O(10^2)
            core.minimize(max_steps=30, fmax_tol=20.0, init_temp_after_k=0.0)
            base_xyz, _, base_e = core.get_state(0)
            f_ana, _ = core.eval_single_point(0, target_temp_k=310.15)

            # Verify softmin cap and GB polarization are genuinely non-zero!
            self.assertGreater(abs(base_e["gb_pol"]), 1.0)
            self.assertGreater(base_e["np_sasa"], 0.01)
            self.assertGreater(base_e["cap_shell"], 0.001)

            eps_fd = 1.0e-5
            max_abs_err = 0.0
            max_rel_err = 0.0

            # Check a thorough sample covering receptor, peptide, hydrogens, and shell waters
            test_atoms = list(range(0, sys_obj.n_atoms, max(1, sys_obj.n_atoms // 35)))
            for ai in test_atoms:
                if sys_obj.zone[ai] == 3:  # Frozen scaffold has zero force by definition
                    continue
                for dim in range(3):
                    pos_p = base_xyz.copy()
                    pos_p[ai, dim] += eps_fd
                    core.set_state(0, coords=pos_p)
                    _, ep = core.eval_single_point(0, target_temp_k=310.15)

                    pos_m = base_xyz.copy()
                    pos_m[ai, dim] -= eps_fd
                    core.set_state(0, coords=pos_m)
                    _, em = core.eval_single_point(0, target_temp_k=310.15)

                    f_num = -(ep["potential"] - em["potential"]) / (2.0 * eps_fd)
                    err = abs(float(f_ana[ai, dim]) - f_num)
                    denom = max(1.0, abs(f_num), abs(float(f_ana[ai, dim])))
                    max_abs_err = max(max_abs_err, err)
                    max_rel_err = max(max_rel_err, err / denom)

            self.assertLess(
                max_rel_err, 5.0e-5,
                f"ASH+OBC2+Cap+Filter gradient mismatch: max_rel_err={max_rel_err:.3e}, max_abs={max_abs_err:.3e}"
            )

    def test_02_explicit_pme_analytical_gradient_and_ewald_reference(self) -> None:
        """Verify 4th-order B-Spline SPME + Direct Ewald + Excluded erf + Filter forces match central FD AND exact Ewald lattice sum."""
        pdb = _make_mini_peptide_pdb()
        sys_obj = build_complex_system(
            pdb,
            receptor_chains=["A"],
            peptide_chains=["B"],
            solvent_mode="explicit-pme",
            box_padding=5.0,
            ionic_strength=0.15,
            seed=77,
        )
        # Use dense PME grid for high-accuracy Ewald comparison
        sys_obj.pme_grid = (32, 32, 32)
        sys_obj.cutoff = 8.5
        sys_obj.switch_on = 7.0
        sys_obj.ewald_alpha = 0.38
        sys_obj.filter_alpha = 0.35

        with AshmdCoreLibrary(num_threads=2) as core:
            core.upload_system(sys_obj, filter_alpha=0.35)
            core.init_replicas(sys_obj.coords, n_replicas=1, seed=77, init_temp_k=0.0)
            core.minimize(max_steps=20, fmax_tol=25.0, init_temp_after_k=0.0)
            base_xyz, _, base_e = core.get_state(0)
            f_ana, _ = core.eval_single_point(0, target_temp_k=310.15)

            self.assertNotEqual(base_e["elec_recip"], 0.0)
            self.assertNotEqual(base_e["elec_self"], 0.0)

            eps_fd = 1.0e-5
            max_rel_err = 0.0
            # Test across solute, ions, and explicit water atoms
            indices = list(range(0, min(60, sys_obj.n_atoms), 5)) + list(range(sys_obj.n_atoms - 15, sys_obj.n_atoms, 3))
            for ai in sorted(set(indices)):
                for dim in range(3):
                    pos_p = base_xyz.copy()
                    pos_p[ai, dim] += eps_fd
                    core.set_state(0, coords=pos_p)
                    _, ep = core.eval_single_point(0, target_temp_k=310.15)

                    pos_m = base_xyz.copy()
                    pos_m[ai, dim] -= eps_fd
                    core.set_state(0, coords=pos_m)
                    _, em = core.eval_single_point(0, target_temp_k=310.15)

                    f_num = -(ep["potential"] - em["potential"]) / (2.0 * eps_fd)
                    err = abs(float(f_ana[ai, dim]) - f_num)
                    denom = max(1.0, abs(f_num), abs(float(f_ana[ai, dim])))
                    max_rel_err = max(max_rel_err, err / denom)

            self.assertLess(max_rel_err, 5.0e-5, f"SPME gradient mismatch: max_rel_err={max_rel_err:.3e}")

        # Cross-verify SPME reciprocal energy on an exact neutral 4-charge cubic crystal vs analytical Fourier Ewald sum!
        sys_chk = build_protein_system(
            "\n".join(l for l in _make_mini_peptide_pdb().splitlines() if l.startswith("ATOM") and l[21] == "B") + "\nEND\n",
            solvent_mode="explicit-pme",
            box_padding=6.0,
            ionic_strength=0.0,
            seed=1,
        )
        # Keep only first 4 atoms with exact neutral charges [+1, -1, +0.5, -0.5], zero exclusions, filter_alpha=0
        sys_chk.n_atoms = 4
        sys_chk.coords = np.array([
            [3.1, 4.2, 5.3],
            [7.4, 2.8, 6.1],
            [5.0, 8.1, 3.7],
            [9.2, 6.5, 8.9],
        ], dtype=np.float64)
        sys_chk.mass = np.array([12.0, 12.0, 12.0, 12.0], dtype=np.float64)
        sys_chk.charge = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float64)
        sys_chk.rmin_half = np.ones(4, dtype=np.float64)
        sys_chk.epsilon = np.zeros(4, dtype=np.float64)
        sys_chk.gb_radius = np.ones(4, dtype=np.float64) * 1.5
        sys_chk.gb_screen = np.ones(4, dtype=np.float64) * 0.8
        sys_chk.zone = np.zeros(4, dtype=np.int32)
        sys_chk.role = np.zeros(4, dtype=np.int32)
        sys_chk.tether_k = np.zeros(4, dtype=np.float64)
        for attr in ("bond_i", "bond_j", "ang_i", "ang_j", "ang_k", "dih_i", "dih_j", "dih_k", "dih_l",
                     "dih_is_pep", "p14_i", "p14_j", "excl_targets", "con_i", "con_j",
                     "wat_o", "wat_h1", "wat_h2", "fg_heavy", "fg_nh", "fg_h_flat"):
            setattr(sys_chk, attr, np.zeros(0, dtype=np.int32))
        for attr in ("bond_kr", "bond_req", "ang_kt", "ang_eq", "dih_pk", "dih_n", "dih_phase",
                     "p14_qprod", "p14_rmin", "p14_eps", "con_d2"):
            setattr(sys_chk, attr, np.zeros(0, dtype=np.float64))
        sys_chk.excl_offsets = np.zeros(5, dtype=np.int32)
        sys_chk.box = (16.0, 16.0, 16.0)
        sys_chk.cutoff = 7.5
        sys_chk.switch_on = 7.0
        sys_chk.ewald_alpha = 0.42
        sys_chk.pme_grid = (64, 64, 64)
        sys_chk.filter_alpha = 0.0

        with AshmdCoreLibrary(num_threads=1) as core:
            core.upload_system(sys_chk, filter_alpha=0.0)
            core.init_replicas(sys_chk.coords, n_replicas=1, init_temp_k=0.0)
            _, e_spme = core.eval_single_point(0)
            u_spme_total = e_spme["elec_dir"] + e_spme["elec_recip"] + e_spme["elec_self"]

        # Exact analytical Ewald sum (real-space n in [-2, 2]^3 + exact Fourier k-space m in [-8, 8]^3)
        L = 16.0
        V = L ** 3
        alpha = 0.42
        C = 332.05221729
        q = sys_chk.charge
        r = sys_chk.coords
        u_real_exact = 0.0
        for i in range(4):
            for j in range(4):
                for nx in range(-2, 3):
                    for ny in range(-2, 3):
                        for nz in range(-2, 3):
                            if i == j and nx == 0 and ny == 0 and nz == 0:
                                continue
                            rij = np.linalg.norm(r[i] - r[j] + L * np.array([nx, ny, nz]))
                            u_real_exact += 0.5 * C * q[i] * q[j] * math.erfc(alpha * rij) / rij
        u_self_exact = -(alpha / math.sqrt(math.pi)) * C * float(np.sum(q * q))
        u_rec_exact = 0.0
        for mx in range(-8, 9):
            for my in range(-8, 9):
                for mz in range(-8, 9):
                    if mx == 0 and my == 0 and mz == 0:
                        continue
                    mvec = np.array([mx / L, my / L, mz / L])
                    m2 = float(np.dot(mvec, mvec))
                    s_re = float(np.sum(q * np.cos(2.0 * math.pi * (r @ mvec))))
                    s_im = float(np.sum(q * np.sin(2.0 * math.pi * (r @ mvec))))
                    u_rec_exact += 0.5 * (C / (math.pi * V)) * (math.exp(-(math.pi**2) * m2 / (alpha**2)) / m2) * (s_re**2 + s_im**2)
        u_ewald_exact = u_real_exact + u_self_exact + u_rec_exact
        self.assertAlmostEqual(u_spme_total, u_ewald_exact, delta=0.015)

    def test_03_fail_closed_guards_must_ring(self) -> None:
        """Verify every fail-closed safety & integrity guard actively triggers when violated."""
        pdb = _make_mini_peptide_pdb()

        # Guard A: Missing sidechain heavy atom -> MUST raise TopologyBuildError
        broken_missing = "\n".join(l for l in pdb.splitlines() if not (l.startswith("ATOM") and "CB  THR A   1" in l))
        with self.assertRaises(TopologyBuildError):
            build_protein_system(broken_missing, chains=["A"])

        # Guard B: Unapproved HETATM ligand -> MUST raise TopologyBuildError when allow_strip_hetatm=False
        het_pdb = pdb.replace("END", "HETATM 9999  C1  LIG A 999      10.000  10.000  10.000  1.00  0.00           C\nEND")
        with self.assertRaises(TopologyBuildError):
            build_protein_system(het_pdb, chains=["A"], allow_strip_hetatm=False)

        # Guard C: Chain break (C-N > 2.3 A) -> MUST raise TopologyBuildError when allow_chain_breaks=False
        gapped_lines = [l for l in pdb.splitlines() if not (l.startswith("ATOM") and int(l[22:26]) == 2)]
        with self.assertRaises(TopologyBuildError):
            build_protein_system("\n".join(gapped_lines), chains=["A"], allow_chain_breaks=False)

        # Guard D: Severe heavy-atom overlap (< 0.8 A) -> MUST raise TopologyBuildError
        lines = pdb.splitlines()
        first_atom_xyz = lines[0][30:54]
        clashed_line = lines[15][:30] + first_atom_xyz + lines[15][54:]
        lines[15] = clashed_line
        with self.assertRaises(TopologyBuildError):
            build_protein_system("\n".join(lines), chains=["A"])

        # Guard E: 1-bit SHA-256 corruption in .chk binary capsule -> MUST fail-closed on load!
        sys_ok = build_protein_system(pdb, chains=["A"], solvent_mode="active-shell", max_shell_waters=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            chk_file = pathlib.Path(tmpdir) / "state.chk"
            with AshmdCoreLibrary(num_threads=1) as core:
                core.upload_system(sys_ok)
                core.init_replicas(sys_ok.coords, n_replicas=1, init_temp_k=310.15)
                core.run_steps(5, dt_ps=0.002, mts_ratio=2, gamma_ps_inv=1.0, target_temp_k=310.15)
                core.save_checkpoint(chk_file)

                # Flip 1 bit in middle of payload
                raw = bytearray(chk_file.read_bytes())
                raw[len(raw) // 2] ^= 0x01
                chk_file.write_bytes(bytes(raw))

                with self.assertRaises(RuntimeError) as ctx:
                    core.load_checkpoint(chk_file)
                self.assertIn("SHA-256", str(ctx.exception))

    def test_04_fast_cross_potentials_exactness_and_ptm_cym_build(self) -> None:
        """Verify O(N_inter) IHRT cross-potential evaluator matches 16x full evals (<1e-8) and PTM/CYM/Kabsch builds."""
        pdb = _make_mini_peptide_pdb()
        lam_sched = [1.00, 0.92, 0.84, 0.76]
        mu_sched = [math.sqrt(x) for x in lam_sched]
        sc_sched = [round(0.06 * (1.0 - x), 6) for x in lam_sched]
        beta0 = 1.0 / (0.00198720425864083 * 310.15)

        for mode in ("active-shell", "explicit-pme"):
            if mode == "active-shell":
                sys_m = build_complex_system(
                    pdb, ["A"], ["B"], solvent_mode=mode, max_shell_waters=8,
                    active_radius=6.0, buffer_radius=14.0, seed=99,
                )
            else:
                sys_m = build_complex_system(
                    pdb, ["A"], ["B"], solvent_mode=mode, box_padding=5.0,
                    ionic_strength=0.15, seed=99,
                )

            with AshmdCoreLibrary(num_threads=2) as core:
                core.upload_system(sys_m, filter_alpha=0.35)
                core.init_replicas(
                    sys_m.coords, n_replicas=4,
                    lambda_inter=lam_sched, mu_torsion=mu_sched, softcore_alpha=sc_sched,
                    seed=99, init_temp_k=310.15,
                )
                core.minimize(max_steps=20, fmax_tol=20.0, init_temp_after_k=310.15)
                core.run_steps(4, dt_ps=0.002, mts_ratio=2, gamma_ps_inv=1.0, target_temp_k=310.15, exchange_every_outer=0)

                u_fast = core.eval_cross_potentials(310.15)
                snap_coords = [core.get_state(r)[0] for r in range(4)]

                u_ref = np.zeros((4, 4), dtype=np.float64)
                for r_conf in range(4):
                    for k_ham in range(4):
                        core.set_replica_tempering(
                            [lam_sched[k_ham]] * 4, [mu_sched[k_ham]] * 4, [sc_sched[k_ham]] * 4, 310.15
                        )
                        core.set_state(0, coords=snap_coords[r_conf])
                        _, ek = core.eval_single_point(0, target_temp_k=310.15)
                        u_ref[r_conf, k_ham] = beta0 * ek["potential"]

                max_diff = float(np.max(np.abs(u_fast - u_ref)))
                self.assertLess(max_diff, 1.0e-8, f"Fast cross-potential mismatch in {mode}: {max_diff:.3e}")

        # Verify real RCSB PTM (SEP in 1QJA, PTR in 1JYR), Zn-thiolate CYM (1G3F), and missing-atom reconstruction (2FLU)
        calib_dir = ROOT / "tests" / "fixtures" / "calib"
        s_1qja = build_complex_system((calib_dir / "1QJA.pdb").read_text(), ["B"], ["Q"], max_shell_waters=8)
        self.assertTrue(any(a.res_name == "SEP" and a.atype == "OP" for a in s_1qja.atoms))

        s_1jyr = build_complex_system((calib_dir / "1JYR.pdb").read_text(), ["A"], ["L"], max_shell_waters=8)
        self.assertTrue(any(a.res_name == "PTR" and a.atype == "OP" for a in s_1jyr.atoms))

        s_1g3f = build_complex_system((calib_dir / "1G3F.pdb").read_text(), ["A"], ["B"], max_shell_waters=8)
        cym_res = {a.res_seq for a in s_1g3f.atoms if a.res_name == "CYM"}
        self.assertEqual(len(cym_res), 3, f"Expected Cys3His1 Zn-finger CYM thiolates in 1G3F, got {cym_res}")

        s_2flu = build_complex_system((calib_dir / "2FLU.pdb").read_text(), ["X"], ["P"], max_shell_waters=8)
        self.assertTrue(any("Reconstructed" in w for w in s_2flu.warnings))

        # Explicit strict fail-closed switches still abort when requested
        with self.assertRaises(TopologyBuildError):
            build_complex_system(
                (calib_dir / "2FLU.pdb").read_text(), ["X"], ["P"], reconstruct_missing_heavy=False
            )
        with self.assertRaises(TopologyBuildError):
            build_complex_system(
                (calib_dir / "1JYR.pdb").read_text(), ["A"], ["L"], allow_ptm_and_caps=False, allow_strip_hetatm=False
            )


if __name__ == "__main__":
    unittest.main()
