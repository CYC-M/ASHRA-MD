"""Strict Fail-Closed All-Atom AMBER99SB-ILDN Topology & Solvation Builder for ASH-MD.

Implements:
  1. Fail-closed PDB validation matching PepFind truthfulness rules:
     - Rejects unknown/modified residues immediately
     - Rejects missing internal/sidechain heavy atoms immediately
     - Rejects unapproved internal backbone breaks (C(i)-N(i+1) > 2.3 A) unless allow_chain_breaks=True
     - Rejects severe heavy-atom steric overlaps (< 0.8 A)
     - Deterministic Zn2+-coordinating Histidine tautomer selection (ND1 -> HIE, NE2 -> HID)
     - Automatic CYX-CYX disulfide bond detection & parameterization (SG-SG <= 2.35 A)
  2. Deterministic analytical hydrogen & C-terminal OXT reconstruction via local Kabsch alignment
     onto verified AMBER reference templates (using 1st + 2nd shell heavy atoms for zero rotational ambiguity).
  3. Complete AMBER99SB-ILDN covalent energy graph:
     - Bonds, Angles, Proper Dihedrals (ff99SB phi/psi + ILDN chi1/chi2 + parm99), Planar Impropers,
       1-4 scaled pairs (SCEE=1.2, SCNB=2.0), CSR exclusions, RATTLE X-H constraints, SETTLE waters,
       and FC-BAOAB covalent-centroid filter groups.
  4. Dual-Mode Solvation:
     - 'explicit-pme': Periodic orthorhombic TIP3P water box + neutralizing/0.15M Na+/Cl- + power-of-2 PME grid
     - 'active-shell': Explicit discrete interfacial TIP3P hydration shell around pocket+peptide +
       Softmin Pocket-Envelope Hydration Cap + Three-Zone (Active / Buffer Rim / Scaffold) partition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from .ff_amber99sb_ildn import (
    ATOM_TYPE_PROPERTIES,
    DEG2RAD,
    FF99SB_PHI_TERMS,
    FF99SB_PSI_TERMS,
    ILDN_SIDECHAIN_OVERRIDES,
    RESIDUE_INTERIOR_DIELECTRIC,
    generic_dihedral_terms,
    load_amber_templates,
    lookup_angle_param,
    lookup_bond_param,
)

COULOMB_KCAL_A = 332.05221729

# Zone constants matching C++ AtomZone enum
ZONE_ACTIVE_SOLUTE = 0
ZONE_ACTIVE_WATER = 1
ZONE_BUFFER_RIM = 2
ZONE_SCAFFOLD = 3

# Role constants matching C++ SoluteRole enum
ROLE_RECEPTOR = 0
ROLE_PEPTIDE = 1
ROLE_ION = 2
ROLE_SOLVENT = 3

WATER_RESNAMES = {"HOH", "WAT", "SOL", "TIP", "TIP3", "DOD"}
RETAINED_IONS = {"ZN": "ZN2+", "CA": "CA2+", "MG": "MG2+", "CL": "CL-", "NA": "NA+"}
POLYMERIC_PTM_CAPS = {"SEP", "TPO", "PTR", "ACE", "NME"}

STANDARD_AA_ALIASES = {
    "HISD": "HID",
    "HISE": "HIE",
    "HISP": "HIP",
    "HSD": "HID",
    "HSE": "HIE",
    "HSP": "HIP",
    "MSE": "MET",
    "S2P": "SEP",
    "T2P": "TPO",
    "Y2P": "PTR",
}

ATOM_NAME_ALIASES = {
    "OT1": "O",
    "O1": "O",
    "OT2": "OXT",
    "O2": "OXT",
    "HN": "H",
    "SE": "SD",
    "OP1": "O1P",
    "OP2": "O2P",
    "OP3": "O3P",
}


class TopologyBuildError(ValueError):
    """Fail-closed error raised on invalid, incomplete, or clashed input structures."""


@dataclass
class AtomMeta:
    index: int
    chain_id: str
    res_seq: int
    res_name: str
    atom_name: str
    atype: str
    is_heavy: bool
    role: int


@dataclass
class BuiltSystem:
    n_atoms: int
    solvent_mode: str  # 'explicit-pme' or 'active-shell'
    coords: np.ndarray  # (N, 3) float64 [A]
    atoms: list[AtomMeta]

    mass: np.ndarray
    charge: np.ndarray
    rmin_half: np.ndarray
    epsilon: np.ndarray
    gb_radius: np.ndarray
    gb_screen: np.ndarray
    zone: np.ndarray
    role: np.ndarray
    tether_k: np.ndarray

    bond_i: np.ndarray
    bond_j: np.ndarray
    bond_kr: np.ndarray
    bond_req: np.ndarray

    ang_i: np.ndarray
    ang_j: np.ndarray
    ang_k: np.ndarray
    ang_kt: np.ndarray
    ang_eq: np.ndarray

    dih_i: np.ndarray
    dih_j: np.ndarray
    dih_k: np.ndarray
    dih_l: np.ndarray
    dih_pk: np.ndarray
    dih_n: np.ndarray
    dih_phase: np.ndarray
    dih_is_pep: np.ndarray

    p14_i: np.ndarray
    p14_j: np.ndarray
    p14_qprod: np.ndarray
    p14_rmin: np.ndarray
    p14_eps: np.ndarray

    excl_offsets: np.ndarray
    excl_targets: np.ndarray

    con_i: np.ndarray
    con_j: np.ndarray
    con_d2: np.ndarray

    wat_o: np.ndarray
    wat_h1: np.ndarray
    wat_h2: np.ndarray

    fg_heavy: np.ndarray
    fg_nh: np.ndarray
    fg_h_flat: np.ndarray

    cap_anchors: np.ndarray
    shell_water_oxygens: np.ndarray
    cap_radius: float = 6.5
    cap_k: float = 3.0
    cap_soft_beta: float = 1.5

    box: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cutoff: float = 9.0
    switch_on: float = 7.5
    ewald_alpha: float = 0.348
    pme_grid: tuple[int, int, int] = (32, 32, 32)

    gb_cutoff: float = 16.0
    eps_in: float = 1.0
    eps_out: float = 78.5
    ionic_strength: float = 0.15
    sasa_gamma: float = 0.0054
    filter_alpha: float = 0.35

    res_diel: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    zinc_sites: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_pdb_block(self, coords_override: np.ndarray | None = None, include_solvent: bool = False) -> str:
        xyz = self.coords if coords_override is None else coords_override
        lines: list[str] = []
        serial = 1
        for i, am in enumerate(self.atoms):
            if not include_solvent and am.role == ROLE_SOLVENT:
                continue
            rec = "HETATM" if am.role == ROLE_ION else "ATOM  "
            elem = am.atom_name[0] if am.role != ROLE_ION else am.res_name[:2].strip()
            aname_fmt = f" {am.atom_name:<3s}" if len(am.atom_name) < 4 else f"{am.atom_name:<4s}"
            x, y, z = xyz[i]
            lines.append(
                f"{rec}{serial:5d} {aname_fmt} {am.res_name:>3s} {am.chain_id:1s}{am.res_seq:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}"
            )
            serial = (serial % 99999) + 1
        lines.append("END")
        return "\n".join(lines) + "\n"


def _kabsch_align(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find proper rotation R (det=+1) and centroids (cP, cQ) mapping P -> Q: x_aligned = (x - cP) @ R + cQ."""
    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)
    X = P - cP
    Y = Q - cQ
    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(U @ Vt)
    D = np.diag([1.0, 1.0, 1.0 if d >= 0.0 else -1.0])
    R = U @ D @ Vt
    return R, cP, cQ


def _place_point_from_local_frame(
    target_ref: np.ndarray,
    template_pts: np.ndarray,
    actual_pts: np.ndarray,
) -> np.ndarray:
    if len(template_pts) >= 3:
        R, cP, cQ = _kabsch_align(template_pts, actual_pts)
        return (target_ref - cP) @ R + cQ
    if len(template_pts) == 2:
        # Align bond vector p0 -> p1 and pick deterministic orthogonal frame
        v_t = template_pts[1] - template_pts[0]
        v_a = actual_pts[1] - actual_pts[0]
        nt = max(np.linalg.norm(v_t), 1e-12)
        na = max(np.linalg.norm(v_a), 1e-12)
        u_t = v_t / nt
        u_a = v_a / na
        axis = np.cross(u_t, u_a)
        sin_a = np.linalg.norm(axis)
        cos_a = float(np.dot(u_t, u_a))
        if sin_a < 1e-8:
            R = np.eye(3) if cos_a > 0 else np.diag([1.0, -1.0, -1.0])
        else:
            k = axis / sin_a
            K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
            R = np.eye(3) + sin_a * K + (1.0 - cos_a) * (K @ K)
        return (target_ref - template_pts[0]) @ R.T + actual_pts[0]
    return actual_pts[0] + (target_ref - template_pts[0])


def _rot_axis_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    u = axis / max(float(np.linalg.norm(axis)), 1e-12)
    K = np.array([[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]], dtype=np.float64)
    return np.eye(3) + math.sin(angle_rad) * K + (1.0 - math.cos(angle_rad)) * (K @ K)


def _reconstruct_missing_heavy_atoms(
    r: dict[str, Any],
    tmpl: dict[str, dict[str, Any]],
    env_heavy_xyz: np.ndarray,
) -> list[str]:
    """Deterministically reconstruct missing carbonyl O or disordered sidechain heavy atoms via
    local Kabsch template alignment + 12-angle chi1/chi2 steric-clearance rotamer search.
    """
    pdb_atoms = r["atoms"]
    rname = r["res_name"]
    expected_heavy = [an for an, info in tmpl.items() if not info["atype"].startswith("H")]
    missing_init = [an for an in expected_heavy if an not in pdb_atoms]
    if not missing_init:
        return []

    if rname not in ("ACE", "NME"):
        for req_bb in ("N", "CA", "C"):
            if req_bb not in pdb_atoms:
                raise TopologyBuildError(
                    f"Residue {r['chain_id']}:{rname}{r['res_seq']} lacks core backbone trace atom {req_bb!r}"
                )
    else:
        raise TopologyBuildError(
            f"Cap residue {r['chain_id']}:{rname}{r['res_seq']} is missing required heavy atoms: {missing_init}"
        )

    rebuilt: list[str] = []

    # 1. Reconstruct carbonyl O if missing
    if "O" in tmpl and "O" not in pdb_atoms:
        t_pts = np.array([tmpl["N"]["xyz"], tmpl["CA"]["xyz"], tmpl["C"]["xyz"]], dtype=np.float64)
        a_pts = np.array([pdb_atoms["N"], pdb_atoms["CA"], pdb_atoms["C"]], dtype=np.float64)
        pdb_atoms["O"] = _place_point_from_local_frame(np.array(tmpl["O"]["xyz"], dtype=np.float64), t_pts, a_pts)
        rebuilt.append("O")

    # 2. Reconstruct OXT on C-terminus if missing
    if "OXT" in tmpl and "OXT" not in pdb_atoms and "O" in pdb_atoms:
        t_pts = np.array([tmpl["CA"]["xyz"], tmpl["C"]["xyz"], tmpl["O"]["xyz"]], dtype=np.float64)
        a_pts = np.array([pdb_atoms["CA"], pdb_atoms["C"], pdb_atoms["O"]], dtype=np.float64)
        pdb_atoms["OXT"] = _place_point_from_local_frame(np.array(tmpl["OXT"]["xyz"], dtype=np.float64), t_pts, a_pts)

    # 3. Reconstruct chiral CB from (N, CA, C) if missing
    if "CB" in tmpl and "CB" not in pdb_atoms:
        t_pts = np.array([tmpl["N"]["xyz"], tmpl["CA"]["xyz"], tmpl["C"]["xyz"]], dtype=np.float64)
        a_pts = np.array([pdb_atoms["N"], pdb_atoms["CA"], pdb_atoms["C"]], dtype=np.float64)
        pdb_atoms["CB"] = _place_point_from_local_frame(np.array(tmpl["CB"]["xyz"], dtype=np.float64), t_pts, a_pts)
        rebuilt.append("CB")

    missing_sc = [an for an in expected_heavy if an not in pdb_atoms]
    if not missing_sc:
        return rebuilt

    # 4. Align known heavy atoms (>= 4: N, CA, C, O, CB, ...) onto AMBER template
    known_h = [an for an in expected_heavy if an in pdb_atoms]
    t_known = np.array([tmpl[an]["xyz"] for an in known_h], dtype=np.float64)
    a_known = np.array([pdb_atoms[an] for an in known_h], dtype=np.float64)
    R0, cP, cQ = _kabsch_align(t_known, a_known)
    base_sc = {
        an: (np.array(tmpl[an]["xyz"], dtype=np.float64) - cP) @ R0 + cQ
        for an in missing_sc
    }

    # 5. Deterministic chi1/chi2 steric-clearance rotamer search if CB->... atoms were missing
    if "CB" in pdb_atoms and len(env_heavy_xyz) > 0:
        ca_pos = pdb_atoms["CA"]
        cb_pos = pdb_atoms["CB"]
        own_core = np.array([pdb_atoms[k] for k in ("N", "CA", "C", "O", "CB") if k in pdb_atoms], dtype=np.float64)
        # Exclude residue's own backbone+CB from steric environment
        d_own = np.min(np.linalg.norm(env_heavy_xyz[:, None, :] - own_core[None, :, :], axis=-1), axis=1)
        ext_env = env_heavy_xyz[d_own > 0.05]

        if len(ext_env) > 0:
            axis_chi1 = cb_pos - ca_pos
            # Test canonical staggered rotamers first, then full 30-deg grid
            chi1_grid_deg = [0.0, 120.0, -120.0, 60.0, -60.0, 180.0, 30.0, -30.0, 90.0, -90.0, 150.0, -150.0]
            cg_candidates = [g for g in ("CG", "CG1", "OG", "OG1", "SG") if g in tmpl]
            cg_name = cg_candidates[0] if cg_candidates else None
            chi2_grid_deg = [0.0, 120.0, -120.0] if (cg_name and len(missing_sc) >= 2) else [0.0]

            best_min_d = -1.0
            best_coords = base_sc
            for ang1 in chi1_grid_deg:
                R1 = _rot_axis_matrix(axis_chi1, math.radians(ang1))
                rot1 = {
                    an: (pos - cb_pos) @ R1.T + cb_pos
                    for an, pos in base_sc.items()
                }
                for ang2 in chi2_grid_deg:
                    if abs(ang2) < 1e-6 or cg_name is None:
                        cand = rot1
                    else:
                        cg_p = rot1[cg_name] if cg_name in rot1 else pdb_atoms[cg_name]
                        R2 = _rot_axis_matrix(cg_p - cb_pos, math.radians(ang2))
                        cand = {
                            an: (pos if an in (cg_name, "CG2") else ((pos - cg_p) @ R2.T + cg_p))
                            for an, pos in rot1.items()
                        }
                    pts = np.array(list(cand.values()), dtype=np.float64)
                    min_d = float(np.min(np.linalg.norm(pts[:, None, :] - ext_env[None, :, :], axis=-1)))
                    if min_d > best_min_d:
                        best_min_d = min_d
                        best_coords = cand
                    if best_min_d >= 3.00 and abs(ang2) < 1e-6:
                        break
                if best_min_d >= 3.00:
                    break
            base_sc = best_coords

    for an in missing_sc:
        pdb_atoms[an] = base_sc[an]
        rebuilt.append(an)
    return rebuilt


def _parse_pdb_residues(
    pdb_text: str,
    selected_chains: set[str] | None,
    metal_handling: str,
    allow_strip_hetatm: bool,
    allow_ptm_and_caps: bool = True,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[str]]:
    templates = load_amber_templates()
    valid_aa = {v["base_resname"] for v in templates.values()} | {"HIS"}

    chains: dict[str, list[dict[str, Any]]] = {}
    res_lookup: dict[tuple[str, int, str], dict[str, Any]] = {}
    ions: list[dict[str, Any]] = []
    warnings: list[str] = []

    for raw in pdb_text.splitlines():
        rec = raw[:6].strip().upper()
        if rec in ("ENDMDL",):
            break  # Only read first NMR model deterministically
        if rec not in ("ATOM", "HETATM"):
            continue
        altloc = raw[16:17].strip() if len(raw) > 16 else ""
        if altloc not in ("", "A", "1"):
            continue

        aname = raw[12:16].strip().upper()
        rname = raw[17:20].strip().upper()
        rname = STANDARD_AA_ALIASES.get(rname, rname)
        chain_id = (raw[21:22].strip() if len(raw) > 21 else "") or "A"
        if selected_chains is not None and chain_id not in selected_chains:
            continue
        icode = raw[26:27].strip() if len(raw) > 26 else ""
        try:
            res_seq = int(raw[22:26].strip())
            x = float(raw[30:38].strip())
            y = float(raw[38:46].strip())
            z = float(raw[46:54].strip())
        except ValueError as exc:
            raise TopologyBuildError(f"Malformed PDB coordinate line: {raw!r}") from exc

        if rname in WATER_RESNAMES:
            continue

        if rec == "HETATM":
            if rname in RETAINED_IONS and metal_handling == "retain":
                ions.append({
                    "chain_id": chain_id,
                    "res_seq": res_seq,
                    "res_name": rname,
                    "atom_name": aname,
                    "xyz": np.array([x, y, z], dtype=np.float64),
                })
                continue
            if rname in POLYMERIC_PTM_CAPS:
                if not allow_ptm_and_caps:
                    raise TopologyBuildError(
                        f"Modified/cap residue {rname} ({chain_id}:{res_seq}) rejected because allow_ptm_and_caps=False."
                    )
                # Fall through to polymeric residue assembly below!
            elif allow_strip_hetatm:
                warnings.append(f"Stripped HETATM {rname} {chain_id}:{res_seq}")
                continue
            else:
                raise TopologyBuildError(
                    f"Unapproved HETATM residue {rname} ({chain_id}:{res_seq}); "
                    f"pass allow_strip_hetatm=True or metal_handling='retain' for ZN/CL/NA."
                )

        if rname in POLYMERIC_PTM_CAPS and not allow_ptm_and_caps:
            raise TopologyBuildError(
                f"Modified/cap residue {rname} ({chain_id}:{res_seq}) rejected because allow_ptm_and_caps=False."
            )

        if rname not in valid_aa:
            raise TopologyBuildError(f"Unsupported or modified residue {rname} at {chain_id}:{res_seq}")

        if rname == "ILE" and aname == "CD":
            aname = "CD1"
        else:
            aname = ATOM_NAME_ALIASES.get(aname, aname)

        key = (chain_id, res_seq, icode)
        if key not in res_lookup:
            entry = {
                "chain_id": chain_id,
                "res_seq": res_seq,
                "icode": icode,
                "res_name": rname,
                "atoms": {},
            }
            res_lookup[key] = entry
            chains.setdefault(chain_id, []).append(entry)
        res_lookup[key]["atoms"][aname] = np.array([x, y, z], dtype=np.float64)

    if not chains:
        raise TopologyBuildError("No valid protein/peptide ATOM records found in input PDB.")
    return chains, ions, warnings


def _next_pow2_at_least(val: float, min_grid: int = 16) -> int:
    n = max(min_grid, int(math.ceil(val)))
    p = 16
    while p < n:
        p <<= 1
    return p


def _build_from_parsed_chains(
    chains: dict[str, list[dict[str, Any]]],
    ions: list[dict[str, Any]],
    peptide_chains: set[str],
    solvent_mode: str,
    allow_chain_breaks: bool,
    box_padding: float,
    shell_waters_cutoff: float,
    max_shell_waters: int,
    active_radius: float,
    buffer_radius: float | None,
    ionic_strength: float,
    seed: int,
    warnings: list[str],
    zinc_coord_mode: str = "harmonic_restraint",
    reconstruct_missing_heavy: bool = False,
) -> BuiltSystem:
    templates = load_amber_templates()
    rng = np.random.default_rng(seed)

    # 1. Fail-closed chain continuity & heavy-atom completeness checks + CYX / Zn-His assignment
    all_res_flat: list[dict[str, Any]] = []
    for cid, res_list in chains.items():
        if len(res_list) < 2:
            raise TopologyBuildError(f"Chain {cid} has < 2 residues ({len(res_list)}); minimum length is 2.")
        for idx, r in enumerate(res_list):
            r["is_nterm"] = (idx == 0)
            r["is_cterm"] = (idx == len(res_list) - 1)
            if idx > 0:
                prev = res_list[idx - 1]
                if "C" not in prev["atoms"] or "N" not in r["atoms"]:
                    raise TopologyBuildError(f"Missing backbone C/N at {cid}:{prev['res_seq']}->{r['res_seq']}")
                cn_dist = float(np.linalg.norm(prev["atoms"]["C"] - r["atoms"]["N"]))
                if cn_dist > 2.30:
                    if not allow_chain_breaks:
                        raise TopologyBuildError(
                            f"Chain break detected at {cid}:{prev['res_seq']}->{r['res_seq']} "
                            f"(C-N = {cn_dist:.2f} A > 2.30 A). Pass allow_chain_breaks=True if authorized."
                        )
                    prev["is_cterm"] = True
                    r["is_nterm"] = True
                    warnings.append(f"Treated gap {cid}:{prev['res_seq']}->{r['res_seq']} ({cn_dist:.2f} A) as chain termini.")
                else:
                    # Handle ACE (caps next residue's N) and NME (caps prev residue's C)
                    if prev["res_name"] == "ACE":
                        prev["is_cterm"] = False
                        r["is_nterm"] = False
                    if r["res_name"] == "NME":
                        prev["is_cterm"] = False
                        r["is_nterm"] = False
            all_res_flat.append(r)

    # Detect CYX disulfide bridges (SG-SG <= 2.35 A)
    cys_list = [r for r in all_res_flat if r["res_name"] in ("CYS", "CYX") and "SG" in r["atoms"]]
    disulfide_pairs: list[tuple[int, int]] = []
    res_id_map = {id(r): idx for idx, r in enumerate(all_res_flat)}
    paired_cys: set[int] = set()
    for i_c in range(len(cys_list)):
        for j_c in range(i_c + 1, len(cys_list)):
            r1, r2 = cys_list[i_c], cys_list[j_c]
            idx1, idx2 = res_id_map[id(r1)], res_id_map[id(r2)]
            if idx1 in paired_cys or idx2 in paired_cys:
                continue
            d_sg = float(np.linalg.norm(r1["atoms"]["SG"] - r2["atoms"]["SG"]))
            if d_sg <= 2.35:
                r1["res_name"] = "CYX"
                r2["res_name"] = "CYX"
                paired_cys.add(idx1)
                paired_cys.add(idx2)
                disulfide_pairs.append((idx1, idx2))

    # Filter PDB HETATM ions to retain only protein-chelated structural divalent cofactors
    # (stripping unchelated crystallization precipitant salts like Cl-, Na+, or outer-sphere [Mg(H2O)6]2+)
    if ions:
        prot_nos_pts = [
            pos for r_it in all_res_flat
            for an_it, pos in r_it["atoms"].items()
            if an_it[:1] in ("N", "O", "S")
        ]
        prot_o_pts = [
            pos for r_it in all_res_flat
            for an_it, pos in r_it["atoms"].items()
            if an_it[:1] == "O"
        ]
        nos_arr = np.array(prot_nos_pts, dtype=np.float64) if prot_nos_pts else np.zeros((0, 3), dtype=np.float64)
        o_arr = np.array(prot_o_pts, dtype=np.float64) if prot_o_pts else np.zeros((0, 3), dtype=np.float64)
        chelated_ions: list[dict[str, Any]] = []
        for ion in ions:
            irname = ion["res_name"]
            ixyz = ion["xyz"]
            if irname == "ZN" and len(nos_arr) > 0:
                n_coord = int(np.sum(np.linalg.norm(nos_arr - ixyz[None, :], axis=-1) <= 2.80))
                if n_coord >= 2:
                    chelated_ions.append(ion)
                    continue
            elif irname in ("CA", "MG") and len(o_arr) > 0:
                n_coord = int(np.sum(np.linalg.norm(o_arr - ixyz[None, :], axis=-1) <= 2.85))
                if n_coord >= 3:
                    chelated_ions.append(ion)
                    continue
            warnings.append(
                f"Stripped unchelated crystallization ion {irname} {ion['chain_id']}:{ion['res_seq']}"
            )
        ions = chelated_ions

    # Deterministic Histidine tautomer & Cysteine thiolate (CYM) selection near retained ZN2+ ions
    zinc_sites: list[dict[str, Any]] = []
    zn_ions = [ion for ion in ions if ion["res_name"] == "ZN"]
    for zn in zn_ions:
        z_xyz = zn["xyz"]
        donors: list[dict[str, Any]] = []
        for r in all_res_flat:
            if r["res_name"] in ("HIS", "HID", "HIE", "HIP"):
                d_nd1 = float(np.linalg.norm(r["atoms"]["ND1"] - z_xyz)) if "ND1" in r["atoms"] else 999.0
                d_ne2 = float(np.linalg.norm(r["atoms"]["NE2"] - z_xyz)) if "NE2" in r["atoms"] else 999.0
                if min(d_nd1, d_ne2) <= 2.65:
                    if d_nd1 < d_ne2:
                        r["res_name"] = "HIE"  # ND1 unprotonated lone pair coordinates Zn2+!
                        donors.append({"res": f"{r['chain_id']}:HIS{r['res_seq']}", "atom": "ND1", "init_dist": d_nd1, "res_obj_id": id(r)})
                    else:
                        r["res_name"] = "HID"  # NE2 unprotonated lone pair coordinates Zn2+!
                        donors.append({"res": f"{r['chain_id']}:HIS{r['res_seq']}", "atom": "NE2", "init_dist": d_ne2, "res_obj_id": id(r)})
            elif r["res_name"] == "ASP":
                for an in ("OD1", "OD2"):
                    if an in r["atoms"]:
                        d = float(np.linalg.norm(r["atoms"][an] - z_xyz))
                        if d <= 2.65:
                            donors.append({"res": f"{r['chain_id']}:ASP{r['res_seq']}", "atom": an, "init_dist": d, "res_obj_id": id(r)})
            elif r["res_name"] == "GLU":
                for an in ("OE1", "OE2"):
                    if an in r["atoms"]:
                        d = float(np.linalg.norm(r["atoms"][an] - z_xyz))
                        if d <= 2.65:
                            donors.append({"res": f"{r['chain_id']}:GLU{r['res_seq']}", "atom": an, "init_dist": d, "res_obj_id": id(r)})
            elif r["res_name"] in ("CYS", "CYM") and "SG" in r["atoms"]:
                d = float(np.linalg.norm(r["atoms"]["SG"] - z_xyz))
                if d <= 2.65:
                    r["res_name"] = "CYM"  # Deprotonated zinc-finger thiolate (-1 e, no HG)!
                    donors.append({"res": f"{r['chain_id']}:CYM{r['res_seq']}", "atom": "SG", "init_dist": d, "res_obj_id": id(r)})
        zinc_sites.append({
            "chain_id": zn["chain_id"],
            "res_seq": zn["res_seq"],
            "donors": donors,
            "ion_obj_id": id(zn),
        })

    # Remaining generic HIS default to neutral HIE (standard AMBER pH 7.4 default)
    for r in all_res_flat:
        if r["res_name"] == "HIS":
            r["res_name"] = "HIE"

    # 2. Verify heavy atoms & reconstruct missing hydrogens + C-term OXT
    atoms_meta: list[AtomMeta] = []
    coords_list: list[np.ndarray] = []
    charges_list: list[float] = []
    rmin_list: list[float] = []
    eps_list: list[float] = []
    gbr_list: list[float] = []
    gbs_list: list[float] = []
    mass_list: list[float] = []
    role_list: list[int] = []

    res_atom_maps: list[dict[str, int]] = []
    bonds_raw: list[tuple[int, int]] = []
    res_diel_list: list[float] = []

    # Snapshot all present heavy-atom coordinates for clash-free sidechain rotamer reconstruction
    all_present_heavy_list = [
        pos for r_it in all_res_flat for an_it, pos in r_it["atoms"].items() if not an_it.startswith("H")
    ] + [ion["xyz"] for ion in ions]
    env_heavy_xyz = np.array(all_present_heavy_list, dtype=np.float64) if all_present_heavy_list else np.zeros((0, 3), dtype=np.float64)

    for r_idx, r in enumerate(all_res_flat):
        if r["res_name"] in ("ACE", "NME"):
            tname = r["res_name"]
        elif r["is_nterm"] and not r["is_cterm"]:
            tname = f"N{r['res_name']}"
        elif r["is_cterm"] and not r["is_nterm"]:
            tname = f"C{r['res_name']}"
        else:
            tname = r["res_name"]

        tmpl = templates[tname]["atoms"]
        pdb_atoms = r["atoms"]

        # Reconstruct OXT on C-terminus if missing in PDB
        if "OXT" in tmpl and "OXT" not in pdb_atoms:
            if all(k in pdb_atoms for k in ("CA", "C", "O")):
                t_pts = np.array([tmpl["CA"]["xyz"], tmpl["C"]["xyz"], tmpl["O"]["xyz"]], dtype=np.float64)
                a_pts = np.array([pdb_atoms["CA"], pdb_atoms["C"], pdb_atoms["O"]], dtype=np.float64)
                pdb_atoms["OXT"] = _place_point_from_local_frame(
                    np.array(tmpl["OXT"]["xyz"], dtype=np.float64), t_pts, a_pts
                )

        expected_heavy = [an for an, info in tmpl.items() if not info["atype"].startswith("H")]
        missing_heavy = [an for an in expected_heavy if an not in pdb_atoms]
        if missing_heavy:
            if not reconstruct_missing_heavy:
                raise TopologyBuildError(
                    f"Residue {r['chain_id']}:{r['res_name']}{r['res_seq']} is missing required heavy atoms: {missing_heavy}"
                )
            rebuilt = _reconstruct_missing_heavy_atoms(r, tmpl, env_heavy_xyz)
            warnings.append(
                f"Reconstructed missing heavy atoms on {r['chain_id']}:{r['res_name']}{r['res_seq']}: {rebuilt}"
            )
            if len(rebuilt) > 0:
                new_pts = np.array([pdb_atoms[an] for an in rebuilt], dtype=np.float64)
                env_heavy_xyz = np.vstack([env_heavy_xyz, new_pts]) if len(env_heavy_xyz) > 0 else new_pts

        # Reconstruct missing hydrogens via local heavy-atom Kabsch frame (1st + 2nd shell heavy atoms!)
        placed_coords: dict[str, np.ndarray] = {an: pdb_atoms[an].copy() for an in expected_heavy}
        for an, info in tmpl.items():
            if not info["atype"].startswith("H"):
                continue
            if an in pdb_atoms:
                placed_coords[an] = pdb_atoms[an].copy()
                continue

            parent = info["bonds"][0]
            # Special exact bisecting plane placement for internal amide H if previous residue C exists!
            if an == "H" and parent == "N" and not r["is_nterm"] and r_idx > 0 and not all_res_flat[r_idx - 1]["is_cterm"]:
                prev_c = all_res_flat[r_idx - 1]["atoms"]["C"]
                n_pos = placed_coords["N"]
                ca_pos = placed_coords["CA"]
                u1 = prev_c - n_pos
                u2 = ca_pos - n_pos
                u1 /= max(np.linalg.norm(u1), 1e-12)
                u2 /= max(np.linalg.norm(u2), 1e-12)
                bisec = -(u1 + u2)
                nb = np.linalg.norm(bisec)
                if nb > 1e-8:
                    placed_coords["H"] = n_pos + 1.010 * (bisec / nb)
                    continue

            # Gather heavy reference set: parent + heavy neighbors of parent + 2nd-shell heavy neighbors
            ref_names: list[str] = [parent]
            for nb1 in tmpl[parent]["bonds"]:
                if not tmpl[nb1]["atype"].startswith("H") and nb1 not in ref_names:
                    ref_names.append(nb1)
            if len(ref_names) < 3:
                for nb1 in list(ref_names[1:]):
                    for nb2 in tmpl[nb1]["bonds"]:
                        if not tmpl[nb2]["atype"].startswith("H") and nb2 not in ref_names:
                            ref_names.append(nb2)
                            if len(ref_names) >= 4:
                                break
                    if len(ref_names) >= 4:
                        break

            t_pts = np.array([tmpl[rn]["xyz"] for rn in ref_names], dtype=np.float64)
            a_pts = np.array([placed_coords[rn] for rn in ref_names], dtype=np.float64)
            placed_coords[an] = _place_point_from_local_frame(
                np.array(info["xyz"], dtype=np.float64), t_pts, a_pts
            )

        role_val = ROLE_PEPTIDE if r["chain_id"] in peptide_chains else ROLE_RECEPTOR
        amap: dict[str, int] = {}
        for an in sorted(tmpl.keys()):
            info = tmpl[an]
            at = info["atype"]
            props = ATOM_TYPE_PROPERTIES[at]
            g_idx = len(atoms_meta)
            amap[an] = g_idx
            atoms_meta.append(AtomMeta(
                index=g_idx,
                chain_id=r["chain_id"],
                res_seq=r["res_seq"],
                res_name=r["res_name"],
                atom_name=an,
                atype=at,
                is_heavy=(not at.startswith("H")),
                role=role_val,
            ))
            coords_list.append(placed_coords[an])
            charges_list.append(float(info["charge"]))
            rmin_list.append(max(float(info["rmin_half"]), float(props["rmin_half"])))
            eps_list.append(float(props["eps"]))
            gbr_list.append(float(props["gb_r"]))
            gbs_list.append(float(props["gb_s"]))
            mass_list.append(float(props["mass"]))
            role_list.append(role_val)
            base_eps_r = RESIDUE_INTERIOR_DIELECTRIC.get(r["res_name"], 2.5)
            # Charged zwitterionic N-terminal (+1 e) or C-terminal (-1 e) residues get uniform formal-charge screening (>= 5.0)
            # across the entire residue so sum_{i in res} q_i / sqrt(eps_i) preserves exact multipole consistency!
            if (r["is_nterm"] or r["is_cterm"]) and r["res_name"] not in ("ACE", "NME"):
                base_eps_r = max(base_eps_r, 4.0)
            res_diel_list.append(float(base_eps_r))

        # Intra-residue covalent bonds
        for an, info in tmpl.items():
            ia = amap[an]
            for bn in info["bonds"]:
                ib = amap[bn]
                if ia < ib:
                    bonds_raw.append((ia, ib))
        res_atom_maps.append(amap)

    # Inter-residue peptide bonds C(i) - N(i+1)
    for r_idx in range(len(all_res_flat) - 1):
        r_cur = all_res_flat[r_idx]
        r_nxt = all_res_flat[r_idx + 1]
        if r_cur["chain_id"] == r_nxt["chain_id"] and not r_cur["is_cterm"] and not r_nxt["is_nterm"]:
            ic = res_atom_maps[r_idx]["C"]
            in_ = res_atom_maps[r_idx + 1]["N"]
            bonds_raw.append((min(ic, in_), max(ic, in_)))

    # Disulfide bonds SG - SG
    for idx1, idx2 in disulfide_pairs:
        isg1 = res_atom_maps[idx1]["SG"]
        isg2 = res_atom_maps[idx2]["SG"]
        bonds_raw.append((min(isg1, isg2), max(isg1, isg2)))

    # Fail-closed steric clash check on input heavy atoms (< 0.80 A between non-bonded heavy atoms)
    solute_heavy_idx = [am.index for am in atoms_meta if am.is_heavy]
    bonded_set = set(bonds_raw)
    if len(solute_heavy_idx) > 1:
        sh_xyz = np.array([coords_list[i] for i in solute_heavy_idx], dtype=np.float64)
        # Fast vectorized check in chunks
        for s0 in range(0, len(solute_heavy_idx), 256):
            s1 = min(len(solute_heavy_idx), s0 + 256)
            diff = sh_xyz[s0:s1, None, :] - sh_xyz[None, :, :]
            d2 = np.sum(diff * diff, axis=-1)
            rows, cols = np.where(d2 < 0.64)  # 0.80^2 = 0.64 A^2
            for r_loc, c_glob in zip(rows, cols):
                r_glob = s0 + int(r_loc)
                if r_glob >= int(c_glob):
                    continue
                ia, ib = solute_heavy_idx[r_glob], solute_heavy_idx[int(c_glob)]
                if (min(ia, ib), max(ia, ib)) in bonded_set:
                    continue
                raise TopologyBuildError(
                    f"Severe heavy-atom clash ({math.sqrt(float(d2[r_loc, c_glob])):.2f} A < 0.80 A) "
                    f"between {atoms_meta[ia].chain_id}:{atoms_meta[ia].res_name}{atoms_meta[ia].res_seq}:{atoms_meta[ia].atom_name} "
                    f"and {atoms_meta[ib].chain_id}:{atoms_meta[ib].res_name}{atoms_meta[ib].res_seq}:{atoms_meta[ib].atom_name}"
                )

    # Append retained structural ions (ZN2+, CL-, NA+)
    ion_idx_by_obj: dict[int, int] = {}
    for ion in ions:
        itype = RETAINED_IONS[ion["res_name"]]
        props = ATOM_TYPE_PROPERTIES[itype]
        g_idx = len(atoms_meta)
        ion_idx_by_obj[id(ion)] = g_idx
        atoms_meta.append(AtomMeta(
            index=g_idx,
            chain_id=ion["chain_id"],
            res_seq=ion["res_seq"],
            res_name=ion["res_name"],
            atom_name=ion["atom_name"],
            atype=itype,
            is_heavy=True,
            role=ROLE_ION,
        ))
        coords_list.append(ion["xyz"].copy())
        charges_list.append(float(props["charge"]))
        rmin_list.append(float(props["rmin_half"]))
        eps_list.append(float(props["eps"]))
        gbr_list.append(float(props["gb_r"]))
        gbs_list.append(float(props["gb_s"]))
        mass_list.append(float(props["mass"]))
        role_list.append(ROLE_ION)
        res_diel_list.append(5.0)

    # Record atom indices for ZincSite donor tracking
    res_obj_to_amap = {id(all_res_flat[idx]): res_atom_maps[idx] for idx in range(len(all_res_flat))}
    for zs in zinc_sites:
        zs["zn_atom_index"] = ion_idx_by_obj[zs.pop("ion_obj_id")]
        for d in zs["donors"]:
            amap = res_obj_to_amap[d.pop("res_obj_id")]
            d["donor_atom_index"] = amap[d["atom"]]

    n_solute = len(atoms_meta)
    solute_xyz = np.array(coords_list, dtype=np.float64)

    # 3. Assign Active-Shell Zones (A / B / C) & Softmin Cap Anchors
    zone_arr_solute = np.zeros(n_solute, dtype=np.int32)
    tether_k_solute = np.zeros(n_solute, dtype=np.float64)

    pep_heavy_indices = [am.index for am in atoms_meta if am.role == ROLE_PEPTIDE and am.is_heavy]
    ion_indices = [am.index for am in atoms_meta if am.role == ROLE_ION]

    if solvent_mode == "active-shell" and pep_heavy_indices:
        pep_xyz = solute_xyz[pep_heavy_indices]
        pocket_ion_indices: list[int] = []
        for g_ion in ion_indices:
            d_ion = float(np.min(np.linalg.norm(pep_xyz - solute_xyz[g_ion][None, :], axis=-1)))
            if d_ion <= active_radius + 4.0 or buffer_radius is None:
                zone_arr_solute[g_ion] = ZONE_ACTIVE_SOLUTE
                pocket_ion_indices.append(g_ion)
            elif d_ion <= buffer_radius:
                zone_arr_solute[g_ion] = ZONE_BUFFER_RIM
                u_ion = min(1.0, max(0.0, (d_ion - active_radius) / max(buffer_radius - active_radius, 1e-6)))
                tether_k_solute[g_ion] = 1.5 * (10.0 * u_ion**3 - 15.0 * u_ion**4 + 6.0 * u_ion**5)
            else:
                zone_arr_solute[g_ion] = ZONE_SCAFFOLD

        focus_xyz = solute_xyz[pep_heavy_indices + pocket_ion_indices]
        # Classify each receptor residue by its minimum heavy-atom distance to peptide/pocket-ion focus
        for r_idx, r in enumerate(all_res_flat):
            amap = res_atom_maps[r_idx]
            if r["chain_id"] in peptide_chains:
                for g_i in amap.values():
                    zone_arr_solute[g_i] = ZONE_ACTIVE_SOLUTE
                continue
            r_heavy = [g_i for an, g_i in amap.items() if atoms_meta[g_i].is_heavy]
            r_xyz = solute_xyz[r_heavy]
            dmat = np.linalg.norm(r_xyz[:, None, :] - focus_xyz[None, :, :], axis=-1)
            dmin = float(np.min(dmat))
            if dmin <= active_radius or buffer_radius is None:
                for g_i in amap.values():
                    zone_arr_solute[g_i] = ZONE_ACTIVE_SOLUTE
            elif dmin <= buffer_radius:
                u = (dmin - active_radius) / max(buffer_radius - active_radius, 1e-6)
                u = min(1.0, max(0.0, u))
                # Smooth quintic ramp from 0.0 at active_radius to 0.8 kcal/(mol*A^2) on backbone at buffer_radius
                sw = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
                for an, g_i in amap.items():
                    zone_arr_solute[g_i] = ZONE_BUFFER_RIM
                    if an in ("N", "CA", "C"):
                        tether_k_solute[g_i] = 0.8 * sw
            else:
                for g_i in amap.values():
                    zone_arr_solute[g_i] = ZONE_SCAFFOLD
    elif solvent_mode == "active-shell" and ion_indices and buffer_radius is not None:
        focus_xyz = solute_xyz[ion_indices]
        for r_idx, r in enumerate(all_res_flat):
            amap = res_atom_maps[r_idx]
            r_heavy = [g_i for an, g_i in amap.items() if atoms_meta[g_i].is_heavy]
            dmin = float(np.min(np.linalg.norm(solute_xyz[r_heavy][:, None, :] - focus_xyz[None, :, :], axis=-1)))
            if dmin <= active_radius:
                for g_i in amap.values():
                    zone_arr_solute[g_i] = ZONE_ACTIVE_SOLUTE
            elif dmin <= buffer_radius:
                u = min(1.0, max(0.0, (dmin - active_radius) / max(buffer_radius - active_radius, 1e-6)))
                sw = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
                for an, g_i in amap.items():
                    zone_arr_solute[g_i] = ZONE_BUFFER_RIM
                    if an in ("N", "CA", "C"):
                        tether_k_solute[g_i] = 1.5 * sw
            else:
                for g_i in amap.values():
                    zone_arr_solute[g_i] = ZONE_SCAFFOLD

    # Cap anchors: C-alpha & polar heavy atoms in Zone A solute (compact ~40-150 anchor envelope!)
    cap_anchor_list: list[int] = []
    if solvent_mode == "active-shell":
        for am in atoms_meta:
            if zone_arr_solute[am.index] == ZONE_ACTIVE_SOLUTE and am.is_heavy:
                if am.role in (ROLE_PEPTIDE, ROLE_ION) or am.atom_name in ("CA", "CB", "N", "O", "OD1", "OE1", "ND1", "NE2", "OH", "NZ"):
                    cap_anchor_list.append(am.index)

    # 4. Solvation: Explicit PME Periodic Box OR Active-Shell Explicit Interfacial Hydration Shell
    wat_triplets: list[tuple[int, int, int]] = []
    shell_wat_oxygens: list[int] = []
    box_dims = (0.0, 0.0, 0.0)
    pme_grid = (32, 32, 32)

    # Ideal TIP3P geometry in local molecular frame: r_OH = 0.9572 A, angle = 104.52 deg
    half_ang = 0.5 * 104.52 * DEG2RAD
    tip3p_h1_local = np.array([0.9572 * math.sin(half_ang), 0.9572 * math.cos(half_ang), 0.0], dtype=np.float64)
    tip3p_h2_local = np.array([-0.9572 * math.sin(half_ang), 0.9572 * math.cos(half_ang), 0.0], dtype=np.float64)

    def _random_so3_matrix() -> np.ndarray:
        u1, u2, u3 = rng.random(3)
        q0 = math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2)
        q1 = math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2)
        q2 = math.sqrt(u1) * math.sin(2.0 * math.pi * u3)
        q3 = math.sqrt(u1) * math.cos(2.0 * math.pi * u3)
        return np.array([
            [1 - 2*(q2*q2 + q3*q3), 2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)],
            [2*(q1*q2 + q0*q3),     1 - 2*(q1*q1 + q3*q3), 2*(q2*q3 - q0*q1)],
            [2*(q1*q3 - q0*q2),     2*(q2*q3 + q0*q1),     1 - 2*(q1*q1 + q2*q2)],
        ], dtype=np.float64)

    def _append_tip3p_water(o_pos: np.ndarray, res_seq_num: int, is_shell: bool) -> None:
        R = _random_so3_matrix()
        h1_pos = o_pos + tip3p_h1_local @ R.T
        h2_pos = o_pos + tip3p_h2_local @ R.T
        io = len(atoms_meta)
        ih1 = io + 1
        ih2 = io + 2
        for idx_w, (an_w, at_w, q_w, pos_w) in enumerate([
            ("O", "OW", -0.8340, o_pos),
            ("H1", "HW", 0.4170, h1_pos),
            ("H2", "HW", 0.4170, h2_pos),
        ]):
            pr = ATOM_TYPE_PROPERTIES[at_w]
            atoms_meta.append(AtomMeta(
                index=io + idx_w,
                chain_id="W",
                res_seq=res_seq_num % 9999 + 1,
                res_name="HOH",
                atom_name=an_w,
                atype=at_w,
                is_heavy=(idx_w == 0),
                role=ROLE_SOLVENT,
            ))
            coords_list.append(pos_w)
            charges_list.append(q_w)
            rmin_list.append(float(pr["rmin_half"]))
            eps_list.append(float(pr["eps"]))
            gbr_list.append(float(pr["gb_r"]))
            gbs_list.append(float(pr["gb_s"]))
            mass_list.append(float(pr["mass"]))
            role_list.append(ROLE_SOLVENT)
            res_diel_list.append(1.0)
        bonds_raw.append((io, ih1))
        bonds_raw.append((io, ih2))
        wat_triplets.append((io, ih1, ih2))
        if is_shell:
            shell_wat_oxygens.append(io)

    if solvent_mode == "explicit-pme":
        min_c = solute_xyz.min(axis=0)
        max_c = solute_xyz.max(axis=0)
        span = max_c - min_c
        bx = max(22.0, float(span[0] + 2.0 * box_padding))
        by = max(22.0, float(span[1] + 2.0 * box_padding))
        bz = max(22.0, float(span[2] + 2.0 * box_padding))
        box_dims = (bx, by, bz)

        # Center solute inside [0, bx) x [0, by) x [0, bz)
        shift = 0.5 * np.array([bx, by, bz]) - 0.5 * (min_c + max_c)
        solute_xyz = solute_xyz + shift
        for i in range(n_solute):
            coords_list[i] = solute_xyz[i]

        pme_grid = (
            _next_pow2_at_least(bx * 1.0),
            _next_pow2_at_least(by * 1.0),
            _next_pow2_at_least(bz * 1.0),
        )

        # Liquid-density lattice a = 3.107 A (~30.0 A^3 per H2O)
        spacing = 3.107
        gx = np.arange(1.5, bx - 1.2, spacing)
        gy = np.arange(1.5, by - 1.2, spacing)
        gz = np.arange(1.5, bz - 1.2, spacing)
        grid_pts = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), axis=-1).reshape(-1, 3)

        sol_heavy_xyz = solute_xyz[[am.index for am in atoms_meta[:n_solute] if am.is_heavy]]
        # Keep non-overlapping water sites (>= 2.65 A from solute heavy atoms)
        valid_sites: list[np.ndarray] = []
        for c0 in range(0, len(grid_pts), 512):
            pts = grid_pts[c0:c0 + 512]
            diff = pts[:, None, :] - sol_heavy_xyz[None, :, :]
            # Minimum image check against solute near box faces
            diff[:, :, 0] -= bx * np.round(diff[:, :, 0] / bx)
            diff[:, :, 1] -= by * np.round(diff[:, :, 1] / by)
            diff[:, :, 2] -= bz * np.round(diff[:, :, 2] / bz)
            min_d2 = np.min(np.sum(diff * diff, axis=-1), axis=1)
            for k_pt, d2v in enumerate(min_d2):
                if d2v >= 2.65 * 2.65:
                    valid_sites.append(pts[k_pt])

        # Neutralize net solute charge + add physiological ionic_strength NaCl
        net_q = int(round(sum(charges_list[:n_solute])))
        vol_liters = (bx * by * bz) * 1.0e-27
        n_salt_pairs = max(0, int(round(ionic_strength * vol_liters * 6.02214076e23)))
        n_na = n_salt_pairs + (abs(net_q) if net_q < 0 else 0)
        n_cl = n_salt_pairs + (net_q if net_q > 0 else 0)
        total_ions_add = n_na + n_cl

        if total_ions_add < len(valid_sites):
            perm = rng.permutation(len(valid_sites))
            ion_site_indices = set(int(x) for x in perm[:total_ions_add])
            ion_Order = list(perm[:total_ions_add])
        else:
            ion_site_indices = set()
            ion_Order = []

        for idx_ion_slot, s_idx in enumerate(ion_Order):
            is_na = idx_ion_slot < n_na
            r_ion = "NA" if is_na else "CL"
            it_ion = "NA+" if is_na else "CL-"
            pr = ATOM_TYPE_PROPERTIES[it_ion]
            gi = len(atoms_meta)
            atoms_meta.append(AtomMeta(
                index=gi, chain_id="I", res_seq=idx_ion_slot + 1,
                res_name=r_ion, atom_name=r_ion, atype=it_ion, is_heavy=True, role=ROLE_ION
            ))
            coords_list.append(valid_sites[s_idx])
            charges_list.append(float(pr["charge"]))
            rmin_list.append(float(pr["rmin_half"]))
            eps_list.append(float(pr["eps"]))
            gbr_list.append(float(pr["gb_r"]))
            gbs_list.append(float(pr["gb_s"]))
            mass_list.append(float(pr["mass"]))
            role_list.append(ROLE_ION)
            res_diel_list.append(5.0)

        w_seq = 0
        for s_idx, o_pt in enumerate(valid_sites):
            if s_idx in ion_site_indices:
                continue
            _append_tip3p_water(o_pt, w_seq, is_shell=False)
            w_seq += 1

    elif solvent_mode == "active-shell" and max_shell_waters > 0 and cap_anchor_list:
        anc_xyz = solute_xyz[cap_anchor_list]
        sol_heavy_xyz = solute_xyz[[am.index for am in atoms_meta[:n_solute] if am.is_heavy]]
        min_a = anc_xyz.min(axis=0) - shell_waters_cutoff
        max_a = anc_xyz.max(axis=0) + shell_waters_cutoff
        spacing = 3.10
        gx = np.arange(min_a[0], max_a[0], spacing)
        gy = np.arange(min_a[1], max_a[1], spacing)
        gz = np.arange(min_a[2], max_a[2], spacing)
        if len(gx) > 0 and len(gy) > 0 and len(gz) > 0:
            grid_pts = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), axis=-1).reshape(-1, 3)
            candidates: list[tuple[float, np.ndarray]] = []
            for c0 in range(0, len(grid_pts), 512):
                pts = grid_pts[c0:c0 + 512]
                d2_sol = np.min(np.sum((pts[:, None, :] - sol_heavy_xyz[None, :, :])**2, axis=-1), axis=1)
                d2_anc = np.min(np.sum((pts[:, None, :] - anc_xyz[None, :, :])**2, axis=-1), axis=1)
                mask = (d2_sol >= 2.70 * 2.70) & (d2_anc <= shell_waters_cutoff * shell_waters_cutoff)
                for k_loc in np.where(mask)[0]:
                    candidates.append((float(d2_anc[k_loc]), pts[int(k_loc)]))
            # Pick the closest interfacial waters solvating the binding groove!
            candidates.sort(key=lambda item: item[0])
            for w_seq, (_, o_pt) in enumerate(candidates[:max_shell_waters]):
                _append_tip3p_water(o_pt, w_seq, is_shell=True)

    # Finalize per-atom arrays
    n_total = len(atoms_meta)
    coords_arr = np.array(coords_list, dtype=np.float64)
    zone_full = np.zeros(n_total, dtype=np.int32)
    zone_full[:n_solute] = zone_arr_solute
    for i in range(n_solute, n_total):
        zone_full[i] = ZONE_ACTIVE_WATER if role_list[i] == ROLE_SOLVENT else ZONE_ACTIVE_SOLUTE

    tether_full = np.zeros(n_total, dtype=np.float64)
    tether_full[:n_solute] = tether_k_solute

    # 5. Build Bond, Angle, Dihedral, Improper, 1-4, Exclusion, Constraint & Filter Tables
    adj: list[list[int]] = [[] for _ in range(n_total)]
    for ia, ib in bonds_raw:
        adj[ia].append(ib)
        adj[ib].append(ia)
    for nb_list in adj:
        nb_list.sort()

    wat_atom_set: set[int] = set()
    for wo, wh1, wh2 in wat_triplets:
        wat_atom_set.update((wo, wh1, wh2))

    # Bonds & X-H RATTLE constraints (exclude rigid SETTLE waters from harmonic bonds & X-H RATTLE!)
    b_i, b_j, b_kr, b_req = [], [], [], []
    c_i, c_j, c_d2 = [], [], []
    for ia, ib in bonds_raw:
        if ia in wat_atom_set and ib in wat_atom_set:
            continue
        t1, t2 = atoms_meta[ia].atype, atoms_meta[ib].atype
        dist_cur = float(np.linalg.norm(coords_arr[ia] - coords_arr[ib]))
        kr, req = lookup_bond_param(t1, t2, dist_cur)
        if not atoms_meta[ia].is_heavy or not atoms_meta[ib].is_heavy:
            # Constrain X-H covalent bond at exact AMBER equilibrium length via RATTLE!
            # Also fix initial H position along bond vector to exact req so t=0 satisfies manifold to 1e-15!
            heavy_i = ia if atoms_meta[ia].is_heavy else ib
            hyd_i = ib if heavy_i == ia else ia
            vec = coords_arr[hyd_i] - coords_arr[heavy_i]
            nrm = max(float(np.linalg.norm(vec)), 1e-12)
            coords_arr[hyd_i] = coords_arr[heavy_i] + req * (vec / nrm)
            if not (zone_full[heavy_i] == ZONE_SCAFFOLD and zone_full[hyd_i] == ZONE_SCAFFOLD):
                c_i.append(heavy_i)
                c_j.append(hyd_i)
                c_d2.append(req * req)
        else:
            if not (zone_full[ia] == ZONE_SCAFFOLD and zone_full[ib] == ZONE_SCAFFOLD):
                b_i.append(ia)
                b_j.append(ib)
                b_kr.append(kr)
                b_req.append(req)

    # Inner-sphere dative coordination restraints for retained structural Zn2+ protein donors
    # (Prevents non-polarizable TIP3P water displacement of neutral His ligands while leaving 4th site open;
    #  explicitly labeled non-ZAFF harmonic coordination restraint in warnings.)
    if zinc_coord_mode == "harmonic_restraint":
        n_zn_coord_added = 0
        for zs in zinc_sites:
            zi = int(zs["zn_atom_index"])
            res_counts: dict[str, int] = {}
            for d in zs["donors"]:
                if float(d["init_dist"]) <= 2.55:
                    res_counts[d["res"]] = res_counts.get(d["res"], 0) + 1
            for d in zs["donors"]:
                if float(d["init_dist"]) > 2.55:
                    continue
                di = int(d["donor_atom_index"])
                an = d["atom"]
                if an in ("ND1", "NE2"):
                    req_zn = 2.05
                elif an == "SG":
                    req_zn = 2.30
                else:
                    # Bidentate carboxylate (e.g. Glu72 in CPA1) vs monodentate Asp/Glu
                    req_zn = 2.22 if res_counts.get(d["res"], 1) >= 2 else 2.08
                b_i.append(min(zi, di))
                b_j.append(max(zi, di))
                b_kr.append(85.0)
                b_req.append(req_zn)
                n_zn_coord_added += 1
        # Also preserve mobile Ca2+/Mg2+ EF-hand / structural oxygen coordination shells if present
        div_ions = [am.index for am in atoms_meta if am.atype in ("CA2+", "MG2+") and zone_full[am.index] != ZONE_SCAFFOLD]
        n_div_coord_added = 0
        for gi in div_ions:
            rcut_div = 2.75 if atoms_meta[gi].atype == "CA2+" else 2.45
            req_div = 2.40 if atoms_meta[gi].atype == "CA2+" else 2.10
            ipos = coords_arr[gi]
            for am in atoms_meta:
                if am.role == ROLE_RECEPTOR and am.atom_name in ("O", "OD1", "OD2", "OE1", "OE2"):
                    if float(np.linalg.norm(coords_arr[am.index] - ipos)) <= rcut_div:
                        b_i.append(min(gi, am.index))
                        b_j.append(max(gi, am.index))
                        b_kr.append(65.0)
                        b_req.append(req_div)
                        n_div_coord_added += 1
        if n_zn_coord_added > 0:
            warnings.append(
                f"Applied crystallographic inner-sphere harmonic Zn2+ coordination restraints "
                f"(k=85.0 kcal/mol/A^2) on {n_zn_coord_added} protein donor bonds (fixed-charge non-ZAFF model)."
            )
        if n_div_coord_added > 0:
            warnings.append(
                f"Applied crystallographic inner-sphere harmonic Ca2+/Mg2+ coordination restraints "
                f"(k=65.0 kcal/mol/A^2) on {n_div_coord_added} protein oxygen donor bonds."
            )

    # Angles (skip internal H-O-H of rigid SETTLE waters; record all 1-2/1-3 topological exclusions)
    pairs_12_13: set[tuple[int, int]] = set()
    for ia, ib in bonds_raw:
        pairs_12_13.add((min(ia, ib), max(ia, ib)))

    a_i, a_j, a_k, a_kt, a_eq = [], [], [], [], []
    for center in range(n_total):
        if center in wat_atom_set:
            continue
        nbs = adj[center]
        if len(nbs) < 2:
            continue
        tc = atoms_meta[center].atype
        for u_idx in range(len(nbs)):
            for v_idx in range(u_idx + 1, len(nbs)):
                ia, ic = nbs[u_idx], nbs[v_idx]
                pairs_12_13.add((min(ia, ic), max(ia, ic)))
                if (
                    zone_full[ia] == ZONE_SCAFFOLD
                    and zone_full[center] == ZONE_SCAFFOLD
                    and zone_full[ic] == ZONE_SCAFFOLD
                ):
                    continue
                v1 = coords_arr[ia] - coords_arr[center]
                v2 = coords_arr[ic] - coords_arr[center]
                cos_th = float(np.dot(v1, v2) / max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-12))
                ideal_rad = math.acos(min(1.0, max(-1.0, cos_th)))
                kt, teq = lookup_angle_param(atoms_meta[ia].atype, tc, atoms_meta[ic].atype, ideal_rad)
                a_i.append(ia)
                a_j.append(center)
                a_k.append(ic)
                a_kt.append(kt)
                a_eq.append(teq)

    # Proper Dihedrals (ff99SB phi/psi + ILDN chi1/chi2 + generic parm99) & 1-4 Pairs
    d_i, d_j, d_k, d_l, d_pk, d_n, d_ph, d_pep = [], [], [], [], [], [], [], []
    pairs_14_set: set[tuple[int, int]] = set()
    for jb, kb in bonds_raw:
        if jb in wat_atom_set:
            continue
        # Enumerate all i - jb - kb - l paths
        for ia in adj[jb]:
            if ia == kb:
                continue
            for il in adj[kb]:
                if il == jb or il == ia:
                    continue
                p14_key = (min(ia, il), max(ia, il))
                if p14_key not in pairs_12_13:
                    pairs_14_set.add(p14_key)
                if (
                    zone_full[ia] == ZONE_SCAFFOLD
                    and zone_full[jb] == ZONE_SCAFFOLD
                    and zone_full[kb] == ZONE_SCAFFOLD
                    and zone_full[il] == ZONE_SCAFFOLD
                ):
                    continue

                am_i, am_j, am_k, am_l = atoms_meta[ia], atoms_meta[jb], atoms_meta[kb], atoms_meta[il]
                names4 = (am_i.atom_name, am_j.atom_name, am_k.atom_name, am_l.atom_name)
                names4_rev = (am_l.atom_name, am_k.atom_name, am_j.atom_name, am_i.atom_name)

                # Check ff99SB backbone phi (C - N - CA - C) & psi (N - CA - C - N)
                terms: list[tuple[float, float, float]] | None = None
                if names4 == ("C", "N", "CA", "C") or names4_rev == ("C", "N", "CA", "C"):
                    terms = FF99SB_PHI_TERMS
                elif names4 == ("N", "CA", "C", "N") or names4_rev == ("N", "CA", "C", "N"):
                    terms = FF99SB_PSI_TERMS
                elif am_j.res_seq == am_k.res_seq and am_j.chain_id == am_k.chain_id:
                    rbase = am_j.res_name
                    if (rbase, *names4) in ILDN_SIDECHAIN_OVERRIDES:
                        terms = ILDN_SIDECHAIN_OVERRIDES[(rbase, *names4)]
                    elif (rbase, *names4_rev) in ILDN_SIDECHAIN_OVERRIDES:
                        terms = ILDN_SIDECHAIN_OVERRIDES[(rbase, *names4_rev)]

                if terms is None:
                    terms = generic_dihedral_terms(am_i.atype, am_j.atype, am_k.atype, am_l.atype)

                # Peptide rotatable torsion tag (scale in IHRT higher replicas; do NOT soften planar peptide omega!)
                is_pep_rot = (
                    1
                    if (am_j.role == ROLE_PEPTIDE and am_k.role == ROLE_PEPTIDE and {am_j.atype, am_k.atype} != {"C", "N"})
                    else 0
                )

                for pk_v, per_n, ph_deg in terms:
                    if abs(pk_v) < 1e-8:
                        continue
                    d_i.append(ia)
                    d_j.append(jb)
                    d_k.append(kb)
                    d_l.append(il)
                    d_pk.append(pk_v)
                    d_n.append(per_n)
                    d_ph.append(ph_deg * DEG2RAD)
                    d_pep.append(is_pep_rot)

    # Planar sp2 Improper Torsions (keep peptide bonds, carboxylates, guanidinium, and aromatic rings crisp!)
    sp2_planar_types = {"C", "CA", "CB", "CC", "CN", "CR", "CV", "CW", "C*", "N", "N2", "NA"}
    for center in range(n_solute):
        at_c = atoms_meta[center].atype
        nbs = adj[center]
        if at_c in sp2_planar_types and len(nbs) == 3:
            if (
                zone_full[center] == ZONE_SCAFFOLD
                and zone_full[nbs[0]] == ZONE_SCAFFOLD
                and zone_full[nbs[1]] == ZONE_SCAFFOLD
                and zone_full[nbs[2]] == ZONE_SCAFFOLD
            ):
                continue
            k_imp = 10.5 if at_c == "C" else 1.1
            d_i.append(nbs[0])
            d_j.append(nbs[1])
            d_k.append(center)
            d_l.append(nbs[2])
            d_pk.append(k_imp)
            d_n.append(2.0)
            d_ph.append(180.0 * DEG2RAD)
            d_pep.append(0)  # Never weaken sp2 planarity in IHRT!

    # 1-4 Scaled Nonbonded Pairs (AMBER SCEE=1.2, SCNB=2.0)
    p14_i, p14_j, p14_qp, p14_rm, p14_ep = [], [], [], [], []
    for ia, il in sorted(pairs_14_set):
        if zone_full[ia] == ZONE_SCAFFOLD and zone_full[il] == ZONE_SCAFFOLD:
            continue
        qp = (COULOMB_KCAL_A * charges_list[ia] * charges_list[il]) / 1.2
        rm = rmin_list[ia] + rmin_list[il]
        ep = math.sqrt(eps_list[ia] * eps_list[il]) / 2.0
        p14_i.append(ia)
        p14_j.append(il)
        p14_qp.append(qp)
        p14_rm.append(rm)
        p14_ep.append(ep)

    # CSR Exclusions (all 1-2, 1-3, and 1-4 pairs stored with i < j)
    all_excl = pairs_12_13 | pairs_14_set
    # Also exclude internal rigid water H1-H2 pairs!
    for wo, wh1, wh2 in wat_triplets:
        all_excl.add((min(wo, wh1), max(wo, wh1)))
        all_excl.add((min(wo, wh2), max(wo, wh2)))
        all_excl.add((min(wh1, wh2), max(wh1, wh2)))

    excl_per_atom: list[list[int]] = [[] for _ in range(n_total)]
    for ia, ib in all_excl:
        excl_per_atom[min(ia, ib)].append(max(ia, ib))
    excl_offsets = [0]
    excl_targets: list[int] = []
    for i in range(n_total):
        row = sorted(set(excl_per_atom[i]))
        excl_targets.extend(row)
        excl_offsets.append(len(excl_targets))

    # FC-BAOAB Covalent-Centroid Filter Groups (every active heavy atom with 1..4 bonded hydrogens)
    fg_heavy, fg_nh, fg_h_flat = [], [], []
    for i in range(n_total):
        if zone_full[i] == ZONE_SCAFFOLD or not atoms_meta[i].is_heavy:
            continue
        h_nbs = [nb for nb in adj[i] if not atoms_meta[nb].is_heavy]
        if 1 <= len(h_nbs) <= 4:
            fg_heavy.append(i)
            fg_nh.append(len(h_nbs))
            padded = h_nbs + [0] * (4 - len(h_nbs))
            fg_h_flat.extend(padded)

    cap_r = float(shell_waters_cutoff + 1.5)

    return BuiltSystem(
        n_atoms=n_total,
        solvent_mode=solvent_mode,
        coords=np.ascontiguousarray(coords_arr, dtype=np.float64),
        atoms=atoms_meta,
        mass=np.ascontiguousarray(mass_list, dtype=np.float64),
        charge=np.ascontiguousarray(charges_list, dtype=np.float64),
        rmin_half=np.ascontiguousarray(rmin_list, dtype=np.float64),
        epsilon=np.ascontiguousarray(eps_list, dtype=np.float64),
        gb_radius=np.ascontiguousarray(gbr_list, dtype=np.float64),
        gb_screen=np.ascontiguousarray(gbs_list, dtype=np.float64),
        zone=np.ascontiguousarray(zone_full, dtype=np.int32),
        role=np.ascontiguousarray(role_list, dtype=np.int32),
        tether_k=np.ascontiguousarray(tether_full, dtype=np.float64),
        bond_i=np.ascontiguousarray(b_i, dtype=np.int32),
        bond_j=np.ascontiguousarray(b_j, dtype=np.int32),
        bond_kr=np.ascontiguousarray(b_kr, dtype=np.float64),
        bond_req=np.ascontiguousarray(b_req, dtype=np.float64),
        ang_i=np.ascontiguousarray(a_i, dtype=np.int32),
        ang_j=np.ascontiguousarray(a_j, dtype=np.int32),
        ang_k=np.ascontiguousarray(a_k, dtype=np.int32),
        ang_kt=np.ascontiguousarray(a_kt, dtype=np.float64),
        ang_eq=np.ascontiguousarray(a_eq, dtype=np.float64),
        dih_i=np.ascontiguousarray(d_i, dtype=np.int32),
        dih_j=np.ascontiguousarray(d_j, dtype=np.int32),
        dih_k=np.ascontiguousarray(d_k, dtype=np.int32),
        dih_l=np.ascontiguousarray(d_l, dtype=np.int32),
        dih_pk=np.ascontiguousarray(d_pk, dtype=np.float64),
        dih_n=np.ascontiguousarray(d_n, dtype=np.float64),
        dih_phase=np.ascontiguousarray(d_ph, dtype=np.float64),
        dih_is_pep=np.ascontiguousarray(d_pep, dtype=np.int32),
        p14_i=np.ascontiguousarray(p14_i, dtype=np.int32),
        p14_j=np.ascontiguousarray(p14_j, dtype=np.int32),
        p14_qprod=np.ascontiguousarray(p14_qp, dtype=np.float64),
        p14_rmin=np.ascontiguousarray(p14_rm, dtype=np.float64),
        p14_eps=np.ascontiguousarray(p14_ep, dtype=np.float64),
        excl_offsets=np.ascontiguousarray(excl_offsets, dtype=np.int32),
        excl_targets=np.ascontiguousarray(excl_targets, dtype=np.int32),
        con_i=np.ascontiguousarray(c_i, dtype=np.int32),
        con_j=np.ascontiguousarray(c_j, dtype=np.int32),
        con_d2=np.ascontiguousarray(c_d2, dtype=np.float64),
        wat_o=np.ascontiguousarray([w[0] for w in wat_triplets], dtype=np.int32),
        wat_h1=np.ascontiguousarray([w[1] for w in wat_triplets], dtype=np.int32),
        wat_h2=np.ascontiguousarray([w[2] for w in wat_triplets], dtype=np.int32),
        fg_heavy=np.ascontiguousarray(fg_heavy, dtype=np.int32),
        fg_nh=np.ascontiguousarray(fg_nh, dtype=np.int32),
        fg_h_flat=np.ascontiguousarray(fg_h_flat, dtype=np.int32),
        cap_anchors=np.ascontiguousarray(cap_anchor_list, dtype=np.int32),
        shell_water_oxygens=np.ascontiguousarray(shell_wat_oxygens, dtype=np.int32),
        cap_radius=cap_r,
        box=box_dims,
        pme_grid=pme_grid,
        ionic_strength=ionic_strength,
        res_diel=np.ascontiguousarray(res_diel_list, dtype=np.float64),
        zinc_sites=zinc_sites,
        warnings=warnings,
    )


def build_protein_system(
    pdb_text: str,
    chains: list[str] | None = None,
    solvent_mode: str = "explicit-pme",
    metal_handling: str = "strip",
    zinc_coord_mode: str = "harmonic_restraint",
    allow_strip_hetatm: bool = False,
    allow_chain_breaks: bool = False,
    reconstruct_missing_heavy: bool = False,
    allow_ptm_and_caps: bool = True,
    box_padding: float = 8.5,
    shell_waters_cutoff: float = 4.8,
    max_shell_waters: int = 80,
    active_radius: float = 10.0,
    buffer_radius: float | None = None,
    ionic_strength: float = 0.15,
    seed: int = 42,
) -> BuiltSystem:
    if solvent_mode not in ("explicit-pme", "active-shell"):
        raise TopologyBuildError(f"Unsupported solvent_mode: {solvent_mode!r}")
    if zinc_coord_mode not in ("harmonic_restraint", "nonbonded_only"):
        raise TopologyBuildError(f"Unsupported zinc_coord_mode: {zinc_coord_mode!r}")
    sel = set(chains) if chains else None
    parsed_chains, ions, warnings = _parse_pdb_residues(
        pdb_text, sel, metal_handling=metal_handling,
        allow_strip_hetatm=allow_strip_hetatm,
        allow_ptm_and_caps=allow_ptm_and_caps,
    )
    eff_buf = buffer_radius
    if eff_buf is None and solvent_mode == "active-shell" and any(ion["res_name"] == "ZN" for ion in ions):
        eff_buf = 16.0
    elif eff_buf is not None and eff_buf < 0:
        eff_buf = None
    return _build_from_parsed_chains(
        parsed_chains,
        ions,
        peptide_chains=set(),
        solvent_mode=solvent_mode,
        allow_chain_breaks=allow_chain_breaks,
        box_padding=box_padding,
        shell_waters_cutoff=shell_waters_cutoff,
        max_shell_waters=max_shell_waters,
        active_radius=active_radius,
        buffer_radius=eff_buf,
        ionic_strength=ionic_strength,
        seed=seed,
        warnings=warnings,
        zinc_coord_mode=zinc_coord_mode,
        reconstruct_missing_heavy=reconstruct_missing_heavy,
    )


def build_complex_system(
    pdb_text: str,
    receptor_chains: list[str],
    peptide_chains: list[str],
    solvent_mode: str = "active-shell",
    metal_handling: str = "retain",
    zinc_coord_mode: str = "harmonic_restraint",
    allow_strip_hetatm: bool = True,
    allow_chain_breaks: bool = True,
    reconstruct_missing_heavy: bool = True,
    allow_ptm_and_caps: bool = True,
    box_padding: float = 8.5,
    shell_waters_cutoff: float = 4.8,
    max_shell_waters: int = 96,
    active_radius: float = 12.5,
    buffer_radius: float | None = 20.0,
    ionic_strength: float = 0.15,
    seed: int = 42,
) -> BuiltSystem:
    if not receptor_chains or not peptide_chains:
        raise TopologyBuildError("Both receptor_chains and peptide_chains must be non-empty.")
    if zinc_coord_mode not in ("harmonic_restraint", "nonbonded_only"):
        raise TopologyBuildError(f"Unsupported zinc_coord_mode: {zinc_coord_mode!r}")
    sel = set(receptor_chains) | set(peptide_chains)
    parsed_chains, ions, warnings = _parse_pdb_residues(
        pdb_text, sel, metal_handling=metal_handling,
        allow_strip_hetatm=allow_strip_hetatm,
        allow_ptm_and_caps=allow_ptm_and_caps,
    )
    for rc in receptor_chains:
        if rc not in parsed_chains:
            raise TopologyBuildError(f"Requested receptor chain {rc!r} not found in PDB.")
    for pc in peptide_chains:
        if pc not in parsed_chains:
            raise TopologyBuildError(f"Requested peptide chain {pc!r} not found in PDB.")
    return _build_from_parsed_chains(
        parsed_chains,
        ions,
        peptide_chains=set(peptide_chains),
        solvent_mode=solvent_mode,
        allow_chain_breaks=allow_chain_breaks,
        box_padding=box_padding,
        shell_waters_cutoff=shell_waters_cutoff,
        max_shell_waters=max_shell_waters,
        active_radius=active_radius,
        buffer_radius=buffer_radius,
        ionic_strength=ionic_strength,
        seed=seed,
        warnings=warnings,
        zinc_coord_mode=zinc_coord_mode,
        reconstruct_missing_heavy=reconstruct_missing_heavy,
    )
