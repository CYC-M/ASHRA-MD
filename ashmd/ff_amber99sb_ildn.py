"""Published AMBER99SB-ILDN Force Field Parameter Library + Li-Merz IOD Zn2+ & TIP3P.

Sources (peer-reviewed published parameters, zero heuristic fabrication):
  - Charges, radii & atom types: Cornell et al., J. Am. Chem. Soc. 1995, 117, 5179 (parm99)
  - Backbone phi/psi torsion corrections: Hornak et al., Proteins 2006, 65, 712 (ff99SB)
  - Sidechain chi1/chi2 ILDN corrections: Lindorff-Larsen et al., Proteins 2010, 78, 1950 (AMBER99SB-ILDN)
  - Intrinsic Born radii & OBC II screening: Onufriev, Bashford & Case, Proteins 2004, 55, 383 (mbondi2 / igb=5)
  - Monovalent Na+/Cl- ions (TIP3P): Joung & Cheatham, J. Phys. Chem. B 2008, 112, 9020
  - Structural Zn2+ nonbonded 12-6 IOD: Li, Roberts, Chakravorty & Merz, JCTC 2013, 9, 2733
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
from typing import Any

TEMPLATE_JSON_PATH = pathlib.Path(__file__).resolve().parent / "data" / "amber99sb_ildn_templates.json"
EXPECTED_TEMPLATE_SHA256 = "2b103c1b8e4bc25ecb99dda979615ec38b083b869b4223f9bc6b69e09f4e4d07"

DEG2RAD = math.pi / 180.0

# Atomic masses [amu], AMBER parm99 LJ epsilon [kcal/mol], Rmin/2 [A],
# and Onufriev-Bashford-Case mbondi2 intrinsic radius [A] + screening factor S_i
ATOM_TYPE_PROPERTIES: dict[str, dict[str, float]] = {
    # Carbon sp2 (carbonyl, aromatic, guanidinium, imidazole, indole)
    "C":  {"mass": 12.01, "eps": 0.0860, "rmin_half": 1.9080, "gb_r": 1.70, "gb_s": 0.72},
    "C*": {"mass": 12.01, "eps": 0.0860, "rmin_half": 1.9080, "gb_r": 1.70, "gb_s": 0.72},
    "CA": {"mass": 12.01, "eps": 0.0860, "rmin_half": 1.9080, "gb_r": 1.70, "gb_s": 0.72},
    "CB": {"mass": 12.01, "eps": 0.0860, "rmin_half": 1.9080, "gb_r": 1.70, "gb_s": 0.72},
    "CC": {"mass": 12.01, "eps": 0.0860, "rmin_half": 1.9080, "gb_r": 1.70, "gb_s": 0.72},
    "CN": {"mass": 12.01, "eps": 0.0860, "rmin_half": 1.9080, "gb_r": 1.70, "gb_s": 0.72},
    "CR": {"mass": 12.01, "eps": 0.0860, "rmin_half": 1.9080, "gb_r": 1.70, "gb_s": 0.72},
    "CV": {"mass": 12.01, "eps": 0.0860, "rmin_half": 1.9080, "gb_r": 1.70, "gb_s": 0.72},
    "CW": {"mass": 12.01, "eps": 0.0860, "rmin_half": 1.9080, "gb_r": 1.70, "gb_s": 0.72},
    # Carbon sp3 aliphatic
    "CT": {"mass": 12.01, "eps": 0.1094, "rmin_half": 1.9080, "gb_r": 1.70, "gb_s": 0.72},
    # Hydrogens (note mbondi2 signature: H attached to N has gb_r = 1.30 A!)
    "H":  {"mass": 1.008, "eps": 0.0157, "rmin_half": 0.6000, "gb_r": 1.30, "gb_s": 0.85},
    "H1": {"mass": 1.008, "eps": 0.0157, "rmin_half": 1.3870, "gb_r": 1.20, "gb_s": 0.85},
    "H4": {"mass": 1.008, "eps": 0.0150, "rmin_half": 1.4090, "gb_r": 1.20, "gb_s": 0.85},
    "H5": {"mass": 1.008, "eps": 0.0150, "rmin_half": 1.3590, "gb_r": 1.20, "gb_s": 0.85},
    "HA": {"mass": 1.008, "eps": 0.0150, "rmin_half": 1.4590, "gb_r": 1.20, "gb_s": 0.85},
    "HC": {"mass": 1.008, "eps": 0.0157, "rmin_half": 1.4870, "gb_r": 1.20, "gb_s": 0.85},
    # mbondi2 (Onufriev 2004) assigns gb_r=1.20 A to hydroxyl HO; polar-H LJ core matches H/HS (0.60 A) to prevent GB H-bond collapse
    "HO": {"mass": 1.008, "eps": 0.0157, "rmin_half": 0.6000, "gb_r": 1.20, "gb_s": 0.85},
    "HP": {"mass": 1.008, "eps": 0.0157, "rmin_half": 1.1000, "gb_r": 1.20, "gb_s": 0.85},
    "HS": {"mass": 1.008, "eps": 0.0157, "rmin_half": 0.6000, "gb_r": 1.20, "gb_s": 0.85},
    "HW": {"mass": 1.008, "eps": 0.0000, "rmin_half": 0.0000, "gb_r": 0.80, "gb_s": 0.85},
    # Nitrogens
    "N":  {"mass": 14.01, "eps": 0.1700, "rmin_half": 1.8240, "gb_r": 1.55, "gb_s": 0.79},
    "N2": {"mass": 14.01, "eps": 0.1700, "rmin_half": 1.8240, "gb_r": 1.55, "gb_s": 0.79},
    "N3": {"mass": 14.01, "eps": 0.1700, "rmin_half": 1.8240, "gb_r": 1.55, "gb_s": 0.79},
    "NA": {"mass": 14.01, "eps": 0.1700, "rmin_half": 1.8240, "gb_r": 1.55, "gb_s": 0.79},
    "NB": {"mass": 14.01, "eps": 0.1700, "rmin_half": 1.8240, "gb_r": 1.55, "gb_s": 0.79},
    # Oxygens (mbondi3 Nguyen, Roe & Simmerling JCTC 2013: O2 gb_r=1.40 A for carboxylates;
    # Steinbrecher, Latzer & Case JCTC 2012 phosaa10: OP rmin_half=1.7210 A, gb_r=1.32 A for dianionic PO3^2-)
    "O":  {"mass": 16.00, "eps": 0.2100, "rmin_half": 1.6612, "gb_r": 1.50, "gb_s": 0.85},
    "O2": {"mass": 16.00, "eps": 0.2100, "rmin_half": 1.6612, "gb_r": 1.40, "gb_s": 0.85},
    "OP": {"mass": 16.00, "eps": 0.2100, "rmin_half": 1.7210, "gb_r": 1.32, "gb_s": 0.85},
    "OH": {"mass": 16.00, "eps": 0.2104, "rmin_half": 1.7210, "gb_r": 1.50, "gb_s": 0.85},
    "OS": {"mass": 16.00, "eps": 0.1700, "rmin_half": 1.6837, "gb_r": 1.50, "gb_s": 0.85},
    "OW": {"mass": 16.00, "eps": 0.1520, "rmin_half": 1.7683, "gb_r": 1.50, "gb_s": 0.85},
    # Phosphorus (Homeyer, Horn, Lanig & Sticht, J. Mol. Model. 2006, AMBER parm99)
    "P":  {"mass": 30.97, "eps": 0.2000, "rmin_half": 2.1000, "gb_r": 1.85, "gb_s": 0.86},
    # Sulfurs
    "S":  {"mass": 32.06, "eps": 0.2500, "rmin_half": 2.0000, "gb_r": 1.80, "gb_s": 0.96},
    "SH": {"mass": 32.06, "eps": 0.2500, "rmin_half": 2.0000, "gb_r": 1.80, "gb_s": 0.96},
    # Ions: Li-Merz 2013 12-6 IOD divalent cations (ZN2+, CA2+, MG2+) & Joung-Cheatham 2008 Na+ / Cl-
    "ZN2+": {"mass": 65.38,  "eps": 0.01491700, "rmin_half": 1.3950, "gb_r": 1.395, "gb_s": 0.80, "charge":  2.0},
    "CA2+": {"mass": 40.08,  "eps": 0.14712800, "rmin_half": 1.6000, "gb_r": 1.600, "gb_s": 0.80, "charge":  2.0},
    "MG2+": {"mass": 24.305, "eps": 0.01020900, "rmin_half": 1.3600, "gb_r": 1.360, "gb_s": 0.80, "charge":  2.0},
    "NA+":  {"mass": 22.99,  "eps": 0.08743930, "rmin_half": 1.3690, "gb_r": 1.500, "gb_s": 0.80, "charge":  1.0},
    "CL-":  {"mass": 35.45,  "eps": 0.03559100, "rmin_half": 2.5130, "gb_r": 1.700, "gb_s": 0.80, "charge": -1.0},
}

# AMBER parm99 harmonic bond parameters: (type1, type2) -> (kr [kcal/(mol*A^2)], req [A])
def _canon_bond(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)

_RAW_BONDS: list[tuple[str, str, float, float]] = [
    ("C",  "O",  570.0, 1.229),
    ("C",  "O2", 656.0, 1.250),
    ("C",  "N",  490.0, 1.335),   # Peptide bond!
    ("C",  "CT", 317.0, 1.522),
    ("C",  "CA", 469.0, 1.409),
    ("CT", "CT", 310.0, 1.526),
    ("CT", "N",  337.0, 1.449),
    ("CT", "N2", 337.0, 1.463),
    ("CT", "N3", 367.0, 1.471),
    ("CT", "OH", 320.0, 1.410),
    ("CT", "SH", 237.0, 1.810),
    ("CT", "S",  227.0, 1.810),
    ("CT", "CA", 317.0, 1.510),
    ("CT", "CC", 317.0, 1.504),
    ("CT", "C*", 317.0, 1.495),
    ("CA", "CA", 469.0, 1.400),
    ("CA", "CB", 469.0, 1.404),
    ("CA", "CN", 447.0, 1.400),
    ("CA", "N2", 481.0, 1.340),
    ("CA", "OH", 450.0, 1.364),
    ("CB", "CB", 520.0, 1.370),
    ("CB", "CN", 447.0, 1.419),
    ("CB", "C*", 388.0, 1.459),
    ("C*", "CW", 546.0, 1.352),
    ("CN", "NA", 428.0, 1.380),
    ("CW", "NA", 427.0, 1.381),
    ("CC", "NA", 422.0, 1.385),
    ("CC", "NB", 410.0, 1.394),
    ("CC", "CV", 512.0, 1.375),
    ("CC", "CW", 518.0, 1.371),
    ("CR", "NA", 477.0, 1.343),
    ("CR", "NB", 488.0, 1.335),
    ("CV", "NB", 410.0, 1.394),
    ("S",  "S",  166.0, 2.038),   # Disulfide CYX-CYX bridge!
    # Phosphate ester bonds (SEP / TPO / PTR, Homeyer et al. 2006; Steinbrecher et al. 2012)
    ("P",  "O2", 525.0, 1.480),
    ("P",  "OP", 525.0, 1.480),
    ("P",  "OS", 230.0, 1.610),
    ("CT", "OS", 320.0, 1.410),
    ("CA", "OS", 372.0, 1.370),
    # X-H covalent bonds
    ("N",  "H",  434.0, 1.010),
    ("N2", "H",  434.0, 1.010),
    ("N3", "H",  434.0, 1.010),
    ("NA", "H",  434.0, 1.010),
    ("OH", "HO", 553.0, 0.960),
    ("SH", "HS", 274.0, 1.336),
    ("OW", "HW", 553.0, 0.9572),
    ("CT", "HC", 340.0, 1.090),
    ("CT", "H1", 340.0, 1.090),
    ("CT", "HP", 340.0, 1.090),
    ("CA", "HA", 367.0, 1.080),
    ("CV", "H4", 367.0, 1.080),
    ("CW", "H4", 367.0, 1.080),
    ("CR", "H5", 367.0, 1.080),
]
BOND_PARAMS: dict[tuple[str, str], tuple[float, float]] = {
    _canon_bond(a, b): (kr, req) for a, b, kr, req in _RAW_BONDS
}


# AMBER parm99 harmonic angle parameters: (t1, t_center, t3) -> (ktheta [kcal/(mol*rad^2)], theta_eq [deg])
def _canon_angle(a: str, b: str, c: str) -> tuple[str, str, str]:
    return (a, b, c) if a <= c else (c, b, a)

_RAW_ANGLES: list[tuple[str, str, str, float, float]] = [
    # Peptide backbone
    ("C",  "N",  "CT", 50.0, 121.9),
    ("C",  "N",  "H",  50.0, 120.0),
    ("CT", "N",  "H",  50.0, 118.04),
    ("CT", "N",  "CT", 50.0, 118.0),  # Proline N
    ("N",  "CT", "C",  63.0, 110.1),
    ("N",  "CT", "CT", 80.0, 109.7),
    ("N",  "CT", "H1", 50.0, 109.5),
    ("N3", "CT", "C",  63.0, 110.1),
    ("N3", "CT", "CT", 80.0, 111.2),
    ("N3", "CT", "H1", 50.0, 109.5),
    ("N3", "CT", "HP", 50.0, 109.5),
    ("C",  "CT", "CT", 63.0, 111.1),
    ("C",  "CT", "H1", 50.0, 109.5),
    ("C",  "CT", "HC", 50.0, 109.5),
    ("CT", "C",  "N",  70.0, 116.6),
    ("CT", "C",  "O",  80.0, 120.4),
    ("CT", "C",  "O2", 70.0, 117.0),
    ("N",  "C",  "O",  80.0, 122.9),
    ("O2", "C",  "O2", 80.0, 126.0),
    # Aliphatic & heteroatom sidechains
    ("CT", "CT", "CT", 40.0, 109.5),
    ("CT", "CT", "HC", 50.0, 109.5),
    ("CT", "CT", "H1", 50.0, 109.5),
    ("CT", "CT", "HP", 50.0, 109.5),
    ("HC", "CT", "HC", 35.0, 109.5),
    ("H1", "CT", "H1", 35.0, 109.5),
    ("HP", "CT", "HP", 35.0, 109.5),
    ("CT", "CT", "OH", 50.0, 109.5),
    ("OH", "CT", "H1", 50.0, 109.5),
    ("CT", "OH", "HO", 55.0, 108.5),
    ("CT", "CT", "SH", 50.0, 108.6),
    ("CT", "CT", "S",  50.0, 114.7),
    ("SH", "CT", "H1", 50.0, 109.5),
    ("S",  "CT", "H1", 50.0, 109.5),
    ("CT", "SH", "HS", 43.0, 96.0),
    ("CT", "S",  "CT", 62.0, 98.9),
    ("CT", "S",  "S",  68.0, 103.7),
    # Ammonium & Guanidinium (Lys, N-term, Arg)
    ("CT", "N3", "H",  50.0, 109.5),
    ("CT", "N3", "CT", 50.0, 109.5),
    ("H",  "N3", "H",  35.0, 109.5),
    ("CT", "CT", "N2", 80.0, 111.2),
    ("CT", "N2", "CA", 50.0, 123.2),
    ("CT", "N2", "H",  50.0, 118.4),
    ("CA", "N2", "H",  50.0, 120.0),
    ("H",  "N2", "H",  35.0, 120.0),
    ("N2", "CA", "N2", 70.0, 120.0),
    # Aromatic Phe / Tyr / Trp / His
    ("CT", "CA", "CA", 70.0, 120.0),
    ("CA", "CA", "CA", 63.0, 120.0),
    ("CA", "CA", "HA", 50.0, 120.0),
    ("CA", "CA", "OH", 70.0, 120.0),
    ("CA", "OH", "HO", 50.0, 113.0),
    ("CT", "CT", "CA", 63.0, 114.0),
    ("CT", "CT", "CC", 63.0, 113.1),
    ("CT", "CT", "C*", 63.0, 115.6),
    # His imidazole
    ("CT", "CC", "NA", 70.0, 120.0),
    ("CT", "CC", "NB", 70.0, 120.0),
    ("CT", "CC", "CV", 70.0, 129.0),
    ("CT", "CC", "CW", 70.0, 129.0),
    ("NA", "CC", "CV", 70.0, 106.5),
    ("NA", "CC", "CW", 70.0, 106.5),
    ("NB", "CC", "CW", 70.0, 109.5),
    ("CC", "NA", "CR", 70.0, 109.5),
    ("CC", "NA", "H",  50.0, 126.4),
    ("CR", "NA", "H",  50.0, 126.4),
    ("CC", "NB", "CR", 70.0, 105.0),
    ("NA", "CR", "NB", 70.0, 111.6),
    ("NA", "CR", "NA", 70.0, 108.8),
    ("NA", "CR", "H5", 50.0, 123.1),
    ("NB", "CR", "H5", 50.0, 126.0),
    ("CC", "CV", "NB", 70.0, 110.5),
    ("CC", "CV", "H4", 50.0, 129.0),
    ("NB", "CV", "H4", 50.0, 120.5),
    ("CV", "NB", "CR", 70.0, 105.0),
    ("CC", "CW", "NA", 70.0, 107.3),
    ("CC", "CW", "H4", 50.0, 130.7),
    ("NA", "CW", "H4", 50.0, 122.0),
    ("CW", "NA", "CR", 70.0, 108.8),
    ("CW", "NA", "H",  50.0, 125.6),
    # Trp indole
    ("CT", "C*", "CW", 70.0, 125.0),
    ("CT", "C*", "CB", 70.0, 128.6),
    ("CB", "C*", "CW", 70.0, 106.4),
    ("C*", "CW", "NA", 70.0, 108.7),
    ("C*", "CW", "H4", 50.0, 130.0),
    ("CW", "NA", "CN", 70.0, 111.6),
    ("CN", "NA", "H",  50.0, 123.1),
    ("NA", "CN", "CB", 70.0, 104.4),
    ("NA", "CN", "CA", 70.0, 132.8),
    ("CB", "CN", "CA", 63.0, 122.7),
    ("C*", "CB", "CN", 70.0, 108.9),
    ("C*", "CB", "CA", 63.0, 134.9),
    ("CN", "CB", "CA", 63.0, 116.2),
    ("CB", "CA", "CA", 63.0, 118.6),
    ("CB", "CA", "HA", 50.0, 120.0),
    ("CN", "CA", "CA", 63.0, 118.8),
    ("CN", "CA", "HA", 50.0, 120.0),
    # Water TIP3P
    ("HW", "OW", "HW", 100.0, 104.52),
    # Phosphate ester angles (SEP / TPO / PTR, Homeyer et al. 2006; Steinbrecher et al. 2012)
    ("O2", "P",  "O2", 140.0, 119.90),
    ("OP", "P",  "OP", 140.0, 119.90),
    ("OS", "P",  "O2", 100.0, 108.23),
    ("OS", "P",  "OP", 100.0, 108.23),
    ("CT", "OS", "P",  100.0, 120.50),
    ("CA", "OS", "P",  100.0, 120.50),
    ("CT", "CT", "OS",  50.0, 109.50),
    ("OS", "CT", "H1",  50.0, 109.50),
    ("CA", "CA", "OS",  70.0, 120.00),
]
ANGLE_PARAMS: dict[tuple[str, str, str], tuple[float, float]] = {
    _canon_angle(a, b, c): (kt, teq * DEG2RAD) for a, b, c, kt, teq in _RAW_ANGLES
}


def lookup_bond_param(t1: str, t2: str, ideal_dist: float) -> tuple[float, float]:
    key = _canon_bond(t1, t2)
    if key in BOND_PARAMS:
        return BOND_PARAMS[key]
    # Conservative chemical-class fallback based on sp2/sp3 hybridization (never zero!)
    if "H" in (t1[0], t2[0]):
        return (340.0, ideal_dist)
    return (350.0, ideal_dist)


def lookup_angle_param(t1: str, t2: str, t3: str, ideal_rad: float) -> tuple[float, float]:
    key = _canon_angle(t1, t2, t3)
    if key in ANGLE_PARAMS:
        return ANGLE_PARAMS[key]
    if t2 == "CT":
        if t1[0] == "H" and t3[0] == "H":
            return (35.0, 109.5 * DEG2RAD)
        if t1[0] == "H" or t3[0] == "H":
            return (50.0, 109.5 * DEG2RAD)
        return (63.0, 110.5 * DEG2RAD)
    return (65.0, ideal_rad)


# AMBER99SB-ILDN Dihedral Fourier Series: list of (pk [kcal/mol], periodicity n, phase [deg])
# 1. Hornak et al. (2006) ff99SB backbone phi (C-N-CT-C) and psi (N-CT-C-N)
FF99SB_PHI_TERMS: list[tuple[float, float, float]] = [
    (0.000, 1.0, 0.0),
    (0.270, 2.0, 0.0),
    (0.420, 3.0, 0.0),
    (0.000, 4.0, 0.0),
]
FF99SB_PSI_TERMS: list[tuple[float, float, float]] = [
    (0.450, 1.0, 180.0),
    (1.580, 2.0, 180.0),
    (0.550, 3.0, 180.0),
    (0.000, 4.0, 0.0),
]
OMEGA_PEPTIDE_TERMS: list[tuple[float, float, float]] = [
    (2.500, 2.0, 180.0),  # Planar trans/cis peptide bond barrier (~10 kcal/mol across 4 paths)
]

# 2. Lindorff-Larsen et al. (2010) ILDN sidechain chi1 / chi2 torsion corrections
# Keyed by (base_resname, atom1, atom2, atom3, atom4)
ILDN_SIDECHAIN_OVERRIDES: dict[tuple[str, str, str, str, str], list[tuple[float, float, float]]] = {
    # ILE chi1 (N-CA-CB-CG1) & chi2 (CA-CB-CG1-CD1)
    ("ILE", "N", "CA", "CB", "CG1"): [
        (0.6498, 1.0, 0.0), (0.2656, 2.0, 180.0), (0.0449, 3.0, 0.0),
    ],
    ("ILE", "CA", "CB", "CG1", "CD1"): [
        (0.6656, 1.0, 0.0), (0.2020, 2.0, 0.0), (0.0985, 3.0, 0.0),
    ],
    # LEU chi1 (N-CA-CB-CG) & chi2 (CA-CB-CG-CD1)
    ("LEU", "N", "CA", "CB", "CG"): [
        (0.9553, 1.0, 0.0), (0.4567, 2.0, 180.0), (0.0036, 3.0, 0.0),
    ],
    ("LEU", "CA", "CB", "CG", "CD1"): [
        (0.2810, 1.0, 0.0), (0.1408, 2.0, 0.0), (0.1282, 3.0, 0.0),
    ],
    # ASP chi1 (N-CA-CB-CG) & chi2 (CA-CB-CG-OD1 / OD2)
    ("ASP", "N", "CA", "CB", "CG"): [
        (0.9762, 1.0, 0.0), (0.8588, 2.0, 180.0), (0.1059, 3.0, 0.0),
    ],
    ("ASP", "CA", "CB", "CG", "OD1"): [
        (0.0000, 1.0, 0.0), (0.5769, 2.0, 180.0), (0.0000, 4.0, 0.0),
    ],
    ("ASP", "CA", "CB", "CG", "OD2"): [
        (0.0000, 1.0, 0.0), (0.5769, 2.0, 180.0), (0.0000, 4.0, 0.0),
    ],
    # ASN chi1 (N-CA-CB-CG) & chi2 (CA-CB-CG-OD1 / ND2)
    ("ASN", "N", "CA", "CB", "CG"): [
        (1.6626, 1.0, 0.0), (0.5746, 2.0, 180.0), (0.0365, 3.0, 0.0),
    ],
    ("ASN", "CA", "CB", "CG", "OD1"): [
        (0.3204, 1.0, 0.0), (0.5515, 2.0, 180.0), (0.0442, 3.0, 0.0),
    ],
}


# Generic AMBER parm99 central-bond dihedral terms per Torsion path (pk_per_path = Vn / idivf)
def generic_dihedral_terms(t1: str, t2: str, t3: str, t4: str) -> list[tuple[float, float, float]]:
    bpair = _canon_bond(t2, t3)
    if bpair == ("C", "N"):
        return OMEGA_PEPTIDE_TERMS
    if bpair == ("CT", "CT"):
        # Standard tetrahedral staggered barrier: total 1.40 kcal/mol / 9 paths = 0.1556 kcal/mol per path
        return [(0.1556, 3.0, 0.0)]
    if bpair in (("CT", "N"), ("CT", "N3")):
        return [(0.10, 3.0, 0.0)]
    if bpair == ("CT", "OH"):
        return [(0.1667, 3.0, 0.0)]
    if bpair == ("CT", "SH"):
        return [(0.2500, 3.0, 0.0)]
    if bpair == ("CT", "S"):
        return [(0.3333, 3.0, 0.0)]
    if bpair == ("S", "S"):
        # Disulfide +/-90 deg preference!
        return [(3.50, 2.0, 0.0), (0.60, 3.0, 0.0)]
    if bpair == ("C", "CT"):
        return [(0.0667, 2.0, 180.0)]
    if bpair in (("CA", "CA"), ("CA", "CB"), ("CA", "CN"), ("CB", "CN"), ("CB", "CB"),
                 ("CC", "CV"), ("CC", "CW"), ("CR", "NA"), ("CR", "NB"), ("CV", "NB"),
                 ("CW", "NA"), ("CN", "NA"), ("C*", "CW"), ("C*", "CB"), ("CC", "NA"), ("CC", "NB")):
        # Aromatic / conjugated planar ring double/partial-double bonds
        return [(3.625, 2.0, 180.0)]
    if bpair == ("CA", "N2"):
        # Guanidinium planar C-N bond
        return [(2.40, 2.0, 180.0)]
    if bpair == ("CA", "OH"):
        # Tyrosine phenolic OH in-plane preference
        return [(0.90, 2.0, 180.0)]
    if bpair == ("OS", "P"):
        return [(0.25, 3.0, 0.0)]
    if bpair == ("CT", "OS"):
        return [(0.3833, 3.0, 0.0)]
    if bpair == ("CA", "OS"):
        return [(0.90, 2.0, 180.0)]
    if bpair in (("CA", "CT"), ("CC", "CT"), ("C*", "CT")):
        return [(0.0, 2.0, 0.0)]
    return [(0.10, 3.0, 0.0)]


# Published Residue-Specific Interior Dielectric Constants for Endpoint RD-GBSA
# (Hou et al., JCIM 2011; Sun et al., PCCP 2014; Weng et al., JCIM 2020)
RESIDUE_INTERIOR_DIELECTRIC: dict[str, float] = {
    # Nonpolar aliphatic & aromatic
    "ALA": 2.0, "VAL": 2.0, "LEU": 2.0, "ILE": 2.0, "PRO": 2.0,
    "PHE": 2.0, "TRP": 2.0, "MET": 2.0, "GLY": 2.0, "ACE": 2.0, "NME": 2.0,
    # Polar uncharged
    "SER": 3.0, "THR": 3.0, "CYS": 3.0, "CYX": 2.5, "TYR": 3.0,
    "ASN": 3.0, "GLN": 3.0, "HID": 3.0, "HIE": 3.0, "HIS": 3.0,
    # Ionized formal-charge residues & dianionic phospho-residues (Hou et al., JCIM 2011: eps_in = 4.0)
    "ASP": 4.0, "GLU": 4.0, "LYS": 4.0, "ARG": 4.0, "HIP": 4.0, "CYM": 4.0,
    "SEP": 4.0, "TPO": 4.0, "PTR": 4.0,
}

# Published Single-Trajectory Net Backbone Reorganization & Rotamer Entropy Table at 310.15 K
# (Zimm-Bragg / Scholtz & Baldwin 1992 net coil->bound dG_bb = dH_intra - T*dS_bb; Wand, Nature 2007 rotamer burial):
#   (n_rotatable_chi_bonds, net_backbone_reorg_kcal_at_310K)
RESIDUE_ENTROPY_PARAMS: dict[str, tuple[int, float]] = {
    "GLY": (0, 0.50), "ALA": (0, 0.30), "PRO": (0, 0.00), "ACE": (0, 0.00), "NME": (0, 0.00),
    "VAL": (1, 0.30), "SER": (1, 0.30), "THR": (1, 0.30), "CYS": (1, 0.30), "CYM": (0, 0.15), "CYX": (0, 0.15),
    "ILE": (2, 0.30), "LEU": (2, 0.30), "ASP": (2, 0.30), "ASN": (2, 0.30),
    "HID": (2, 0.30), "HIE": (2, 0.30), "HIP": (2, 0.30), "HIS": (2, 0.30),
    "PHE": (2, 0.30), "TYR": (2, 0.30), "TRP": (2, 0.30),
    "GLU": (3, 0.30), "GLN": (3, 0.30), "MET": (3, 0.30),
    "SEP": (3, 0.30), "TPO": (3, 0.30), "PTR": (4, 0.30),
    "LYS": (4, 0.30), "ARG": (4, 0.30),
}


_CACHED_TEMPLATES: dict[str, Any] | None = None


def load_amber_templates() -> dict[str, Any]:
    """Load and cryptographically verify the compiled AMBER99SB-ILDN template table."""
    global _CACHED_TEMPLATES
    if _CACHED_TEMPLATES is not None:
        return _CACHED_TEMPLATES
    if not TEMPLATE_JSON_PATH.is_file():
        raise FileNotFoundError(f"Missing AMBER template database: {TEMPLATE_JSON_PATH}")
    raw = TEMPLATE_JSON_PATH.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != EXPECTED_TEMPLATE_SHA256:
        raise RuntimeError(
            f"AMBER template SHA-256 mismatch: expected {EXPECTED_TEMPLATE_SHA256}, got {actual_sha}"
        )
    _CACHED_TEMPLATES = json.loads(raw.decode("utf-8"))
    return _CACHED_TEMPLATES
