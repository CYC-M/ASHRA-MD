"""Zero-copy ctypes bridge to ASH-MD C++17 native engine (libashmd.dylib / libashmd.so)."""

from __future__ import annotations

import ctypes
import os
import pathlib
import platform
import subprocess
from typing import Any

import numpy as np

_CORE_DIR = pathlib.Path(__file__).resolve().parent.parent / "core"


class AshmdEnergyStruct(ctypes.Structure):
    _fields_ = [
        ("bond", ctypes.c_double),
        ("angle", ctypes.c_double),
        ("dihedral", ctypes.c_double),
        ("pair14_lj", ctypes.c_double),
        ("pair14_el", ctypes.c_double),
        ("lj_sr", ctypes.c_double),
        ("lj_disp", ctypes.c_double),
        ("elec_dir", ctypes.c_double),
        ("elec_recip", ctypes.c_double),
        ("elec_self", ctypes.c_double),
        ("elec_excl", ctypes.c_double),
        ("gb_pol", ctypes.c_double),
        ("np_sasa", ctypes.c_double),
        ("cap_shell", ctypes.c_double),
        ("tether_buf", ctypes.c_double),
        ("potential", ctypes.c_double),
        ("kinetic", ctypes.c_double),
        ("total", ctypes.c_double),
        ("temperature", ctypes.c_double),
        ("raw_pep_torsion", ctypes.c_double),
        ("raw_interface_nonb", ctypes.c_double),
    ]

    def to_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name, _ in self._fields_}


def _locate_or_build_library() -> pathlib.Path:
    ext = "dylib" if platform.system() == "Darwin" else "so"
    lib_path = _CORE_DIR / "build" / f"libashmd.{ext}"
    if not lib_path.is_file():
        subprocess.run(["make"], cwd=str(_CORE_DIR), check=True)
    if not lib_path.is_file():
        raise RuntimeError(f"Failed to build ASH-MD native library at {lib_path}")
    return lib_path


_LIB_SINGLETON: ctypes.CDLL | None = None


def _get_cdll() -> ctypes.CDLL:
    global _LIB_SINGLETON
    if _LIB_SINGLETON is not None:
        return _LIB_SINGLETON

    lib = ctypes.CDLL(str(_locate_or_build_library()))
    c_i32 = ctypes.c_int32
    c_u64 = ctypes.c_uint64
    c_dbl = ctypes.c_double
    p_i32 = ctypes.POINTER(ctypes.c_int32)
    p_i64 = ctypes.POINTER(ctypes.c_int64)
    p_u64 = ctypes.POINTER(ctypes.c_uint64)
    p_dbl = ctypes.POINTER(ctypes.c_double)
    p_eng = ctypes.POINTER(AshmdEnergyStruct)

    lib.ashmd_simd_backend.argtypes = []
    lib.ashmd_simd_backend.restype = ctypes.c_char_p

    lib.ashmd_create_engine.argtypes = [c_i32]
    lib.ashmd_create_engine.restype = ctypes.c_void_p

    lib.ashmd_destroy_engine.argtypes = [ctypes.c_void_p]
    lib.ashmd_destroy_engine.restype = None

    lib.ashmd_last_error.argtypes = [ctypes.c_void_p]
    lib.ashmd_last_error.restype = ctypes.c_char_p

    lib.ashmd_num_replicas.argtypes = [ctypes.c_void_p]
    lib.ashmd_num_replicas.restype = c_i32

    lib.ashmd_set_atoms.argtypes = [
        ctypes.c_void_p, c_i32, c_i32,
        p_dbl, p_dbl, p_dbl, p_dbl, p_dbl, p_dbl,
        p_i32, p_i32,
    ]
    lib.ashmd_set_atoms.restype = c_i32

    lib.ashmd_set_res_diel.argtypes = [ctypes.c_void_p, p_dbl]
    lib.ashmd_set_res_diel.restype = c_i32

    lib.ashmd_set_nonbonded_params.argtypes = [
        ctypes.c_void_p,
        c_dbl, c_dbl, c_dbl,
        c_dbl, c_dbl, c_dbl,
        c_i32, c_i32, c_i32,
        c_dbl, c_dbl, c_dbl,
        c_dbl, c_dbl, c_dbl,
    ]
    lib.ashmd_set_nonbonded_params.restype = c_i32

    lib.ashmd_set_bonded_tables.argtypes = [
        ctypes.c_void_p,
        c_i32, p_i32, p_i32, p_dbl, p_dbl,
        c_i32, p_i32, p_i32, p_i32, p_dbl, p_dbl,
        c_i32, p_i32, p_i32, p_i32, p_i32, p_dbl, p_dbl, p_dbl, p_i32,
        c_i32, p_i32, p_i32, p_dbl, p_dbl, p_dbl,
        p_i32, c_i32, p_i32,
    ]
    lib.ashmd_set_bonded_tables.restype = c_i32

    lib.ashmd_set_constraints_and_filter.argtypes = [
        ctypes.c_void_p,
        c_i32, p_i32, p_i32, p_dbl,
        c_i32, p_i32, p_i32, p_i32,
        c_i32, p_i32, p_i32, p_i32,
    ]
    lib.ashmd_set_constraints_and_filter.restype = c_i32

    lib.ashmd_set_tethers_and_cap.argtypes = [
        ctypes.c_void_p,
        p_dbl, p_dbl, p_dbl, p_dbl,
        c_i32, p_i32,
        c_i32, p_i32,
        c_dbl, c_dbl, c_dbl,
    ]
    lib.ashmd_set_tethers_and_cap.restype = c_i32

    lib.ashmd_init_replicas.argtypes = [
        ctypes.c_void_p, c_i32,
        p_dbl, p_dbl, p_dbl,
        c_u64,
        p_dbl, p_dbl, p_dbl,
        c_dbl,
    ]
    lib.ashmd_init_replicas.restype = c_i32

    lib.ashmd_set_state.argtypes = [
        ctypes.c_void_p, c_i32,
        p_dbl, p_dbl, p_dbl,
        p_dbl, p_dbl, p_dbl,
    ]
    lib.ashmd_set_state.restype = c_i32

    lib.ashmd_get_state.argtypes = [
        ctypes.c_void_p, c_i32,
        p_dbl, p_dbl, p_dbl,
        p_dbl, p_dbl, p_dbl,
        p_eng,
    ]
    lib.ashmd_get_state.restype = c_i32

    lib.ashmd_eval_single_point.argtypes = [
        ctypes.c_void_p, c_i32, c_dbl,
        p_dbl, p_dbl, p_dbl,
        p_eng,
    ]
    lib.ashmd_eval_single_point.restype = c_i32

    lib.ashmd_eval_binding_dg.argtypes = [
        ctypes.c_void_p,
        p_dbl, p_dbl, p_dbl,
        c_dbl, c_dbl, c_dbl,
        p_dbl,
    ]
    lib.ashmd_eval_binding_dg.restype = c_i32

    lib.ashmd_eval_binding_rd_gbsa_sasa.argtypes = [
        ctypes.c_void_p,
        p_dbl, p_dbl, p_dbl,
        c_dbl, c_dbl, c_dbl,
        p_dbl, p_dbl, p_dbl,
    ]
    lib.ashmd_eval_binding_rd_gbsa_sasa.restype = c_i32

    lib.ashmd_set_replica_tempering.argtypes = [
        ctypes.c_void_p, p_dbl, p_dbl, p_dbl, c_dbl
    ]
    lib.ashmd_set_replica_tempering.restype = c_i32

    lib.ashmd_eval_cross_potentials.argtypes = [
        ctypes.c_void_p, c_dbl, p_dbl
    ]
    lib.ashmd_eval_cross_potentials.restype = c_i32

    lib.ashmd_minimize.argtypes = [ctypes.c_void_p, c_i32, c_dbl, c_dbl]
    lib.ashmd_minimize.restype = c_i32

    lib.ashmd_run_steps.argtypes = [
        ctypes.c_void_p, c_i32, c_dbl, c_i32, c_dbl, c_dbl, c_i32, c_i32
    ]
    lib.ashmd_run_steps.restype = c_i32

    lib.ashmd_get_mbar_summary.argtypes = [
        ctypes.c_void_p, p_i32, p_dbl, p_dbl, p_i32, p_i64, p_i64
    ]
    lib.ashmd_get_mbar_summary.restype = c_i32

    lib.ashmd_save_checkpoint.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.ashmd_save_checkpoint.restype = c_i32

    lib.ashmd_load_checkpoint.argtypes = [ctypes.c_void_p, ctypes.c_char_p, p_u64]
    lib.ashmd_load_checkpoint.restype = c_i32

    _LIB_SINGLETON = lib
    return lib


def simd_backend() -> str:
    return _get_cdll().ashmd_simd_backend().decode("utf-8")


def _ptr_f64(arr: np.ndarray | None):
    if arr is None or arr.size == 0:
        return None
    assert arr.dtype == np.float64 and arr.flags["C_CONTIGUOUS"]
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _ptr_i32(arr: np.ndarray | None):
    if arr is None or arr.size == 0:
        return None
    assert arr.dtype == np.int32 and arr.flags["C_CONTIGUOUS"]
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


class AshmdCoreLibrary:
    """RAII wrapper around a C++17 ASH-MD EngineHandle."""

    def __init__(self, num_threads: int = 4) -> None:
        self._lib = _get_cdll()
        self._handle = self._lib.ashmd_create_engine(int(max(1, num_threads)))
        if not self._handle:
            raise MemoryError("Failed to allocate native AshmdEngine")
        self.n_atoms = 0
        self.n_replicas = 1

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.ashmd_destroy_engine(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "AshmdCoreLibrary":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def last_error(self) -> str:
        if not self._handle:
            return "Closed handle"
        raw = self._lib.ashmd_last_error(self._handle)
        return raw.decode("utf-8", errors="replace") if raw else ""

    def upload_system(self, sys_obj: Any, filter_alpha: float | None = None) -> None:
        n = int(sys_obj.n_atoms)
        self.n_atoms = n
        mode_int = 0 if sys_obj.solvent_mode == "explicit-pme" else 1

        rc = self._lib.ashmd_set_atoms(
            self._handle, n, mode_int,
            _ptr_f64(sys_obj.mass),
            _ptr_f64(sys_obj.charge),
            _ptr_f64(sys_obj.rmin_half),
            _ptr_f64(sys_obj.epsilon),
            _ptr_f64(sys_obj.gb_radius),
            _ptr_f64(sys_obj.gb_screen),
            _ptr_i32(sys_obj.zone),
            _ptr_i32(sys_obj.role),
        )
        if rc != 0:
            raise RuntimeError(f"ashmd_set_atoms failed ({rc})")

        res_diel_arr = getattr(sys_obj, "res_diel", None)
        if res_diel_arr is not None and len(res_diel_arr) == n:
            rd_contig = np.ascontiguousarray(res_diel_arr, dtype=np.float64)
            rc = self._lib.ashmd_set_res_diel(self._handle, _ptr_f64(rd_contig))
            if rc != 0:
                raise RuntimeError(f"ashmd_set_res_diel failed ({rc})")

        fa = float(sys_obj.filter_alpha if filter_alpha is None else filter_alpha)
        rc = self._lib.ashmd_set_nonbonded_params(
            self._handle,
            float(sys_obj.box[0]), float(sys_obj.box[1]), float(sys_obj.box[2]),
            float(sys_obj.cutoff), float(sys_obj.switch_on), float(sys_obj.ewald_alpha),
            int(sys_obj.pme_grid[0]), int(sys_obj.pme_grid[1]), int(sys_obj.pme_grid[2]),
            float(sys_obj.gb_cutoff), float(sys_obj.eps_in), float(sys_obj.eps_out),
            float(sys_obj.ionic_strength), float(sys_obj.sasa_gamma),
            fa,
        )
        if rc != 0:
            raise RuntimeError(f"ashmd_set_nonbonded_params failed ({rc})")

        rc = self._lib.ashmd_set_bonded_tables(
            self._handle,
            len(sys_obj.bond_i), _ptr_i32(sys_obj.bond_i), _ptr_i32(sys_obj.bond_j),
            _ptr_f64(sys_obj.bond_kr), _ptr_f64(sys_obj.bond_req),
            len(sys_obj.ang_i), _ptr_i32(sys_obj.ang_i), _ptr_i32(sys_obj.ang_j), _ptr_i32(sys_obj.ang_k),
            _ptr_f64(sys_obj.ang_kt), _ptr_f64(sys_obj.ang_eq),
            len(sys_obj.dih_i), _ptr_i32(sys_obj.dih_i), _ptr_i32(sys_obj.dih_j),
            _ptr_i32(sys_obj.dih_k), _ptr_i32(sys_obj.dih_l),
            _ptr_f64(sys_obj.dih_pk), _ptr_f64(sys_obj.dih_n),
            _ptr_f64(sys_obj.dih_phase), _ptr_i32(sys_obj.dih_is_pep),
            len(sys_obj.p14_i), _ptr_i32(sys_obj.p14_i), _ptr_i32(sys_obj.p14_j),
            _ptr_f64(sys_obj.p14_qprod), _ptr_f64(sys_obj.p14_rmin), _ptr_f64(sys_obj.p14_eps),
            _ptr_i32(sys_obj.excl_offsets),
            len(sys_obj.excl_targets), _ptr_i32(sys_obj.excl_targets),
        )
        if rc != 0:
            raise RuntimeError(f"ashmd_set_bonded_tables failed ({rc})")

        rc = self._lib.ashmd_set_constraints_and_filter(
            self._handle,
            len(sys_obj.con_i), _ptr_i32(sys_obj.con_i), _ptr_i32(sys_obj.con_j), _ptr_f64(sys_obj.con_d2),
            len(sys_obj.wat_o), _ptr_i32(sys_obj.wat_o), _ptr_i32(sys_obj.wat_h1), _ptr_i32(sys_obj.wat_h2),
            len(sys_obj.fg_heavy), _ptr_i32(sys_obj.fg_heavy), _ptr_i32(sys_obj.fg_nh), _ptr_i32(sys_obj.fg_h_flat),
        )
        if rc != 0:
            raise RuntimeError(f"ashmd_set_constraints_and_filter failed ({rc})")

        ref_x = np.ascontiguousarray(sys_obj.coords[:, 0], dtype=np.float64)
        ref_y = np.ascontiguousarray(sys_obj.coords[:, 1], dtype=np.float64)
        ref_z = np.ascontiguousarray(sys_obj.coords[:, 2], dtype=np.float64)
        rc = self._lib.ashmd_set_tethers_and_cap(
            self._handle,
            _ptr_f64(ref_x), _ptr_f64(ref_y), _ptr_f64(ref_z), _ptr_f64(sys_obj.tether_k),
            len(sys_obj.cap_anchors), _ptr_i32(sys_obj.cap_anchors),
            len(sys_obj.shell_water_oxygens), _ptr_i32(sys_obj.shell_water_oxygens),
            float(sys_obj.cap_radius), float(sys_obj.cap_k), float(sys_obj.cap_soft_beta),
        )
        if rc != 0:
            raise RuntimeError(f"ashmd_set_tethers_and_cap failed ({rc})")

    def init_replicas(
        self,
        coords: np.ndarray,
        n_replicas: int = 1,
        lambda_inter: list[float] | None = None,
        mu_torsion: list[float] | None = None,
        softcore_alpha: list[float] | None = None,
        seed: int = 42,
        init_temp_k: float = 310.15,
    ) -> None:
        self.n_replicas = int(max(1, n_replicas))
        lam = np.ascontiguousarray(lambda_inter if lambda_inter is not None else [1.0] * self.n_replicas, dtype=np.float64)
        mu = np.ascontiguousarray(mu_torsion if mu_torsion is not None else [1.0] * self.n_replicas, dtype=np.float64)
        sc = np.ascontiguousarray(softcore_alpha if softcore_alpha is not None else [0.0] * self.n_replicas, dtype=np.float64)

        rx = np.ascontiguousarray(coords[:, 0], dtype=np.float64)
        ry = np.ascontiguousarray(coords[:, 1], dtype=np.float64)
        rz = np.ascontiguousarray(coords[:, 2], dtype=np.float64)

        rc = self._lib.ashmd_init_replicas(
            self._handle, self.n_replicas,
            _ptr_f64(lam), _ptr_f64(mu), _ptr_f64(sc),
            int(seed),
            _ptr_f64(rx), _ptr_f64(ry), _ptr_f64(rz),
            float(init_temp_k),
        )
        if rc != 0:
            raise RuntimeError(f"ashmd_init_replicas failed ({rc})")

    def set_state(
        self,
        replica_idx: int,
        coords: np.ndarray | None = None,
        velocities: np.ndarray | None = None,
    ) -> None:
        rx = ry = rz = None
        if coords is not None:
            rx = np.ascontiguousarray(coords[:, 0], dtype=np.float64)
            ry = np.ascontiguousarray(coords[:, 1], dtype=np.float64)
            rz = np.ascontiguousarray(coords[:, 2], dtype=np.float64)
        vx = vy = vz = None
        if velocities is not None:
            vx = np.ascontiguousarray(velocities[:, 0], dtype=np.float64)
            vy = np.ascontiguousarray(velocities[:, 1], dtype=np.float64)
            vz = np.ascontiguousarray(velocities[:, 2], dtype=np.float64)
        rc = self._lib.ashmd_set_state(
            self._handle, int(replica_idx),
            _ptr_f64(rx), _ptr_f64(ry), _ptr_f64(rz),
            _ptr_f64(vx), _ptr_f64(vy), _ptr_f64(vz),
        )
        if rc != 0:
            raise RuntimeError(f"ashmd_set_state failed ({rc})")

    def get_state(self, replica_idx: int = 0) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        rx = np.empty(self.n_atoms, dtype=np.float64)
        ry = np.empty(self.n_atoms, dtype=np.float64)
        rz = np.empty(self.n_atoms, dtype=np.float64)
        vx = np.empty(self.n_atoms, dtype=np.float64)
        vy = np.empty(self.n_atoms, dtype=np.float64)
        vz = np.empty(self.n_atoms, dtype=np.float64)
        eng = AshmdEnergyStruct()
        rc = self._lib.ashmd_get_state(
            self._handle, int(replica_idx),
            _ptr_f64(rx), _ptr_f64(ry), _ptr_f64(rz),
            _ptr_f64(vx), _ptr_f64(vy), _ptr_f64(vz),
            ctypes.byref(eng),
        )
        if rc != 0:
            raise RuntimeError(f"ashmd_get_state failed ({rc})")
        return np.column_stack([rx, ry, rz]), np.column_stack([vx, vy, vz]), eng.to_dict()

    def eval_single_point(
        self,
        replica_idx: int = 0,
        target_temp_k: float = 310.15,
    ) -> tuple[np.ndarray, dict[str, float]]:
        fx = np.empty(self.n_atoms, dtype=np.float64)
        fy = np.empty(self.n_atoms, dtype=np.float64)
        fz = np.empty(self.n_atoms, dtype=np.float64)
        eng = AshmdEnergyStruct()
        rc = self._lib.ashmd_eval_single_point(
            self._handle, int(replica_idx), float(target_temp_k),
            _ptr_f64(fx), _ptr_f64(fy), _ptr_f64(fz),
            ctypes.byref(eng),
        )
        if rc != 0:
            raise RuntimeError(f"ashmd_eval_single_point failed ({rc})")
        return np.column_stack([fx, fy, fz]), eng.to_dict()

    def eval_binding_dg(
        self,
        coords: np.ndarray,
        target_temp_k: float = 310.15,
        sasa_gamma_gbsa: float = 0.0072,
        eps_in_gbsa: float = 2.0,
    ) -> dict[str, float]:
        rx = np.ascontiguousarray(coords[:, 0], dtype=np.float64)
        ry = np.ascontiguousarray(coords[:, 1], dtype=np.float64)
        rz = np.ascontiguousarray(coords[:, 2], dtype=np.float64)
        out5 = np.zeros(5, dtype=np.float64)
        rc = self._lib.ashmd_eval_binding_dg(
            self._handle,
            _ptr_f64(rx), _ptr_f64(ry), _ptr_f64(rz),
            float(target_temp_k), float(sasa_gamma_gbsa), float(eps_in_gbsa),
            _ptr_f64(out5),
        )
        if rc != 0:
            raise RuntimeError(f"ashmd_eval_binding_dg failed ({rc})")
        return {
            "mmgbsa_vdw_kcal": float(out5[0]),
            "mmgbsa_elec_kcal": float(out5[1]),
            "mmgbsa_gb_kcal": float(out5[2]),
            "mmgbsa_np_kcal": float(out5[3]),
            "mmgbsa_dg_kcal": float(out5[4]),
        }

    def eval_binding_rd_gbsa_sasa(
        self,
        coords: np.ndarray,
        target_temp_k: float = 310.15,
        cav_gamma: float = 0.0072,
        disp_scale: float = 1.0,
        return_atom_sasa: bool = True,
    ) -> tuple[dict[str, float], np.ndarray | None, np.ndarray | None]:
        rx = np.ascontiguousarray(coords[:, 0], dtype=np.float64)
        ry = np.ascontiguousarray(coords[:, 1], dtype=np.float64)
        rz = np.ascontiguousarray(coords[:, 2], dtype=np.float64)
        out8 = np.zeros(8, dtype=np.float64)
        sasa_free = np.zeros(self.n_atoms, dtype=np.float64) if return_atom_sasa else None
        sasa_bound = np.zeros(self.n_atoms, dtype=np.float64) if return_atom_sasa else None

        rc = self._lib.ashmd_eval_binding_rd_gbsa_sasa(
            self._handle,
            _ptr_f64(rx), _ptr_f64(ry), _ptr_f64(rz),
            float(target_temp_k), float(cav_gamma), float(disp_scale),
            _ptr_f64(out8),
            _ptr_f64(sasa_free) if sasa_free is not None else None,
            _ptr_f64(sasa_bound) if sasa_bound is not None else None,
        )
        if rc != 0:
            raise RuntimeError(f"ashmd_eval_binding_rd_gbsa_sasa failed ({rc})")

        terms = {
            "mmgbsa_vdw_kcal": float(out8[0]),
            "disp_solv_kcal": float(out8[1]),
            "cav_sasa_kcal": float(out8[2]),
            "mmgbsa_elec_kcal": float(out8[3]),
            "mmgbsa_gb_kcal": float(out8[4]),
            "buried_sasa_a2": float(out8[5]),
            "mmgbsa_np_kcal": float(out8[1] + out8[2]),
            "nonpolar_net_kcal": float(out8[6]),
            "mmgbsa_dg_kcal": float(out8[7]),
        }
        return terms, sasa_free, sasa_bound

    def set_replica_tempering(
        self,
        lambda_inter: list[float],
        mu_torsion: list[float],
        softcore_alpha: list[float],
        eval_temp_k: float = 310.15,
    ) -> None:
        lam = np.ascontiguousarray(lambda_inter, dtype=np.float64)
        mu = np.ascontiguousarray(mu_torsion, dtype=np.float64)
        sc = np.ascontiguousarray(softcore_alpha, dtype=np.float64)
        rc = self._lib.ashmd_set_replica_tempering(
            self._handle, _ptr_f64(lam), _ptr_f64(mu), _ptr_f64(sc), float(eval_temp_k)
        )
        if rc != 0:
            raise RuntimeError(f"ashmd_set_replica_tempering failed ({rc})")

    def eval_cross_potentials(self, target_temp_k: float = 310.15) -> np.ndarray:
        M = max(1, int(self._lib.ashmd_num_replicas(self._handle)))
        u_flat = np.zeros(M * M, dtype=np.float64)
        rc = self._lib.ashmd_eval_cross_potentials(self._handle, float(target_temp_k), _ptr_f64(u_flat))
        if rc != 0:
            raise RuntimeError(f"ashmd_eval_cross_potentials failed ({rc})")
        return u_flat.reshape(M, M)

    def minimize(self, max_steps: int = 500, fmax_tol: float = 10.0, init_temp_after_k: float = 310.15) -> int:
        rc = self._lib.ashmd_minimize(self._handle, int(max_steps), float(fmax_tol), float(init_temp_after_k))
        if rc < 0:
            raise RuntimeError(f"ashmd_minimize failed ({rc})")
        return int(rc)

    def run_steps(
        self,
        n_outer_steps: int,
        dt_ps: float = 0.002,
        mts_ratio: int = 2,
        gamma_ps_inv: float = 1.0,
        target_temp_k: float = 310.15,
        exchange_every_outer: int = 50,
        record_mbar: bool = True,
    ) -> None:
        rc = self._lib.ashmd_run_steps(
            self._handle,
            int(n_outer_steps),
            float(dt_ps),
            int(mts_ratio),
            float(gamma_ps_inv),
            float(target_temp_k),
            int(exchange_every_outer),
            1 if record_mbar else 0,
        )
        if rc != 0:
            raise RuntimeError(f"ashmd_run_steps failed ({rc})")

    def get_mbar_summary(self) -> dict[str, Any]:
        n_samp = ctypes.c_int32(0)
        self._lib.ashmd_get_mbar_summary(self._handle, ctypes.byref(n_samp), None, None, None, None, None)
        N = int(n_samp.value)
        M = max(1, int(self._lib.ashmd_num_replicas(self._handle)))
        self.n_replicas = M
        f_states = np.zeros(M, dtype=np.float64)
        w0 = np.zeros(max(1, N), dtype=np.float64)
        orig = np.zeros(max(1, N), dtype=np.int32)
        att = np.zeros(max(1, M - 1), dtype=np.int64)
        acc = np.zeros(max(1, M - 1), dtype=np.int64)

        self._lib.ashmd_get_mbar_summary(
            self._handle,
            ctypes.byref(n_samp),
            _ptr_f64(f_states) if N > 0 else None,
            _ptr_f64(w0) if N > 0 else None,
            _ptr_i32(orig) if N > 0 else None,
            att.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)) if M > 1 else None,
            acc.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)) if M > 1 else None,
        )
        return {
            "n_samples": N,
            "f_dimensionless": f_states.tolist(),
            "w0_weights": w0[:N] if N > 0 else np.array([], dtype=np.float64),
            "origin_states": orig[:N] if N > 0 else np.array([], dtype=np.int32),
            "swap_attempts": att[: max(0, M - 1)].tolist(),
            "swap_accepts": acc[: max(0, M - 1)].tolist(),
        }

    def save_checkpoint(self, path: str | os.PathLike) -> None:
        bpath = str(path).encode("utf-8")
        rc = self._lib.ashmd_save_checkpoint(self._handle, bpath)
        if rc != 0:
            raise RuntimeError(f"Checkpoint save failed ({rc}): {self.last_error()}")

    def load_checkpoint(self, path: str | os.PathLike) -> int:
        bpath = str(path).encode("utf-8")
        step_done = ctypes.c_uint64(0)
        rc = self._lib.ashmd_load_checkpoint(self._handle, bpath, ctypes.byref(step_done))
        if rc != 0:
            raise RuntimeError(f"Checkpoint load failed ({rc}): {self.last_error()}")
        self.n_replicas = max(1, int(self._lib.ashmd_num_replicas(self._handle)))
        return int(step_done.value)
