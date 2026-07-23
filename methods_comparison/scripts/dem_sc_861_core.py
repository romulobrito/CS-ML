#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective-medium rock physics core for Well 861 (DEM + SC).

Berryman DEM and self-consistent scheme (Berryman 1992; see crosscheck_dem_rockphypy.py).
Moduli in GPa; density in g/cc; velocities in m/s and km/s in exported tables.
ASCII-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.integrate import odeint

# Mineral end-members (GPa, g/cc) -- carbonate defaults for MOGNO
CALCITE_K_GPA: float = 77.0
CALCITE_G_GPA: float = 32.0
CALCITE_RHO_GCC: float = 2.71

DOLOMITE_K_GPA: float = 95.0
DOLOMITE_G_GPA: float = 45.0
DOLOMITE_RHO_GCC: float = 2.87

GPa_TO_PA: float = 1.0e9
GCC_TO_KG_M3: float = 1000.0

PHI_MIN: float = 1.0e-4
PHI_MAX: float = 0.95
SC_ITERATIONS: int = 80


@dataclass(frozen=True)
class MatrixModuli:
    """Mineral matrix elastic properties."""

    k_gpa: float
    g_gpa: float
    rho_gcc: float


@dataclass(frozen=True)
class DryElasticResult:
    """Dry-frame moduli from one effective-medium model."""

    k_gpa: float
    g_gpa: float
    model: str


@dataclass(frozen=True)
class VelocityResult:
    """P and S velocities and ratio."""

    vp_m_s: float
    vs_m_s: float
    vp_vs: float
    vp_km_s: float
    vs_km_s: float
    rho_gcc: float


def vrh_hill(volumes: np.ndarray, moduli: np.ndarray) -> float:
    """Voigt-Reuss-Hill average of elastic modulus."""
    vol = np.asarray(volumes, dtype=np.float64)
    mod = np.asarray(moduli, dtype=np.float64)
    if vol.shape != mod.shape:
        raise ValueError("volumes and moduli must have same shape")
    if np.any(vol < 0.0):
        raise ValueError("volume fractions must be non-negative")
    s = float(vol.sum())
    if s <= 0.0:
        raise ValueError("volume sum must be positive")
    vol = vol / s
    m_v = float(np.dot(vol, mod))
    m_r = float(1.0 / np.dot(vol, 1.0 / mod))
    return 0.5 * (m_v + m_r)


def matrix_from_solids(
    solid1_pct: float,
    solid2_pct: float,
) -> MatrixModuli:
    """
    Build matrix moduli from corrected solid fractions (%).

    solid1 -> calcite, solid2 -> dolomite (MOGNO assumption; document sensitivity).
    """
    s1 = max(float(solid1_pct), 0.0)
    s2 = max(float(solid2_pct), 0.0)
    total = s1 + s2
    if total <= 0.0:
        raise ValueError("solid fractions must sum to a positive value")
    f1 = s1 / total
    f2 = s2 / total
    vol = np.array([f1, f2], dtype=np.float64)
    k_gpa = vrh_hill(vol, np.array([CALCITE_K_GPA, DOLOMITE_K_GPA]))
    g_gpa = vrh_hill(vol, np.array([CALCITE_G_GPA, DOLOMITE_G_GPA]))
    rho = vrh_hill(vol, np.array([CALCITE_RHO_GCC, DOLOMITE_RHO_GCC]))
    return MatrixModuli(k_gpa=k_gpa, g_gpa=g_gpa, rho_gcc=rho)


def clip_porosity(phi: float) -> float:
    """Clip porosity to a numerically safe range for DEM/SC."""
    return float(np.clip(phi, PHI_MIN, PHI_MAX))


def pq_factors(
    km: float,
    gm: float,
    ki: float,
    gi: float,
    alpha: float,
) -> Tuple[float, float]:
    """Berryman geometric strain concentration factors P and Q."""
    if alpha == 1.0:
        p_val = (km + 4.0 * gm / 3.0) / (ki + 4.0 * gm / 3.0)
        kesai = gm / 6.0 * (9.0 * km + 8.0 * gm) / (km + 2.0 * gm)
        q_val = (gm + kesai) / (gi + kesai)
        return float(p_val), float(q_val)

    a = float(alpha)
    if a < 1.0:
        theta = a / (1.0 - a ** 2) ** 1.5 * (
            np.arccos(a) - a * np.sqrt(1.0 - a ** 2)
        )
    else:
        theta = a / (a ** 2 - 1.0) ** 1.5 * (
            a * np.sqrt(a ** 2 - 1.0) - np.arccosh(a)
        )
    f_shape = a ** 2 * (3.0 * theta - 2.0) / (1.0 - a ** 2)

    big_a = gi / gm - 1.0
    big_b = (ki / km - gi / gm) / 3.0
    r_val = gm / (km + (4.0 / 3.0) * gm)

    f1 = 1.0 + big_a * (
        1.5 * (f_shape + theta)
        - r_val * (1.5 * f_shape + 2.5 * theta - 4.0 / 3.0)
    )
    f2 = (
        1.0
        + big_a * (1.0 + 1.5 * (f_shape + theta) - r_val * (1.5 * f_shape + 2.5 * theta))
        + big_b * (3.0 - 4.0 * r_val)
        + big_a
        * (big_a + 3.0 * big_b)
        * (1.5 - 2.0 * r_val)
        * (f_shape + theta - r_val * (f_shape - theta + 2.0 * theta ** 2))
    )
    f3 = 1.0 + big_a * (1.0 - f_shape - 1.5 * theta + r_val * (f_shape + theta))
    f4 = 1.0 + (big_a / 4.0) * (f_shape + 3.0 * theta - r_val * (f_shape - theta))
    f5 = big_a * (-f_shape + r_val * (f_shape + theta - 4.0 / 3.0)) + big_b * theta * (
        3.0 - 4.0 * r_val
    )
    f6 = (
        1.0
        + big_a * (1.0 + f_shape - r_val * (f_shape + theta))
        + big_b * (1.0 - theta) * (3.0 - 4.0 * r_val)
    )
    f7 = (
        2.0
        + (big_a / 4.0) * (3.0 * f_shape + 9.0 * theta - r_val * (3.0 * f_shape + 5.0 * theta))
        + big_b * theta * (3.0 - 4.0 * r_val)
    )
    f8 = (
        big_a * (1.0 - 2.0 * r_val + (f_shape / 2.0) * (r_val - 1.0) + (theta / 2.0) * (5.0 * r_val - 3.0))
        + big_b * (1.0 - theta) * (3.0 - 4.0 * r_val)
    )
    f9 = big_a * ((r_val - 1.0) * f_shape - r_val * theta) + big_b * theta * (3.0 - 4.0 * r_val)

    tiijj = 3.0 * f1 / f2
    tijij = (
        tiijj / 3.0
        + 2.0 / f3
        + 1.0 / f4
        + (f4 * f5 + f6 * f7 - f8 * f9) / (f2 * f4)
    )
    p_val = tiijj / 3.0
    q_val = (tijij - p_val) / 5.0
    return float(p_val), float(q_val)


def _dem_ode(state: np.ndarray, t: float, params: Tuple[float, float, float]) -> list:
    """ODE system for Berryman DEM."""
    k_eff, g_eff = float(state[0]), float(state[1])
    gi, ki, alpha = params
    p_val, q_val = pq_factors(k_eff, g_eff, ki, gi, alpha)
    if t >= 1.0 - 1.0e-8:
        return [0.0, 0.0]
    denom = 1.0 - t
    dk = (ki - k_eff) * p_val / denom
    dg = (gi - g_eff) * q_val / denom
    return [dk, dg]


def berryman_dem(
    km_gpa: float,
    gm_gpa: float,
    ki_gpa: float,
    gi_gpa: float,
    alpha: float,
    phi: float,
) -> DryElasticResult:
    """Berryman differential effective medium (dry pores if ki=gi=0)."""
    phi_use = clip_porosity(phi)
    if phi_use <= 0.0:
        return DryElasticResult(k_gpa=km_gpa, g_gpa=gm_gpa, model="DEM")
    t_inc = 0.005
    t_grid = np.arange(0.0, phi_use + t_inc, t_inc, dtype=np.float64)
    params = (float(gi_gpa), float(ki_gpa), float(alpha))
    y0 = [float(km_gpa), float(gm_gpa)]
    sol = odeint(_dem_ode, y0, t_grid, args=(params,))
    k_final = float(sol[-1, 0])
    g_final = float(sol[-1, 1])
    if k_final <= 0.0 or g_final <= 0.0:
        raise ValueError("DEM produced non-positive moduli")
    return DryElasticResult(k_gpa=k_final, g_gpa=g_final, model="DEM")


def self_consistent_flex(
    km_gpa: float,
    gm_gpa: float,
    ki_gpa: float,
    gi_gpa: float,
    phi: float,
    iter_n: int = SC_ITERATIONS,
) -> DryElasticResult:
    """Self-consistent model for dry spherical/oblate inclusions (flex scheme)."""
    phi_use = clip_porosity(phi)
    if phi_use <= 0.0:
        return DryElasticResult(k_gpa=km_gpa, g_gpa=gm_gpa, model="SC")
    k_eff = float(km_gpa)
    g_eff = float(gm_gpa)
    f_val = float(phi_use)
    for _ in range(iter_n):
        nu = 0.5 * (3.0 * k_eff - 2.0 * g_eff) / (3.0 * k_eff + g_eff)
        s1 = (1.0 + nu) / (3.0 * (1.0 - nu))
        s2 = 2.0 * (4.0 - 5.0 * nu) / (15.0 * (1.0 - nu))
        if abs(k_eff - ki_gpa) < 1.0e-12:
            k_eff = km_gpa
        else:
            k_eff = (
                1.0
                - f_val
                * k_eff
                * (km_gpa - ki_gpa)
                / (km_gpa * (k_eff - ki_gpa))
                * (k_eff / (k_eff - ki_gpa) - s1) ** -1
            ) * km_gpa
        if abs(g_eff - gi_gpa) < 1.0e-12:
            g_eff = gm_gpa
        else:
            g_eff = (
                1.0
                - f_val
                * g_eff
                * (gm_gpa - gi_gpa)
                / (gm_gpa * (g_eff - gi_gpa))
                * (g_eff / (g_eff - gi_gpa) - s2) ** -1
            ) * gm_gpa
        if k_eff <= 0.0 or g_eff <= 0.0:
            raise ValueError("SC produced non-positive moduli")
    return DryElasticResult(k_gpa=k_eff, g_gpa=g_eff, model="SC")


def dry_density(phi: float, matrix_rho_gcc: float) -> float:
    """Dry bulk density (g/cc)."""
    phi_use = clip_porosity(phi)
    return (1.0 - phi_use) * matrix_rho_gcc


def saturated_density(
    phi: float,
    matrix_rho_gcc: float,
    rho_fluid_gcc: float,
    sw: float = 1.0,
) -> float:
    """Saturated bulk density (g/cc) with water/brine in pore space."""
    phi_use = clip_porosity(phi)
    sw_use = float(np.clip(sw, 0.0, 1.0))
    rho_f = sw_use * float(rho_fluid_gcc)
    return (1.0 - phi_use) * matrix_rho_gcc + phi_use * rho_f


def gassmann_bulk_saturation(
    k_dry_gpa: float,
    g_dry_gpa: float,
    k0_gpa: float,
    kf_gpa: float,
    phi: float,
) -> Tuple[float, float]:
    """
    Gassmann fluid substitution (Mavko et al. form).

    K0: mineral/matrix bulk modulus (GPa).
    Kf: pore fluid bulk modulus (GPa).
    Returns (K_sat, G_sat) in GPa; G_sat = G_dry.
    """
    phi_use = clip_porosity(phi)
    k_dry = float(k_dry_gpa)
    g_dry = float(g_dry_gpa)
    k0 = float(k0_gpa)
    kf = float(kf_gpa)
    if k0 <= 0.0 or kf <= 0.0 or k_dry <= 0.0 or g_dry <= 0.0:
        raise ValueError("Gassmann moduli must be positive")
    if phi_use <= 0.0:
        return k_dry, g_dry
    numer = (1.0 - k_dry / k0) ** 2
    denom = phi_use / kf + (1.0 - phi_use) / k0 - k_dry / (k0 ** 2)
    if abs(denom) < 1.0e-14:
        raise ValueError("Gassmann denominator near zero")
    k_sat = k_dry + numer / denom
    if k_sat <= 0.0:
        raise ValueError("Gassmann produced non-positive K_sat")
    return k_sat, g_dry


def velocities_from_moduli(
    k_gpa: float,
    g_gpa: float,
    rho_gcc: float,
) -> VelocityResult:
    """Compute Vp, Vs from moduli (GPa) and density (g/cc)."""
    if rho_gcc <= 0.0:
        raise ValueError("density must be positive")
    k_pa = k_gpa * GPa_TO_PA
    g_pa = g_gpa * GPa_TO_PA
    rho_si = rho_gcc * GCC_TO_KG_M3
    vp = float(np.sqrt((k_pa + (4.0 / 3.0) * g_pa) / rho_si))
    vs = float(np.sqrt(g_pa / rho_si))
    if vs <= 0.0:
        raise ValueError("shear modulus yielded zero Vs")
    return VelocityResult(
        vp_m_s=vp,
        vs_m_s=vs,
        vp_vs=vp / vs,
        vp_km_s=vp / 1000.0,
        vs_km_s=vs / 1000.0,
        rho_gcc=rho_gcc,
    )


def run_from_matrix_moduli(
    phi: float,
    alpha: float,
    km_gpa: float,
    gm_gpa: float,
    rho_matrix_gcc: float,
) -> dict:
    """
    DEM + SC + velocities using pre-calibrated matrix moduli (HFU extrapolation).
    """
    ki = 0.0
    gi = 0.0
    dem = berryman_dem(km_gpa, gm_gpa, ki, gi, alpha, phi)
    sc = self_consistent_flex(km_gpa, gm_gpa, ki, gi, phi)
    rho = dry_density(phi, rho_matrix_gcc)
    vel_dem = velocities_from_moduli(dem.k_gpa, dem.g_gpa, rho)
    vel_sc = velocities_from_moduli(sc.k_gpa, sc.g_gpa, rho)
    vp_rel_diff = abs(vel_dem.vp_m_s - vel_sc.vp_m_s) / vel_dem.vp_m_s
    vpvs_rel_diff = abs(vel_dem.vp_vs - vel_sc.vp_vs) / vel_dem.vp_vs
    return {
        "phi_input": phi,
        "alpha": alpha,
        "matrix_k_gpa": km_gpa,
        "matrix_g_gpa": gm_gpa,
        "matrix_rho_gcc": rho_matrix_gcc,
        "dem_k_gpa": dem.k_gpa,
        "dem_g_gpa": dem.g_gpa,
        "sc_k_gpa": sc.k_gpa,
        "sc_g_gpa": sc.g_gpa,
        "rho_gcc": rho,
        "vp_dem_m_s": vel_dem.vp_m_s,
        "vs_dem_m_s": vel_dem.vs_m_s,
        "vpvs_dem": vel_dem.vp_vs,
        "vp_dem_km_s": vel_dem.vp_km_s,
        "vs_dem_km_s": vel_dem.vs_km_s,
        "vp_sc_m_s": vel_sc.vp_m_s,
        "vs_sc_m_s": vel_sc.vs_m_s,
        "vpvs_sc": vel_sc.vp_vs,
        "vp_sc_km_s": vel_sc.vp_km_s,
        "vs_sc_km_s": vel_sc.vs_km_s,
        "vp_rel_diff_dem_sc": vp_rel_diff,
        "vpvs_rel_diff_dem_sc": vpvs_rel_diff,
    }


def run_from_matrix_moduli_saturated(
    phi: float,
    alpha: float,
    km_gpa: float,
    gm_gpa: float,
    rho_matrix_gcc: float,
    kf_gpa: float,
    rho_fluid_gcc: float,
    sw: float = 1.0,
) -> dict:
    """
    DEM dry frame + Gassmann saturation + velocities.

    Returns dry and saturated moduli/velocities; vp_dem_km_s is saturated.
    """
    dry = run_from_matrix_moduli(phi, alpha, km_gpa, gm_gpa, rho_matrix_gcc)
    k_sat, g_sat = gassmann_bulk_saturation(
        dry["dem_k_gpa"],
        dry["dem_g_gpa"],
        km_gpa,
        kf_gpa,
        phi,
    )
    rho_sat = saturated_density(phi, rho_matrix_gcc, rho_fluid_gcc, sw)
    vel_sat = velocities_from_moduli(k_sat, g_sat, rho_sat)
    vel_dry = velocities_from_moduli(dry["dem_k_gpa"], dry["dem_g_gpa"], dry["rho_gcc"])
    return {
        **dry,
        "kf_gpa": float(kf_gpa),
        "rho_fluid_gcc": float(rho_fluid_gcc),
        "sw": float(sw),
        "k_sat_gpa": k_sat,
        "g_sat_gpa": g_sat,
        "rho_sat_gcc": rho_sat,
        "vp_dem_dry_km_s": vel_dry.vp_km_s,
        "vs_dem_dry_km_s": vel_dry.vs_km_s,
        "vpvs_dem_dry": vel_dry.vp_vs,
        "vp_dem_m_s": vel_sat.vp_m_s,
        "vs_dem_m_s": vel_sat.vs_m_s,
        "vpvs_dem": vel_sat.vp_vs,
        "vp_dem_km_s": vel_sat.vp_km_s,
        "vs_dem_km_s": vel_sat.vs_km_s,
    }


def run_plug_case(
    phi_lab: float,
    alpha: float,
    solid1_pct: float,
    solid2_pct: float,
    matrix_k_scale: float = 1.0,
    matrix_g_scale: float = 1.0,
) -> dict:
    """
    Full DEM + SC + velocities for one plug (dry rock).

    Optional matrix_k_scale / matrix_g_scale apply to VRH matrix moduli before DEM.

    Returns dict with matrix, dry moduli, velocities, and DEM-SC deltas.
    """
    matrix = matrix_from_solids(solid1_pct, solid2_pct)
    km = matrix.k_gpa * float(matrix_k_scale)
    gm = matrix.g_gpa * float(matrix_g_scale)
    ki = 0.0
    gi = 0.0
    dem = berryman_dem(km, gm, ki, gi, alpha, phi_lab)
    sc = self_consistent_flex(km, gm, ki, gi, phi_lab)
    rho = dry_density(phi_lab, matrix.rho_gcc)
    vel_dem = velocities_from_moduli(dem.k_gpa, dem.g_gpa, rho)
    vel_sc = velocities_from_moduli(sc.k_gpa, sc.g_gpa, rho)

    vp_rel_diff = abs(vel_dem.vp_m_s - vel_sc.vp_m_s) / vel_dem.vp_m_s
    vpvs_rel_diff = abs(vel_dem.vp_vs - vel_sc.vp_vs) / vel_dem.vp_vs

    return {
        "phi_lab": phi_lab,
        "alpha": alpha,
        "matrix_k_scale": float(matrix_k_scale),
        "matrix_g_scale": float(matrix_g_scale),
        "matrix_k_gpa": km,
        "matrix_g_gpa": gm,
        "matrix_rho_gcc": matrix.rho_gcc,
        "dem_k_gpa": dem.k_gpa,
        "dem_g_gpa": dem.g_gpa,
        "sc_k_gpa": sc.k_gpa,
        "sc_g_gpa": sc.g_gpa,
        "rho_gcc": rho,
        "vp_dem_m_s": vel_dem.vp_m_s,
        "vs_dem_m_s": vel_dem.vs_m_s,
        "vpvs_dem": vel_dem.vp_vs,
        "vp_dem_km_s": vel_dem.vp_km_s,
        "vs_dem_km_s": vel_dem.vs_km_s,
        "vp_sc_m_s": vel_sc.vp_m_s,
        "vs_sc_m_s": vel_sc.vs_m_s,
        "vpvs_sc": vel_sc.vp_vs,
        "vp_sc_km_s": vel_sc.vp_km_s,
        "vs_sc_km_s": vel_sc.vs_km_s,
        "vp_rel_diff_dem_sc": vp_rel_diff,
        "vpvs_rel_diff_dem_sc": vpvs_rel_diff,
    }
