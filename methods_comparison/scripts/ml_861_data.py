#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Well 861 ML dataset loaders and feature/target builders.

Protocol: methods_comparison/planning/etapa1c_ml_baseline_poco861.md
ASCII-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WELL_ID = "861"
DEPTH_COL = "Depth(m)"

DEFAULT_ENRICHED = (
    ROOT / "methods_comparison" / "data" / "processed" / "861_integrated_logs_enriched.xlsx"
)
DEFAULT_CT = (
    ROOT / "methods_comparison" / "data" / "processed" / "861_integrated_ct_samples.xlsx"
)

LOG_FEATURE_COLUMNS: Tuple[str, ...] = (
    "GR (API)",
    "Density (g/cc)",
    "Res_Deep",
    "Res_Shallow",
    "Phi_Neutron (pu)",
    "Phi_Sonic (pu)",
    "Phi_ND (pu)",
    "Lithotype",
)

CT_FEATURE_COLUMNS: Tuple[str, ...] = (
    "ct_ar_mean",
    "ct_mean_gamma",
    "ct_porosity_pct",
    "phi_meso_macropores_vv",
    "ar_meso_macropores",
    "corrected_solid1_pct",
    "corrected_solid2_pct",
)

TARGET_COLUMNS: Tuple[str, ...] = (
    "FZI_lab",
    "Phi_lab (pu)",
    "k_lab (mD)",
    "HFU",
)

ORACLE_LAB_FEATURES: Tuple[str, ...] = (
    "Phi_lab (pu)",
    "k_lab (mD)",
)

LEAKAGE_FOR_FZI: Tuple[str, ...] = (
    "RQI",
    "FZI_lab",
    "Phi_lab (pu)",
    "k_lab (mD)",
    "HFU",
)

META_EXCLUDE: Tuple[str, ...] = (
    "Depth(m)",
    "depth_index",
    "well_id",
    "hfu_label",
    "has_ct_sample",
    "sample_id",
    "ct_depth_m",
    "depth_delta_m",
    "match_quality",
    "log_depth_m",
)

FeatureMode = Literal["log_only", "log_plus_ct"]
DatasetMode = Literal["enriched", "ct"]

ML_RUNS_ROOT = ROOT / "methods_comparison" / "data" / "processed" / "ml_runs"

# well_profile: 87-row enriched wireline table, depth-block CV
WELL_PROFILE_ML_ROOT = ML_RUNS_ROOT / "well_profile"
WELL_PROFILE_PRIMARY_TARGET = "Phi_lab (pu)"
WELL_PROFILE_HFU_TARGET = "HFU"

# ct_plugs: 10 microCT-integrated plugs, leave-one-plug-out CV
CT_PLUGS_ML_ROOT = ML_RUNS_ROOT / "ct_plugs"
CT_PLUGS_TARGETS: Tuple[str, ...] = ("Phi_lab (pu)", "FZI_lab")
CT_PLUGS_FEATURE_MODES: Tuple[FeatureMode, ...] = ("log_only", "log_plus_ct")

# Etapa 2: DEM/SC rock physics (Vp/Vs)
DEM_SC_ROOT = ROOT / "methods_comparison" / "data" / "processed" / "dem_sc_runs"
DEM_SC_POC_ROOT = DEM_SC_ROOT / "poc_10plugs"
DEM_SC_HFU_CALIB_ROOT = DEM_SC_ROOT / "hfu_calibration"
DEM_SC_PROFILE_ROOT = DEM_SC_ROOT / "profile_87"
DEM_SC_PROFILE_LAB_CALIB_ROOT = DEM_SC_ROOT / "profile_87_lab_calib"
DEM_SC_PROFILE_GASSMANN_ROOT = DEM_SC_ROOT / "profile_87_gassmann"
DEM_SC_LAB_VALIDATION_ROOT = DEM_SC_ROOT / "lab_validation"
DEM_SC_LAB_CALIBRATION_ROOT = DEM_SC_ROOT / "lab_calibration"
DEM_SC_MULTISCALE_AB_ROOT = DEM_SC_ROOT / "multiscale_ab"

ROCKPHYS_RAW_XLSX = (
    ROOT
    / "methods_comparison"
    / "data"
    / "ROCKPHYS_Database_04_12_2024 (7).xlsx"
)
ROCKPHYS_PROCESSED_ROOT = (
    ROOT / "methods_comparison" / "data" / "processed" / "rockphys_861"
)

# DLIS sonic extraction + DEM validation (Etapa 2f)
DLIS_RAW_DIR = ROOT / "methods_comparison" / "data"
DLIS_PROCESSED_ROOT = ROOT / "methods_comparison" / "data" / "processed" / "dlis_861"
DLIS_GASSMANN_ROOT = ROOT / "methods_comparison" / "data" / "processed" / "dlis_861_gassmann"
DLIS_SONIC_CSV = DLIS_PROCESSED_ROOT / "tables" / "sonic_log.csv"
DLIS_VALIDATION_CSV = DLIS_PROCESSED_ROOT / "tables" / "dem_vs_sonic_validation.csv"
DLIS_GASSMANN_VALIDATION_CSV = (
    DLIS_GASSMANN_ROOT / "tables" / "dem_vs_sonic_validation.csv"
)

# Etapa 3: ML residual on Vp (after Gassmann physics)
ML_RESIDUAL_VP_ROOT = ML_RUNS_ROOT / "residual_vp"
CLP_861_VP_RESIDUAL_ROOT = ML_RESIDUAL_VP_ROOT / "clp_csgm"
CLP_861_VP_RHO_ROOT = CLP_861_VP_RESIDUAL_ROOT / "rho_subsample"
RESIDUAL_VP_TARGET = "vp_residual_km_s"
RESIDUAL_VP_FEATURE_EXTRA: Tuple[str, ...] = ("vp_gassmann_km_s",)
LEAKAGE_FOR_VP_RESIDUAL: Tuple[str, ...] = (
    "vp_sonic_km_s",
    "vs_sonic_km_s",
    "vpvs_sonic",
    "vp_residual_km_s",
    "vp_bias_km_s",
    "vp_hybrid_km_s",
    "dtco_usft",
    "dtsm_usft",
)

# Etapa 1f: CLP-CSGM phi_lab profile reconstruction (861 MOGNO)
CLP_861_ML_ROOT = ML_RUNS_ROOT / "clp_861"
CLP_861_PRIMARY_TARGET = "Phi_lab (pu)"
CLP_861_DEPTH_MIN_M = 5205.91
CLP_861_DEPTH_MAX_M = 5233.72
CLP_861_SCENARIO_PLUG_SPARSE = "plug_sparse_b"
CLP_861_SCENARIO_RHO_SUBSAMPLE = "rho_subsample"
CLP_861_SCENARIO_WIRELINE_PLUS_CT = "wireline_plus_ct_u"


@dataclass(frozen=True)
class XYBundle:
    """Feature matrix, target vector, and column metadata."""

    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    target: str
    df: pd.DataFrame


@dataclass(frozen=True)
class DepthBlockFold:
    """One contiguous depth-block cross-validation fold."""

    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    depth_min_m: float
    depth_max_m: float


def load_logs_enriched(path: Optional[Path] = None) -> pd.DataFrame:
    """Load 87-row enriched well table."""
    p = Path(path) if path is not None else DEFAULT_ENRICHED
    if not p.is_file():
        raise FileNotFoundError(str(p))
    df = pd.read_excel(p)
    if DEPTH_COL not in df.columns:
        raise ValueError("Missing column: {}".format(DEPTH_COL))
    return df.sort_values(DEPTH_COL).reset_index(drop=True)


def load_ct_samples(path: Optional[Path] = None) -> pd.DataFrame:
    """Load 10-row CT-integrated sample table."""
    p = Path(path) if path is not None else DEFAULT_CT
    if not p.is_file():
        raise FileNotFoundError(str(p))
    df = pd.read_excel(p)
    if "sample_id" not in df.columns:
        raise ValueError("Missing column: sample_id")
    return df.sort_values("ct_depth_m").reset_index(drop=True)


def target_slug(target: str) -> str:
    """Filesystem-safe slug for target column names."""
    return (
        target.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )


def compare_out_dir_for_target(
    target: str,
    base: Optional[Path] = None,
) -> Path:
    """Per-target output directory under compare_861/by_target/."""
    root = base if base is not None else (
        ROOT / "methods_comparison" / "data" / "processed" / "ml_runs" / "compare_861"
    )
    return root / "by_target" / target_slug(target)


def feature_mode_slug(feature_mode: FeatureMode) -> str:
    """Filesystem slug describing wireline vs wireline+CT feature sets."""
    if feature_mode == "log_only":
        return "wireline_only"
    return "wireline_plus_ct"


def residual_vp_rf_dir() -> Path:
    """RF output for Vp residual ML (depth-block CV, Etapa 3)."""
    return ML_RESIDUAL_VP_ROOT / "rf"


def clp_vp_residual_dir() -> Path:
    """CLP-CSGM Vp residual outputs (Etapa 3b)."""
    return CLP_861_VP_RESIDUAL_ROOT


def clp_vp_rho_dir() -> Path:
    """CLP Vp rho subsample sweep (Etapa 3c)."""
    return CLP_861_VP_RHO_ROOT


def well_profile_phi_rf_dir() -> Path:
    """RF output for Phi_lab well-profile baseline (depth-block CV)."""
    return WELL_PROFILE_ML_ROOT / "phi_lab" / "rf"


def build_residual_feature_columns(df: pd.DataFrame) -> List[str]:
    """Wireline log features + Vp Gassmann for residual ML."""
    cols = list(LOG_FEATURE_COLUMNS)
    for extra in RESIDUAL_VP_FEATURE_EXTRA:
        if extra not in df.columns:
            raise ValueError("Missing residual feature column: {}".format(extra))
        cols.append(extra)
    leak = set(LEAKAGE_FOR_VP_RESIDUAL)
    return [c for c in cols if c not in leak]


def well_profile_phi_compare_dir() -> Path:
    """Five-regressor comparison for Phi_lab well-profile baseline."""
    return WELL_PROFILE_ML_ROOT / "phi_lab" / "compare"


def well_profile_phi_alternatives_dir() -> Path:
    """Ridge/GAM Phi_lab alternatives vs RF baseline."""
    return WELL_PROFILE_ML_ROOT / "phi_lab" / "alternatives"


def well_profile_hfu_classifier_dir() -> Path:
    """HFU classification well-profile depth-block CV."""
    return WELL_PROFILE_ML_ROOT / "hfu" / "classifier"


def clp_861_scenario_dir(
    scenario: str,
    base: Optional[Path] = None,
) -> Path:
    """Output directory for one CLP-861 scenario under clp_861/phi_lab/."""
    root = base if base is not None else CLP_861_ML_ROOT
    return root / "phi_lab" / scenario


def clp_861_compare_rf_dir() -> Path:
    """RF Phi_lab baseline copy or symlink for side-by-side CLP comparison."""
    return CLP_861_ML_ROOT / "compare_rf_baseline"


def ct_plugs_scenario_dir(
    target: str,
    feature_mode: FeatureMode,
    base: Optional[Path] = None,
) -> Path:
    """Output dir for one ct_plugs target + feature-mode scenario."""
    root = base if base is not None else CT_PLUGS_ML_ROOT
    return (
        root
        / "by_target"
        / target_slug(target)
        / feature_mode_slug(feature_mode)
    )


def load_dataset(mode: DatasetMode, path: Optional[Path] = None) -> pd.DataFrame:
    """Load enriched (87 rows) or ct (10 rows) table."""
    if mode == "enriched":
        return load_logs_enriched(path)
    return load_ct_samples(path)


def _leakage_columns(target: str) -> Tuple[str, ...]:
    if target == "FZI_lab":
        return LEAKAGE_FOR_FZI
    if target == "Phi_lab (pu)":
        return ("Phi_lab (pu)", "RQI", "FZI_lab")
    if target == "k_lab (mD)":
        return ("k_lab (mD)", "RQI", "FZI_lab")
    if target == "HFU":
        return ("HFU", "FZI_lab", "RQI")
    return (target,)


def resolve_feature_columns(
    df: pd.DataFrame,
    feature_mode: FeatureMode,
) -> List[str]:
    """Return ordered feature column names for the given mode."""
    cols = list(LOG_FEATURE_COLUMNS)
    missing_log = [c for c in cols if c not in df.columns]
    if missing_log:
        raise ValueError("Missing log feature columns: {}".format(missing_log))

    if feature_mode == "log_plus_ct":
        for c in CT_FEATURE_COLUMNS:
            if c in df.columns:
                cols.append(c)
    return cols


def build_xy_from_columns(
    df: pd.DataFrame,
    target: str,
    feature_columns: Sequence[str],
) -> XYBundle:
    """Build (X, y) from an explicit feature column list."""
    if target not in df.columns:
        raise ValueError("Target column not found: {}".format(target))
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError("Missing feature columns: {}".format(missing))

    feature_cols = list(feature_columns)
    work = df.dropna(subset=[target] + feature_cols).copy()
    if work.empty:
        raise ValueError("No rows left after dropping NaN for target={}".format(target))

    X = work[feature_cols].to_numpy(dtype=np.float64)
    y = work[target].to_numpy(dtype=np.float64)
    if target == "FZI_lab":
        y = np.round(y, 1)

    return XYBundle(
        X=X,
        y=y,
        feature_names=feature_cols,
        target=target,
        df=work.reset_index(drop=True),
    )


def build_xy(
    df: pd.DataFrame,
    target: str,
    feature_mode: FeatureMode = "log_only",
) -> XYBundle:
    """Build (X, y) arrays with leakage-safe feature selection."""
    if target not in df.columns:
        raise ValueError("Target column not found: {}".format(target))

    leak = set(_leakage_columns(target))
    feature_cols = [
        c for c in resolve_feature_columns(df, feature_mode) if c not in leak
    ]

    return build_xy_from_columns(df, target=target, feature_columns=feature_cols)


def depth_block_splits(
    df: pd.DataFrame,
    n_blocks: int = 3,
) -> List[DepthBlockFold]:
    """
    Contiguous depth-block CV folds.

    Each fold holds out one depth block; trains on the remaining blocks.
    """
    if n_blocks < 2:
        raise ValueError("n_blocks must be >= 2")
    if DEPTH_COL not in df.columns:
        raise ValueError("Missing depth column: {}".format(DEPTH_COL))

    ordered = df.sort_values(DEPTH_COL).reset_index(drop=True)
    n = len(ordered)
    if n < n_blocks:
        raise ValueError("Not enough rows ({}) for {} blocks".format(n, n_blocks))

    indices = np.arange(n)
    splits = np.array_split(indices, n_blocks)
    folds: List[DepthBlockFold] = []

    for fold_id, test_idx in enumerate(splits):
        test_idx = np.asarray(test_idx, dtype=np.int64)
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        train_idx = indices[train_mask]
        depths = ordered.loc[test_idx, DEPTH_COL]
        folds.append(
            DepthBlockFold(
                fold_id=fold_id,
                train_idx=train_idx,
                test_idx=test_idx,
                depth_min_m=float(depths.min()),
                depth_max_m=float(depths.max()),
            )
        )
    return folds


def leave_one_plug_out_splits(df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """Leave-one-sample-out splits for CT table (sample_id required)."""
    if "sample_id" not in df.columns:
        raise ValueError("sample_id required for plug-out CV")
    n = len(df)
    folds: List[Tuple[np.ndarray, np.ndarray, str]] = []
    for i in range(n):
        test_idx = np.array([i], dtype=np.int64)
        train_idx = np.array([j for j in range(n) if j != i], dtype=np.int64)
        sid = str(df.loc[i, "sample_id"])
        folds.append((train_idx, test_idx, sid))
    return folds


def iter_fold_arrays(
    bundle: XYBundle,
    folds: Sequence[DepthBlockFold],
) -> Iterator[Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield (fold_id, X_train, y_train, X_test, y_test) aligned to bundle.df order."""
    for fold in folds:
        train_idx = fold.train_idx
        test_idx = fold.test_idx
        yield (
            fold.fold_id,
            bundle.X[train_idx],
            bundle.y[train_idx],
            bundle.X[test_idx],
            bundle.y[test_idx],
        )
