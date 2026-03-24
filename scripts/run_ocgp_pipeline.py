"""
Pipeline for one-class GP experiments on fetal heart rate features.

The script reads the feature table, trains one GP per feature using
CAT-1 samples as healthy data, and then evaluates both per-feature
and ensemble anomaly scores against CAT-3.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.svm import OneClassSVM

# Configuration
RANDOM_STATE = 42
EPS = 1e-8
HEALTHY = 1.0
UNHEALTHY = 0.0

DATA_PATH = "data/features.xlsx"
OUT_DIR = "./ocgp_results"

LABEL_COL = "cat"
HEALTHY_LABEL = "CAT-1"
UNHEALTHY_LABEL = "CAT-3"

MAKE_PLOTS = True
SAVE_PNGS = True
CORR_PLOT = True

os.makedirs(OUT_DIR, exist_ok=True)
if SAVE_PNGS:
    os.makedirs(os.path.join(OUT_DIR, "figs"), exist_ok=True)

FEATURE_COLUMNS = [
    "baseline_fhr",
    "num_accelerations",
    "num_decelerations",
    "accel_duration_seconds",
    "decel_duration_seconds",
    "mean_fhr",
    "median_fhr",
    "std_fhr",
    "min_fhr",
    "max_fhr",
    "range_fhr",
    "rmssd",
    "peak_frequency",
    "lf_power",
    "hf_power",
    "lf_hf_ratio",
    "approx_entropy",
    "sample_entropy",
    "dfa",
    "variance_fhr",
    "iqr_fhr",
    "percentile_25",
    "percentile_75",
]

WINSOR_COLUMNS = [
    "num_accelerations",
    "num_decelerations",
    "accel_duration_seconds",
    "decel_duration_seconds",
    "lf_power",
    "hf_power",
    "lf_hf_ratio",
    "peak_frequency",
]

DROPPED_COLUMNS = {"lf_hf_ratio", "variance_fhr", "range_fhr", "iqr_fhr"}

LOG1P_FEATURES = [
    "num_accelerations",
    "num_decelerations",
    "accel_duration_seconds",
    "decel_duration_seconds",
    "lf_power",
    "hf_power",
]

SHARE_FEATURES = ["lf_share", "hf_share"]
YJ_COLUMNS = ["peak_frequency"]

FINAL_FEATURES = sorted(
    list((set(FEATURE_COLUMNS) - DROPPED_COLUMNS).union({"lf_share", "hf_share"}))
)


# Feature preprocessing
def normalize_labels(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)
    return s.str.upper()


def compute_winsor_thresholds(
    df: pd.DataFrame, cols: List[str], lower: float = 0.5, upper: float = 99.5
) -> Dict[str, tuple[float, float]]:
    """Return per-column percentile thresholds for winsorization."""
    thresholds = {}
    for col in cols:
        if col not in df.columns:
            continue
        arr = df[col].to_numpy()
        lo = np.nanpercentile(arr, lower)
        hi = np.nanpercentile(arr, upper)
        thresholds[col] = (lo, hi)
    return thresholds


def apply_winsorization(df: pd.DataFrame, thresholds: Dict[str, tuple[float, float]]) -> pd.DataFrame:
    df = df.copy()
    for col, (lo, hi) in thresholds.items():
        if col in df.columns:
            df[col] = np.clip(df[col].to_numpy(), lo, hi)
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "lf_power" in df.columns and "hf_power" in df.columns:
        total = df["lf_power"] + df["hf_power"] + 1e-6
        df["lf_share"] = df["lf_power"] / total
        df["hf_share"] = 1.0 - df["lf_share"]
    return df.drop(columns=[c for c in DROPPED_COLUMNS if c in df.columns], errors="ignore")


def apply_log1p(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = np.log1p(np.maximum(df[col].to_numpy(), 0))
    return df


def apply_logit_transform(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            x = np.clip(df[col].to_numpy(), 1e-6, 1 - 1e-6)
            df[col] = np.log(x / (1 - x))
    return df


def ocgp_score(gp: GaussianProcessRegressor, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    mu, std = gp.predict(X, return_std=True)
    std = np.maximum(std, 1e-6)
    z = (y - mu) / std
    return np.abs(z)


def extract_kernel_params(gp: GaussianProcessRegressor) -> Dict[str, float | List[float]]:
    params: Dict[str, float | List[float]] = {}
    try:
        kernel = gp.kernel_
        if hasattr(kernel, "k1") and hasattr(kernel, "k2"):
            k_sig = kernel.k1
            k_noise = kernel.k2
            if hasattr(k_sig, "k1") and hasattr(k_sig, "k2"):
                ck = k_sig.k1
                rbfk = k_sig.k2
                params["const"] = float(getattr(ck, "constant_value", np.nan))
                ls = getattr(rbfk, "length_scale", None)
                if isinstance(ls, np.ndarray):
                    params["length_scales"] = [float(x) for x in ls]
                else:
                    params["length_scales"] = [float(ls)] if ls is not None else []
            params["noise_level"] = float(getattr(k_noise, "noise_level", np.nan))
        else:
            if hasattr(kernel, "length_scale"):
                ls = kernel.length_scale
                params["length_scales"] = [
                    float(x) for x in (ls if isinstance(ls, np.ndarray) else [ls])
                ]
            if hasattr(kernel, "noise_level"):
                params["noise_level"] = float(kernel.noise_level)
    except Exception as e:
        params["extract_error"] = str(e)
    return params


# Main model fit
@dataclass
class TargetResult:
    target: str
    threshold: float
    metrics: Dict[str, float]
    kernel_params: Dict[str, float | List[float]]
    columns_used_as_X: List[str]
    holdout_idx: Optional[np.ndarray] = None
    unhealthy_idx: Optional[np.ndarray] = None
    scores_holdout: Optional[np.ndarray] = None
    scores_unhealthy: Optional[np.ndarray] = None


def fit_one_target(
    cleaned_df: pd.DataFrame,
    target_col: str,
    y_bin: np.ndarray,
    tr_abs: np.ndarray,
    ho_abs: np.ndarray,
    healthy_quantile: float = 0.95,
    restarts: int = 2,
) -> TargetResult:
    """Fit one GP model for a single target feature."""
    cols_all = list(FINAL_FEATURES)
    if target_col not in cols_all:
        raise ValueError(f"{target_col} not in FINAL_FEATURES.")

    x_cols = [c for c in cols_all if c != target_col]
    if not x_cols:
        raise ValueError("No predictors remain after excluding the target column.")

    unhealthy_idx = np.where(y_bin == UNHEALTHY)[0]

    # Fit preprocessing only on the healthy training split
    df_work = cleaned_df.copy()

    # Winsorize using thresholds from the healthy training split
    wins_eff = [c for c in WINSOR_COLUMNS if c in df_work.columns]
    if wins_eff:
        df_train = df_work.iloc[tr_abs].copy()
        wins_th = compute_winsor_thresholds(df_train[wins_eff], wins_eff, lower=0.5, upper=99.5)
        df_work = apply_winsorization(df_work, wins_th)

    # Add engineered features and apply monotone transforms
    df_work = add_engineered_features(df_work)
    df_work = apply_log1p(df_work, [c for c in LOG1P_FEATURES if c in df_work.columns])
    df_work = apply_logit_transform(df_work, [c for c in SHARE_FEATURES if c in df_work.columns])

    # Fit Yeo-Johnson on training data only
    if all(c in df_work.columns for c in YJ_COLUMNS):
        df_train = df_work.iloc[tr_abs].copy()
        do_yj = df_train[YJ_COLUMNS].nunique().min() > 1
        if do_yj:
            yj_model = PowerTransformer(method="yeo-johnson", standardize=False)
            yj_model.fit(df_train[YJ_COLUMNS].to_numpy())
            df_work.loc[:, YJ_COLUMNS] = yj_model.transform(df_work[YJ_COLUMNS].to_numpy())

    cols_present = [c for c in FINAL_FEATURES if c in df_work.columns]
    if target_col not in cols_present:
        raise ValueError(f"Target '{target_col}' missing after preprocessing.")

    x_all = df_work[cols_present].copy()

    x_train_raw = x_all.iloc[tr_abs][x_cols].values
    y_train = x_all.iloc[tr_abs][target_col].values

    x_hold_raw = x_all.iloc[ho_abs][x_cols].values
    y_hold = x_all.iloc[ho_abs][target_col].values

    if len(unhealthy_idx) > 0:
        x_unhealthy_raw = x_all.iloc[unhealthy_idx][x_cols].values
        y_unhealthy = x_all.iloc[unhealthy_idx][target_col].values
    else:
        x_unhealthy_raw = None
        y_unhealthy = None

    # Scale predictors
    scaler_x = RobustScaler()
    scaler_x.fit(x_train_raw)
    x_train = scaler_x.transform(x_train_raw)
    x_hold = scaler_x.transform(x_hold_raw)
    x_unhealthy = scaler_x.transform(x_unhealthy_raw) if x_unhealthy_raw is not None else None

    kernel = (
        C(1.0, (1e-3, 1e3))
        * RBF(length_scale=np.ones(len(x_cols)), length_scale_bounds=(1e-3, 1e3))
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=False,
        alpha=EPS,
        random_state=RANDOM_STATE,
        n_restarts_optimizer=restarts,
    )
    gp.fit(x_train, y_train)

    score_hold = ocgp_score(gp, x_hold, y_hold)
    threshold = float(np.quantile(score_hold, healthy_quantile))

    score_unhealthy = None
    if x_unhealthy is not None:
        score_unhealthy = ocgp_score(gp, x_unhealthy, y_unhealthy)

        y_true = np.concatenate([
            np.zeros_like(score_hold, dtype=int),
            np.ones_like(score_unhealthy, dtype=int),
        ])
        s_eval = np.concatenate([score_hold, score_unhealthy])
        y_pred = (s_eval > threshold).astype(int)

        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", pos_label=1
        )
        try:
            auroc = roc_auc_score(y_true, s_eval)
        except ValueError:
            auroc = np.nan
        try:
            auprc = average_precision_score(y_true, s_eval)
        except ValueError:
            auprc = np.nan

        if MAKE_PLOTS:
            plt.figure(figsize=(7, 5))
            plt.hist(score_hold, bins=30, alpha=0.6, label="Healthy (CAT-1 holdout)", density=True)
            plt.hist(score_unhealthy, bins=30, alpha=0.6, label="Unhealthy (CAT-3)", density=True)
            plt.axvline(threshold, linestyle="--", color="red", label=f"Threshold = {threshold:.3f}")
            plt.title(f"OCGP anomaly scores\nTarget: {target_col}")
            plt.xlabel("|z| anomaly score")
            plt.ylabel("Density")
            plt.legend(title="Class")
            if SAVE_PNGS:
                plt.tight_layout()
                plt.savefig(os.path.join(OUT_DIR, "figs", f"scores_{target_col}.png"), dpi=200)
            time.sleep(1)
            plt.show()
    else:
        acc = prec = rec = f1 = auroc = auprc = np.nan

    metrics = {
        "accuracy": float(acc) if acc == acc else np.nan,
        "precision_anom": float(prec) if prec == prec else np.nan,
        "recall_anom": float(rec) if rec == rec else np.nan,
        "f1_anom": float(f1) if f1 == f1 else np.nan,
        "auroc": float(auroc) if auroc == auroc else np.nan,
        "auprc": float(auprc) if auprc == auprc else np.nan,
        "n_healthy_train": int(len(tr_abs)),
        "n_healthy_holdout": int(len(ho_abs)),
        "n_unhealthy": int(len(unhealthy_idx)),
    }

    return TargetResult(
        target=target_col,
        threshold=threshold,
        metrics=metrics,
        kernel_params=extract_kernel_params(gp),
        columns_used_as_X=x_cols,
        holdout_idx=ho_abs,
        unhealthy_idx=unhealthy_idx,
        scores_holdout=score_hold,
        scores_unhealthy=score_unhealthy,
    )


# Plotting
def _to_1d(a):
    return np.asarray(a).ravel()


def _finite_xy(y, s):
    y = _to_1d(y).astype(int)
    s = _to_1d(s).astype(float)
    mask = np.isfinite(y) & np.isfinite(s)
    return y[mask], s[mask]


def plot_ensemble_curves(
    f_h,
    f_u,
    frac_h,
    frac_u,
    results,
    summary,
    outdir="./ocgp_results",
    fname="ensembles_features",
    n_features=5,
):
    os.makedirs(outdir, exist_ok=True)

    f_h, f_u, frac_h, frac_u = map(_to_1d, [f_h, f_u, frac_h, frac_u])

    y_fisher = np.concatenate([np.zeros_like(f_h, dtype=int), np.ones_like(f_u, dtype=int)])
    s_fisher = np.concatenate([f_h, f_u])

    y_frac = np.concatenate([np.zeros_like(frac_h, dtype=int), np.ones_like(frac_u, dtype=int)])
    s_frac = np.concatenate([frac_h, frac_u])

    y_fisher, s_fisher = _finite_xy(y_fisher, s_fisher)
    y_frac, s_frac = _finite_xy(y_frac, s_frac)

    pos_rate = float(y_fisher.mean()) if y_fisher.size else 0.0

    name2res = {r.target: r for r in results if r is not None}
    per_target = []
    for t in summary.sort_values("auroc", ascending=False)["target"].tolist():
        r = name2res.get(t)
        if r is None or r.scores_holdout is None or r.scores_unhealthy is None:
            continue

        y_t = np.concatenate([
            np.zeros_like(_to_1d(r.scores_holdout), dtype=int),
            np.ones_like(_to_1d(r.scores_unhealthy), dtype=int),
        ])
        s_t = np.concatenate([_to_1d(r.scores_holdout), _to_1d(r.scores_unhealthy)])
        y_t, s_t = _finite_xy(y_t, s_t)

        if s_t.size == 0 or np.allclose(s_t, s_t[0]):
            continue

        per_target.append((t, y_t, s_t))
        if len(per_target) == int(n_features):
            break

    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )

    okabe_ito = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
    c_ens_fish, c_ens_frac = okabe_ito[0], okabe_ito[1]
    feat_colors = [okabe_ito[2], okabe_ito[3], okabe_ito[4], okabe_ito[5]]
    c_baseline = "#9AA0A6"

    ls_ens_fish, ls_ens_frac = "-", "--"
    feat_styles = ["-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]
    lw_ens, lw_feat, lw_base = 3.0, 2.5, 1.5

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(7.2, 3.6))
    for ax in (ax_roc, ax_pr):
        ax.set_aspect(1.0, adjustable="box")

    fpr_f, tpr_f, _ = roc_curve(y_fisher, s_fisher)
    auc_f = auc(fpr_f, tpr_f)
    ax_roc.plot(fpr_f, tpr_f, color=c_ens_fish, lw=lw_ens, linestyle=ls_ens_fish)

    fpr_q, tpr_q, _ = roc_curve(y_frac, s_frac)
    auc_q = auc(fpr_q, tpr_q)
    ax_roc.plot(fpr_q, tpr_q, color=c_ens_frac, lw=lw_ens, linestyle=ls_ens_frac)

    feat_auc_rows = []
    for i, (tname, y_t, s_t) in enumerate(per_target):
        fpr_t, tpr_t, _ = roc_curve(y_t, s_t)
        auc_t = auc(fpr_t, tpr_t)
        ax_roc.plot(
            fpr_t,
            tpr_t,
            color=feat_colors[i % len(feat_colors)],
            lw=lw_feat,
            linestyle=feat_styles[i % len(feat_styles)],
        )
        feat_auc_rows.append((tname, auc_t))

    ax_roc.plot([0, 1], [0, 1], color=c_baseline, lw=lw_base, linestyle=":")
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC")

    roc_lines = [f"Fisher: AUC {auc_f:.3f}", f"Fraction: AUC {auc_q:.3f}"]
    roc_lines += [f"{t}: AUC {a:.3f}" for t, a in feat_auc_rows]
    at_roc = AnchoredText("\n".join(roc_lines), loc="lower right", prop=dict(size=8), frameon=True, borderpad=0.3)
    at_roc.patch.set_alpha(0.92)
    at_roc.patch.set_edgecolor("#DDDDDD")
    at_roc.patch.set_linewidth(0.7)
    ax_roc.add_artist(at_roc)

    prec_f, rec_f, _ = precision_recall_curve(y_fisher, s_fisher)
    ap_f = average_precision_score(y_fisher, s_fisher)
    ax_pr.plot(rec_f, prec_f, color=c_ens_fish, lw=lw_ens, linestyle=ls_ens_fish)

    prec_q, rec_q, _ = precision_recall_curve(y_frac, s_frac)
    ap_q = average_precision_score(y_frac, s_frac)
    ax_pr.plot(rec_q, prec_q, color=c_ens_frac, lw=lw_ens, linestyle=ls_ens_frac)

    feat_ap_rows = []
    for i, (tname, y_t, s_t) in enumerate(per_target):
        prec_t, rec_t, _ = precision_recall_curve(y_t, s_t)
        ap_t = average_precision_score(y_t, s_t)
        ax_pr.plot(
            rec_t,
            prec_t,
            color=feat_colors[i % len(feat_colors)],
            lw=lw_feat,
            linestyle=feat_styles[i % len(feat_styles)],
        )
        feat_ap_rows.append((tname, ap_t))

    ax_pr.hlines(y=pos_rate, xmin=0, xmax=1, color=c_baseline, lw=lw_base, linestyles=":")
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1.0)
    ax_pr.set_xlabel("Recall (unhealthy)")
    ax_pr.set_ylabel("Precision (unhealthy)")
    ax_pr.set_title("Precision–Recall")

    pr_lines = [f"Fisher: AP {ap_f:.3f}", f"Fraction: AP {ap_q:.3f}"]
    pr_lines += [f"{t}: AP {apv:.3f}" for t, apv in feat_ap_rows]
    pr_lines.append(f"No-skill: {pos_rate:.2f}")
    at_pr = AnchoredText("\n".join(pr_lines), loc="lower left", prop=dict(size=8), frameon=True, borderpad=0.3)
    at_pr.patch.set_alpha(0.92)
    at_pr.patch.set_edgecolor("#DDDDDD")
    at_pr.patch.set_linewidth(0.7)
    ax_pr.add_artist(at_pr)

    legend_handles = [
        Line2D([0], [0], color=c_ens_fish, lw=lw_ens, linestyle=ls_ens_fish, label="Ensemble (Fisher)"),
        Line2D([0], [0], color=c_ens_frac, lw=lw_ens, linestyle=ls_ens_frac, label="Ensemble (Fraction)"),
    ]
    for i, (tname, _, _) in enumerate(per_target):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=feat_colors[i % len(feat_colors)],
                lw=lw_feat,
                linestyle=feat_styles[i % len(feat_styles)],
                label=f"Feature: {tname}",
            )
        )
    legend_handles.append(Line2D([0], [0], color=c_baseline, lw=lw_base, linestyle=":", label="No-skill baselines"))

    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.28, wspace=0.3)

    fig.savefig(os.path.join(outdir, f"{fname}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"{fname}.png"), bbox_inches="tight")
    plt.show()


# OCSVM baseline
def _preprocess_all_features_leakfree(
    cleaned_df: pd.DataFrame,
    y_bin: np.ndarray,
    final_features: list[str],
    wins_cols: list[str],
    log1p_features: list[str],
    share_features: list[str],
    yj_cols: list[str],
    dropped_cols: set[str],
    holdout_frac: float = 0.25,
    random_state: int = 42,
):
    healthy_idx = np.where(y_bin == HEALTHY)[0]
    unhealthy_idx = np.where(y_bin == UNHEALTHY)[0]

    tr_idx_h, ho_idx_h = train_test_split(
        np.arange(len(healthy_idx)),
        test_size=holdout_frac,
        random_state=random_state,
        shuffle=True,
    )
    tr_abs = healthy_idx[tr_idx_h]
    ho_abs = healthy_idx[ho_idx_h]

    df_work = cleaned_df.copy()

    wins_eff = [c for c in wins_cols if c in df_work.columns]
    if wins_eff:
        df_train = df_work.iloc[tr_abs].copy()
        wins_th = {}
        for c in wins_eff:
            arr = df_train[c].to_numpy()
            lo = np.percentile(arr, 0.5)
            hi = np.percentile(arr, 99.5)
            wins_th[c] = (lo, hi)
        for c, (lo, hi) in wins_th.items():
            df_work[c] = np.clip(df_work[c].to_numpy(), lo, hi)

    if "lf_power" in df_work.columns and "hf_power" in df_work.columns:
        total = df_work["lf_power"] + df_work["hf_power"] + 1e-6
        df_work["lf_share"] = df_work["lf_power"] / total
        df_work["hf_share"] = 1.0 - df_work["lf_share"]

    drop_now = [c for c in dropped_cols if c in df_work.columns]
    if drop_now:
        df_work = df_work.drop(columns=drop_now, errors="ignore")

    for c in log1p_features:
        if c in df_work.columns:
            df_work[c] = np.log1p(np.maximum(df_work[c].to_numpy(), 0))

    for c in share_features:
        if c in df_work.columns:
            x = np.clip(df_work[c].to_numpy(), 1e-6, 1 - 1e-6)
            df_work[c] = np.log(x / (1 - x))

    yj_eff = [c for c in yj_cols if c in df_work.columns]
    yj_model = None
    if yj_eff:
        df_train = df_work.iloc[tr_abs].copy()
        do_yj = df_train[yj_eff].nunique().min() > 1
        if do_yj:
            yj_model = PowerTransformer(method="yeo-johnson", standardize=False)
            yj_model.fit(df_train[yj_eff].to_numpy())
            df_work.loc[:, yj_eff] = yj_model.transform(df_work[yj_eff].to_numpy())

    cols_present = [c for c in final_features if c in df_work.columns]
    x_all = df_work[cols_present].to_numpy()

    x_train_raw = x_all[tr_abs]
    x_hold_raw = x_all[ho_abs]
    x_unhealthy_raw = x_all[unhealthy_idx] if len(unhealthy_idx) else None

    scaler = RobustScaler()
    scaler.fit(x_train_raw)

    x_train = scaler.transform(x_train_raw)
    x_hold = scaler.transform(x_hold_raw)
    x_unhealthy = scaler.transform(x_unhealthy_raw) if x_unhealthy_raw is not None else None

    return {
        "X_train": x_train,
        "X_hold": x_hold,
        "X_unhealthy": x_unhealthy,
        "tr_abs": tr_abs,
        "ho_abs": ho_abs,
        "unhealthy_idx": unhealthy_idx,
        "cols_present": cols_present,
        "wins_eff": wins_eff,
        "yj_eff": yj_eff,
        "scaler": scaler,
        "yj_model": yj_model,
    }


def run_ocsvm_baseline(
    cleaned_df: pd.DataFrame,
    y_bin: np.ndarray,
    final_features: list[str],
    wins_cols: list[str],
    log1p_features: list[str],
    share_features: list[str],
    yj_cols: list[str],
    dropped_cols: set[str],
    holdout_frac: float = 0.25,
    q: float = 0.95,
    random_state: int = 42,
    nu: float = 0.05,
    gamma: str | float = "scale",
):
    pack = _preprocess_all_features_leakfree(
        cleaned_df=cleaned_df,
        y_bin=y_bin,
        final_features=final_features,
        wins_cols=wins_cols,
        log1p_features=log1p_features,
        share_features=share_features,
        yj_cols=yj_cols,
        dropped_cols=dropped_cols,
        holdout_frac=holdout_frac,
        random_state=random_state,
    )

    x_train = pack["X_train"]
    x_hold = pack["X_hold"]
    x_unhealthy = pack["X_unhealthy"]

    if x_unhealthy is None:
        raise ValueError("No unhealthy samples found (CAT-3).")

    ocsvm = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
    ocsvm.fit(x_train)

    score_hold = -ocsvm.decision_function(x_hold).ravel()
    score_unhealthy = -ocsvm.decision_function(x_unhealthy).ravel()

    threshold = float(np.nanquantile(score_hold, q))

    y_true = np.concatenate([np.zeros_like(score_hold, dtype=int), np.ones_like(score_unhealthy, dtype=int)])
    s_eval = np.concatenate([score_hold, score_unhealthy])
    y_pred = (s_eval > threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1)
    auroc = roc_auc_score(y_true, s_eval)
    auprc = average_precision_score(y_true, s_eval)

    return {
        "method": "OCSVM",
        "nu": float(nu),
        "gamma": gamma,
        "q": float(q),
        "threshold": threshold,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "n_healthy_train": int(x_train.shape[0]),
        "n_healthy_holdout": int(x_hold.shape[0]),
        "n_unhealthy": int(x_unhealthy.shape[0]),
        "features_used": pack["cols_present"],
    }


if __name__ == "__main__":
    print("Loading:", DATA_PATH)
    data = pd.read_excel(DATA_PATH)

    cleaned = data.dropna(subset=list(set(FEATURE_COLUMNS + [LABEL_COL]))).copy()
    cleaned["_CAT_NORM_"] = normalize_labels(cleaned[LABEL_COL])

    print("Label distribution (normalized):")
    print(cleaned["_CAT_NORM_"].value_counts(dropna=False).to_string())

    healthy_mask = cleaned["_CAT_NORM_"] == HEALTHY_LABEL
    unhealthy_mask = cleaned["_CAT_NORM_"] == UNHEALTHY_LABEL
    n_h, n_u = int(healthy_mask.sum()), int(unhealthy_mask.sum())
    print(f"Detected healthy (CAT-1): {n_h}, unhealthy (CAT-3): {n_u}")

    if n_h == 0:
        raise ValueError("No CAT-1 rows found. Check label values.")
    if n_u == 0:
        warnings.warn("No CAT-3 rows found, so anomaly metrics will be undefined.", RuntimeWarning)

    y_bin = np.full(len(cleaned), np.nan, dtype=float)
    y_bin[healthy_mask.to_numpy()] = HEALTHY
    y_bin[unhealthy_mask.to_numpy()] = UNHEALTHY

    print(f"Using {len(FINAL_FEATURES)} final features")
    print(sorted(FINAL_FEATURES))

    healthy_idx_global = np.where(y_bin == HEALTHY)[0]

    tr_pos, ho_pos = train_test_split(
        np.arange(len(healthy_idx_global)),
        test_size=0.25,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    tr_abs_global = healthy_idx_global[tr_pos]
    ho_abs_global = healthy_idx_global[ho_pos]

    if MAKE_PLOTS and CORR_PLOT:
        try:
            df_work = cleaned.copy()

            wins_eff = [c for c in WINSOR_COLUMNS if c in df_work.columns]
            if wins_eff:
                df_train = df_work.iloc[tr_abs_global].copy()
                wins_th_train = compute_winsor_thresholds(
                    df_train[wins_eff], wins_eff, lower=0.5, upper=99.5
                )
                df_work = apply_winsorization(df_work, wins_th_train)

            df_work = add_engineered_features(df_work)
            df_work = apply_log1p(df_work, [c for c in LOG1P_FEATURES if c in df_work.columns])
            df_work = apply_logit_transform(df_work, [c for c in SHARE_FEATURES if c in df_work.columns])

            if all(c in df_work.columns for c in YJ_COLUMNS):
                df_train = df_work.iloc[tr_abs_global].copy()
                do_yj = df_train[YJ_COLUMNS].nunique().min() > 1
                if do_yj:
                    yj_local = PowerTransformer(method="yeo-johnson", standardize=False)
                    yj_local.fit(df_train[YJ_COLUMNS].to_numpy())
                    df_work.loc[:, YJ_COLUMNS] = yj_local.transform(df_work[YJ_COLUMNS].to_numpy())

            cols_present = [c for c in FINAL_FEATURES if c in df_work.columns]
            all_features_df = df_work[cols_present].copy()

            scaler_corr = RobustScaler()
            scaler_corr.fit(all_features_df.iloc[tr_abs_global].to_numpy())
            x_corr = scaler_corr.transform(all_features_df.to_numpy())
            corr_src = pd.DataFrame(x_corr, columns=cols_present, index=df_work.index)

            plt.figure(figsize=(max(8, 0.5 * len(cols_present)), max(6, 0.5 * len(cols_present))))
            sns.heatmap(corr_src.corr(), cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": 0.7})
            plt.title("Feature correlation (transforms fit on CAT-1 train)")
            if SAVE_PNGS:
                plt.tight_layout()
                plt.savefig(os.path.join(OUT_DIR, "figs", "corr_leakfree.png"), dpi=200)
            time.sleep(1)
            plt.show()
        except Exception as e:
            warnings.warn(f"Correlation plot skipped: {e}")

    targets = list(FINAL_FEATURES)
    print(f"Fitting GP models for {len(targets)} target features...")

    def fit_target_safely(target_name):
        try:
            return fit_one_target(
                cleaned_df=cleaned,
                target_col=target_name,
                y_bin=y_bin,
                tr_abs=tr_abs_global,
                ho_abs=ho_abs_global,
                healthy_quantile=0.95,
                restarts=2,
            )
        except Exception as e:
            warnings.warn(f"Skipping {target_name}: {e}")
            return None

    results_raw = Parallel(n_jobs=1, prefer="threads")(
        delayed(fit_target_safely)(t) for t in targets
    )
    results = [r for r in results_raw if r is not None]

    rows = [{"target": r.target, **r.metrics} for r in results]
    if len(rows) == 0:
        print("\nNo targets were successfully fit. See warnings above.")
        summary = pd.DataFrame()
    else:
        summary = pd.DataFrame(rows)
        if "auroc" in summary.columns:
            summary = summary.sort_values("auroc", ascending=False)
        print("\nPer-feature results:")
        print(summary.to_string(index=False))

    summary_path = os.path.join(OUT_DIR, "ocgp_summary.csv")
    summary.to_csv(summary_path, index=False)

    thr_df = pd.DataFrame([{"target": r.target, "threshold": r.threshold} for r in results])
    thr_df.to_csv(os.path.join(OUT_DIR, "ocgp_thresholds.csv"), index=False)

    hp_rows = []
    for r in results:
        row = {"target": r.target, "noise_level": np.nan, "const": np.nan}
        ls = r.kernel_params.get("length_scales", [])
        for i, val in enumerate(ls):
            row[f"ls_{i}"] = val
        for k in ("noise_level", "const"):
            if k in r.kernel_params:
                row[k] = r.kernel_params[k]
        hp_rows.append(row)

    hp_df = pd.DataFrame(hp_rows).sort_values("target") if len(hp_rows) else pd.DataFrame(columns=["target"])
    hp_df.to_csv(os.path.join(OUT_DIR, "ocgp_kernel_hyperparams.csv"), index=False)

    print("\nTop-10 by AUROC:")
    if not summary.empty:
        print(summary.head(10).to_string(index=False))

    cfg = {
        "file": DATA_PATH,
        "label_col": LABEL_COL,
        "healthy_label": HEALTHY_LABEL,
        "unhealthy_label": UNHEALTHY_LABEL,
        "winsor_cols": WINSOR_COLUMNS,
        "dropped_cols": list(DROPPED_COLUMNS),
        "log1p_features": LOG1P_FEATURES,
        "share_features": SHARE_FEATURES,
        "yj_cols": YJ_COLUMNS,
        "final_features": FINAL_FEATURES,
        "random_state": RANDOM_STATE,
    }
    with open(os.path.join(OUT_DIR, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print("\nSaved results to:")
    print(summary_path)
    print(os.path.join(OUT_DIR, "ocgp_thresholds.csv"))
    print(os.path.join(OUT_DIR, "ocgp_kernel_hyperparams.csv"))
    print(os.path.join(OUT_DIR, "run_config.json"))

    valid = [
        r
        for r in results
        if (
            r is not None
            and r.scores_holdout is not None
            and r.scores_unhealthy is not None
            and r.threshold == r.threshold
        )
    ]

    if len(valid) == 0:
        print("No valid target results to ensemble.")
    else:
        s_h = np.vstack([r.scores_holdout for r in valid]).T
        s_u = np.vstack([r.scores_unhealthy for r in valid]).T

        thr_vec = np.array([float(r.threshold) for r in valid])[None, :]
        e_h = (s_h > thr_vec).astype(float)
        e_u = (s_u > thr_vec).astype(float)

        frac_h = e_h.mean(axis=1)
        frac_u = e_u.mean(axis=1)

        thr_vote = float(np.quantile(frac_h, 0.95))

        y_true = np.concatenate([np.zeros_like(frac_h, dtype=int), np.ones_like(frac_u, dtype=int)])
        s_eval = np.concatenate([frac_h, frac_u])
        y_pred_anom = (s_eval >= thr_vote).astype(int)

        acc_vote = accuracy_score(y_true, y_pred_anom)
        prec_vote, rec_vote, f1_vote, _ = precision_recall_fscore_support(
            y_true, y_pred_anom, average="binary", pos_label=1
        )
        auroc_vote = roc_auc_score(y_true, s_eval)
        auprc_vote = average_precision_score(y_true, s_eval)

        print("\n[Ensemble: Fraction-flagged]")
        print(
            f"Cutoff (95% CAT-1): {thr_vote:.3f} | "
            f"ACC {acc_vote:.3f} | P {prec_vote:.3f} | R {rec_vote:.3f} | F1 {f1_vote:.3f} | "
            f"AUROC {auroc_vote:.3f} | AUPRC {auprc_vote:.3f}"
        )

        def fisher_scores(s):
            p = 2.0 * (1.0 - norm.cdf(s))
            p = np.clip(p, 1e-12, 1.0)
            return -2.0 * np.sum(np.log(p), axis=1)

        f_h = fisher_scores(s_h)
        f_u = fisher_scores(s_u)
        thr_fisher = float(np.quantile(f_h, 0.95))

        y_true = np.concatenate([np.zeros_like(f_h, dtype=int), np.ones_like(f_u, dtype=int)])
        s_eval = np.concatenate([f_h, f_u])
        y_pred_anom = (s_eval >= thr_fisher).astype(int)

        acc_f = accuracy_score(y_true, y_pred_anom)
        prec_f, rec_f, f1_f, _ = precision_recall_fscore_support(
            y_true, y_pred_anom, average="binary", pos_label=1
        )
        auroc_f = roc_auc_score(y_true, s_eval)
        auprc_f = average_precision_score(y_true, s_eval)

        print("\n[Ensemble: Fisher p-combo]")
        print(
            f"Cutoff (95% CAT-1): {thr_fisher:.3f} | "
            f"ACC {acc_f:.3f} | P {prec_f:.3f} | R {rec_f:.3f} | F1 {f1_f:.3f} | "
            f"AUROC {auroc_f:.3f} | AUPRC {auprc_f:.3f}"
        )

        plot_ensemble_curves(f_h, f_u, frac_h, frac_u, results, summary, n_features=5)

    if MAKE_PLOTS and (not summary.empty):
        plt.figure(figsize=(9.5, max(4, 0.28 * len(summary))))
        s_plot = summary.sort_values("auroc", ascending=False).copy()
        sns.barplot(data=s_plot, x="auroc", y="target")
        plt.axvline(0.5, linestyle="--", linewidth=1)
        plt.title("Per-target AUROC (higher = better)")
        plt.xlabel("AUROC")
        plt.ylabel("Feature")
        if SAVE_PNGS:
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, "figs", "bar_auroc_sorted.png"), dpi=200)
        time.sleep(1)
        plt.show()

        plt.figure(figsize=(7.2, 5.4))
        s_sc = summary.copy()
        sns.scatterplot(data=s_sc, x="auroc", y="recall_anom", alpha=0.9)
        plt.title("Per-target AUROC vs. anomaly recall")
        plt.xlabel("AUROC")
        plt.ylabel("Recall (unhealthy)")
        for _, row in s_sc.sort_values("auroc", ascending=False).head(8).iterrows():
            if pd.notna(row["auroc"]) and pd.notna(row["recall_anom"]):
                plt.text(row["auroc"] + 0.005, row["recall_anom"] + 0.005, row["target"], fontsize=8)
        if SAVE_PNGS:
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, "figs", "scatter_auroc_vs_recall.png"), dpi=200)
        time.sleep(1)
        plt.show()

    if n_u > 0:
        ocsvm_res = run_ocsvm_baseline(
            cleaned_df=cleaned,
            y_bin=y_bin,
            final_features=FINAL_FEATURES,
            wins_cols=WINSOR_COLUMNS,
            log1p_features=LOG1P_FEATURES,
            share_features=SHARE_FEATURES,
            yj_cols=YJ_COLUMNS,
            dropped_cols=DROPPED_COLUMNS,
            holdout_frac=0.25,
            q=0.95,
            random_state=RANDOM_STATE,
            nu=0.05,
            gamma="scale",
        )

        print("\n[Baseline: One-Class SVM]")
        print(
            f"nu={ocsvm_res['nu']}, gamma={ocsvm_res['gamma']} | "
            f"Cutoff (q={ocsvm_res['q']:.2f} healthy holdout): {ocsvm_res['threshold']:.3f} | "
            f"ACC {ocsvm_res['accuracy']:.3f} | P {ocsvm_res['precision']:.3f} | "
            f"R {ocsvm_res['recall']:.3f} | F1 {ocsvm_res['f1']:.3f} | "
            f"AUROC {ocsvm_res['auroc']:.3f} | AUPRC {ocsvm_res['auprc']:.3f}"
        )
    else:
        print("\nSkipping OCSVM baseline because no CAT-3 rows were found.")
