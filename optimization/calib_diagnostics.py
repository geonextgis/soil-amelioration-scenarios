#!/usr/bin/env python
"""Diagnostics the calibration agents reason with.

A single RMSE tells an agent *that* a parameter set is wrong, not *which*
parameter is wrong. This module turns a finished run into the evidence needed to
attribute the error:

**Phenology** — the flowering error and the anthesis-to-maturity *duration*
error are separated, because that is exactly the TSUM1/TSUM2 split; the residual
is then tested against season warmth (is it a thermal-response problem?) and
against latitude (is it a photoperiod problem?).

**LAI** — the curve is decomposed into the features the parameters actually
control: the emergence level (TDWI), the rate of increase (RGRLAI), the peak and
its timing (SLA, LAICR), how long the plateau lasts (LAICR/RDRSHM), how fast the
canopy declines (DVSDLT, RDRLeaves) and where the bias sits along DVS (the SLA
table element by element). Plus the temporal residual structure, which
distinguishes a uniform level error from a phase error.

**Yield** — the error is decomposed along the identity ``yield = AGBiomass x HI``
and correlated with the simulated stress indicators (maxLAI, TRANRF, NNI), which
is what separates "the crop did not make enough biomass" (RUE / KDIF / LAI) from
"the crop made the biomass and did not put it in the grain" (partitioning,
translocation, YieldAdjustRatio).

Observed and simulated series are always compared **on the matched dates only**,
so a sparse 8-day GLASS retrieval is never compared against a daily simulated
statistic it cannot support. Simulated-only statistics from the full daily curve
are reported separately and labelled ``sim_full_*``.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

warnings.filterwarnings("ignore")

# Same identity as evaluate_run.ipynb: observed = blue, simulated = orange.
OBS, SIM = "#2a78d6", "#eb6834"
INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#8a8880", "#fcfcfb", "#e6e5e1"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 10.5, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "axes.titlepad": 9, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.frameon": False,
    "legend.fontsize": 8.5, "figure.dpi": 110,
})

DVS_BIN_LABELS = ["0.00-0.25 emergence", "0.25-0.50 tillering", "0.50-1.00 stem elong.",
                  "1.00-1.25 anthesis", "1.25-1.50 filling early", "1.50-1.75 filling mid",
                  "1.75-2.00 filling late"]


# ---------------------------------------------------------------------------
# Generic metrics
# ---------------------------------------------------------------------------
def metrics(obs, sim) -> dict:
    """RMSE, MAE, bias, R2, nRMSE%, Nash-Sutcliffe EF. Same definitions as the notebook."""
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    ok = np.isfinite(obs) & np.isfinite(sim)
    obs, sim = obs[ok], sim[ok]
    if len(obs) < 2:
        return {"n": int(len(obs))}
    err = sim - obs
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    r = np.corrcoef(obs, sim)[0, 1] if obs.std() > 0 and sim.std() > 0 else np.nan
    return {
        "n": int(len(obs)),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "bias": float(err.mean()),
        "R2": float(r ** 2) if np.isfinite(r) else None,
        "nRMSE_pct": float(100 * np.sqrt(np.mean(err ** 2)) / obs.mean()) if obs.mean() else None,
        "EF": float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else None,
    }


def _slope(x, y) -> float | None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0:
        return None
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def _round(obj, nd: int = 4):
    """Recursively round for compact, diff-able JSON in the ledger."""
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v, nd) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else round(float(obj), nd)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


# ===========================================================================
# Phenology
# ===========================================================================
# A raw maturity-date error is not attributable on its own: maturity is dated
# *from* anthesis, so a TSUM1 error propagates into it unchanged. What separates
# the two parameters is the duration.
#
#     flowering residual         -> TSUM1  (or TSUMEM/TBASEM, see below)
#     duration residual          -> TSUM2
#     residual x season warmth   -> TEFFMX / TsumIncrementTableRate
#     residual x latitude        -> PhotoperiodTableFactor
#
# TSUM1 and TSUMEM produce the same signature on flowering alone. What tells them
# apart is that a TSUMEM/TBASEM error shifts emergence and therefore *everything*
# by a constant, while TSUM1 also depends on the emergence-to-anthesis weather —
# so a TSUMEM error is the flatter, more uniform of the two across years.

# Below this the residual is inside the reporting resolution of the observations
# and is not worth attributing to a parameter.
PHENOLOGY_TOLERANCE_DAYS = 2.0


def _bias_table(df: pd.DataFrame, by: str) -> list[dict]:
    """Median flowering / maturity / duration residual within each group."""
    rows = []
    for key, g in df.groupby(by, observed=True):
        rows.append({
            by: (int(key) if isinstance(key, (int, np.integer)) else key),
            "n": int(len(g)),
            "flowering_bias": float((g["flowering_doy_sim"] - g["flowering_doy"]).median()),
            "maturity_bias": float((g["maturity_doy_sim"] - g["maturity_doy"]).median()),
            "duration_bias": float((g["duration_sim"] - g["duration_obs"]).median()),
        })
    return sorted(rows, key=lambda r: r[by])


def _phenology_verdict(flowering_bias: float, duration_bias: float,
                       warmth_slope: float | None, latitude_slope: float | None) -> str:
    """One sentence naming the parameter the residual structure implicates."""
    tol = PHENOLOGY_TOLERANCE_DAYS
    parts = []
    if abs(flowering_bias) > tol:
        way = "late" if flowering_bias > 0 else "early"
        move = "raise" if flowering_bias < 0 else "lower"
        parts.append(f"flowering is {abs(flowering_bias):.1f} d too {way} -> {move} TSUM1 "
                     f"(TSUMEM/TBASEM if the offset is equally flat across all years)")
    if abs(duration_bias) > tol:
        way = "long" if duration_bias > 0 else "short"
        move = "lower" if duration_bias > 0 else "raise"
        parts.append(f"the anthesis-to-maturity duration is {abs(duration_bias):.1f} d too "
                     f"{way} -> {move} TSUM2")
    # Slopes are in days of residual per day of observed flowering DOY / per degree
    # of latitude. Anything this steep is a structure no single thermal-time
    # constant can remove.
    if warmth_slope is not None and abs(warmth_slope) > 0.25:
        parts.append(f"the residual scales with season warmth (slope {warmth_slope:+.2f} "
                     f"d/d) -> TEFFMX or TsumIncrementTableRate")
    if latitude_slope is not None and abs(latitude_slope) > 1.0:
        parts.append(f"the residual has a north-south gradient ({latitude_slope:+.2f} d/deg) "
                     f"-> PhotoperiodTableFactor, if IDSL enables it")
    if not parts:
        return (f"flowering and duration are both within {tol:.0f} d — the optimized "
                f"phenology reproduces the observations; no parameter change is indicated")
    return "; ".join(parts)


def phenology_diagnostics(df: pd.DataFrame, points: pd.DataFrame | None = None) -> dict:
    """Attribute a phenology error to flowering timing, duration, warmth or latitude.

    ``points`` (PointID, Latitude) is optional; without it the photoperiod
    signature cannot be computed and is reported as null rather than guessed.
    """
    df = df.copy()
    df["duration_obs"] = df["maturity_doy"] - df["flowering_doy"]
    df["duration_sim"] = df["maturity_doy_sim"] - df["flowering_doy_sim"]
    flowering_res = df["flowering_doy_sim"] - df["flowering_doy"]
    duration_res = df["duration_sim"] - df["duration_obs"]

    # Observed flowering DOY as the warmth proxy: an early observed anthesis is a
    # warm season. This avoids re-reading the weather files for a signal that is
    # already in the observations.
    warmth_slope = _slope(df["flowering_doy"], flowering_res)

    latitude_slope = None
    by_latitude = None
    if points is not None and not points.empty:
        joined = df.merge(points, on="PointID", how="inner")
        if len(joined) > 10 and "Latitude" in joined:
            res = joined["flowering_doy_sim"] - joined["flowering_doy"]
            latitude_slope = _slope(joined["Latitude"], res)
            bands = pd.cut(joined["Latitude"], bins=4)
            by_latitude = [
                {"latitude_band": str(k), "n": int(len(g)),
                 "flowering_bias": float((g["flowering_doy_sim"] - g["flowering_doy"]).median())}
                for k, g in joined.groupby(bands, observed=True)
            ]

    by_year = _bias_table(df, "harvest_year")
    per_location = df.groupby("PointID").apply(
        lambda g: float((g["flowering_doy_sim"] - g["flowering_doy"]).median()),
        include_groups=False)

    out = {
        "n_pairs": int(len(df)),
        "n_locations": int(df["PointID"].nunique()),
        "years": [int(df["harvest_year"].min()), int(df["harvest_year"].max())],
        "excluded_autumn_bolting": int(df.attrs.get("n_bolting", 0)),
        "flowering": metrics(df["flowering_doy"], df["flowering_doy_sim"]),
        "maturity": metrics(df["maturity_doy"], df["maturity_doy_sim"]),
        "duration": metrics(df["duration_obs"], df["duration_sim"]),
        "residual_structure": {
            "flowering_bias_median": float(flowering_res.median()),
            "duration_bias_median": float(duration_res.median()),
            # Spread across years: a large spread with a small median means the
            # level is right and the *response* is wrong.
            "flowering_bias_iqr": float(flowering_res.quantile(0.75) - flowering_res.quantile(0.25)),
            "duration_bias_iqr": float(duration_res.quantile(0.75) - duration_res.quantile(0.25)),
            "vs_season_warmth_slope": warmth_slope,
            "vs_latitude_slope": latitude_slope,
            "vs_year_slope": _slope(df["harvest_year"], flowering_res),
            "worst_locations": [
                {"PointID": int(k), "flowering_bias": float(per_location[k])}
                for k in per_location.abs().nlargest(8).index
            ],
        },
        "by_year": by_year,
        "by_latitude": by_latitude,
    }
    out["attribution"] = {
        "verdict": _phenology_verdict(
            out["residual_structure"]["flowering_bias_median"],
            out["residual_structure"]["duration_bias_median"],
            warmth_slope, latitude_slope),
        "tolerance_days": PHENOLOGY_TOLERANCE_DAYS,
    }
    return _round(out)


def phenology_plots(df: pd.DataFrame, out_dir: Path, title: str,
                    points: pd.DataFrame | None = None) -> list[str]:
    """Two figures: agreement (flowering / maturity / duration) and residual structure."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["duration_obs"] = df["maturity_doy"] - df["flowering_doy"]
    df["duration_sim"] = df["maturity_doy_sim"] - df["flowering_doy_sim"]
    written = []

    # --- agreement -----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    panels = [("flowering DOY", "flowering_doy", "flowering_doy_sim"),
              ("maturity DOY", "maturity_doy", "maturity_doy_sim"),
              ("anthesis->maturity (days)", "duration_obs", "duration_sim")]
    for ax, (label, ocol, scol) in zip(axes, panels):
        ax.scatter(df[ocol], df[scol], s=9, alpha=0.35, color=SIM, edgecolor="none")
        lo = float(min(df[ocol].min(), df[scol].min()))
        hi = float(max(df[ocol].max(), df[scol].max()))
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=1.0, ls="--")
        m = metrics(df[ocol], df[scol])
        ax.set_title(f"{label}\nRMSE {m.get('RMSE', float('nan')):.1f} d   "
                     f"bias {m.get('bias', float('nan')):+.1f} d   n={m.get('n', 0):,}")
        ax.set_xlabel("observed")
        ax.set_ylabel("simulated")
    fig.suptitle(title, x=0.008, ha="left", fontsize=11.5, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = out_dir / "phenology_agreement.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    written.append(str(path))

    # --- residual structure --------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    flowering_res = df["flowering_doy_sim"] - df["flowering_doy"]
    duration_res = df["duration_sim"] - df["duration_obs"]

    by_year = df.assign(fr=flowering_res, dr=duration_res).groupby("harvest_year")
    years = sorted(by_year.groups)
    axes[0].plot(years, [by_year.get_group(y)["fr"].median() for y in years],
                 marker="o", ms=3.5, color=SIM, label="flowering")
    axes[0].plot(years, [by_year.get_group(y)["dr"].median() for y in years],
                 marker="s", ms=3.5, color=OBS, label="duration")
    axes[0].axhline(0, color=INK2, lw=1.0, ls="--")
    axes[0].set_title("residual by year")
    axes[0].set_xlabel("harvest year")
    axes[0].set_ylabel("simulated - observed (days)")
    axes[0].legend()

    # Observed flowering DOY as the warmth proxy — early anthesis = warm season.
    axes[1].scatter(df["flowering_doy"], flowering_res, s=9, alpha=0.3,
                    color=SIM, edgecolor="none")
    slope = _slope(df["flowering_doy"], flowering_res)
    axes[1].axhline(0, color=INK2, lw=1.0, ls="--")
    axes[1].set_title("flowering residual vs season warmth\n"
                      + (f"slope {slope:+.2f} d/d (early obs = warm year)"
                         if slope is not None else "slope undefined"))
    axes[1].set_xlabel("observed flowering DOY")
    axes[1].set_ylabel("simulated - observed (days)")

    if points is not None and not points.empty and "Latitude" in points:
        joined = df.merge(points, on="PointID", how="inner")
        res = joined["flowering_doy_sim"] - joined["flowering_doy"]
        slope = _slope(joined["Latitude"], res)
        axes[2].scatter(joined["Latitude"], res, s=9, alpha=0.3, color=SIM, edgecolor="none")
        axes[2].axhline(0, color=INK2, lw=1.0, ls="--")
        axes[2].set_title("flowering residual vs latitude\n"
                          + (f"slope {slope:+.2f} d/deg (photoperiod signature)"
                             if slope is not None else "slope undefined"))
        axes[2].set_xlabel("latitude (deg N)")
        axes[2].set_ylabel("simulated - observed (days)")
    else:
        axes[2].hist(flowering_res.dropna(), bins=40, color=SIM, alpha=0.75)
        axes[2].axvline(0, color=INK2, lw=1.0, ls="--")
        axes[2].set_title("flowering residual distribution\n(no point coordinates available)")
        axes[2].set_xlabel("simulated - observed (days)")
        axes[2].set_ylabel("location-years")

    fig.suptitle(title, x=0.008, ha="left", fontsize=11.5, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = out_dir / "phenology_residuals.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    written.append(str(path))

    return written


def load_points(run) -> pd.DataFrame | None:
    """PointID -> latitude, for the photoperiod signature. None if unavailable."""
    path = run.repo_root / "data" / "raw" / "point_to_nearest_grid.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, usecols=["PointID", "Latitude", "Longitude"])


# ===========================================================================
# LAI
# ===========================================================================
def lai_bin_table(pairs: pd.DataFrame, bins: list[float]) -> pd.DataFrame:
    """Observed vs simulated LAI per development-stage bin.

    Element *i* of ``SLATableSLA`` sits on ``SLATableDVS[i]``, so a bin's bias
    points straight at the SLA element to move — this table is the main
    parameter-attribution device for LAI.
    """
    df = pairs.assign(dvs_bin=pd.cut(pairs["dvs"], bins=bins, include_lowest=True))
    out = (df.groupby("dvs_bin", observed=True)
             .apply(lambda g: pd.Series({
                 "n": len(g),
                 "obs_mean": g["LAI"].mean(),
                 "sim_mean": g["LAI_sim"].mean(),
                 "bias": (g["LAI_sim"] - g["LAI"]).mean(),
                 "rmse": float(np.sqrt(np.mean((g["LAI_sim"] - g["LAI"]) ** 2))),
             }), include_groups=False)
             .reset_index())
    out["dvs_bin"] = out["dvs_bin"].astype(str)
    out["rel_bias_pct"] = 100 * out["bias"] / out["obs_mean"].replace(0, np.nan)
    return out


def _season_shape(group: pd.DataFrame) -> dict | None:
    """Curve-shape features of one location-season, from the matched pairs only."""
    g = group.sort_values("date")
    if len(g) < 5:
        return None
    doy = g["date"].dt.dayofyear.to_numpy(float)
    # Winter crops span the new year: make the day axis monotone before fitting
    # slopes, or the January wrap turns a rising canopy into a negative slope.
    day = doy.copy()
    for i in range(1, len(day)):
        if day[i] < day[i - 1]:
            day[i:] += 365
            break

    obs, sim = g["LAI"].to_numpy(float), g["LAI_sim"].to_numpy(float)
    dvs = g["dvs"].to_numpy(float)
    i_obs, i_sim = int(np.argmax(obs)), int(np.argmax(sim))

    def rise(values):
        m = (dvs >= 0.2) & (dvs <= 1.0)
        return _slope(day[m], values[m]) if m.sum() >= 3 else None

    def decline(values, peak_i):
        m = np.arange(len(values)) > peak_i
        return _slope(day[m], values[m]) if m.sum() >= 3 else None

    def plateau(values, peak_i):
        """Days between the first and last matched point at >= 80% of the peak."""
        above = np.flatnonzero(values >= 0.8 * values[peak_i])
        return float(day[above[-1]] - day[above[0]]) if len(above) >= 2 else 0.0

    early = dvs <= 0.3
    return {
        "location": int(g["location"].iloc[0]), "year": int(g["year"].iloc[0]), "n": len(g),
        "obs_peak": float(obs[i_obs]), "sim_peak": float(sim[i_sim]),
        "obs_peak_doy": float(doy[i_obs]), "sim_peak_doy": float(doy[i_sim]),
        "obs_peak_dvs": float(dvs[i_obs]), "sim_peak_dvs": float(dvs[i_sim]),
        "obs_early": float(obs[early].mean()) if early.any() else None,
        "sim_early": float(sim[early].mean()) if early.any() else None,
        "obs_rise": rise(obs), "sim_rise": rise(sim),
        "obs_decline": decline(obs, i_obs), "sim_decline": decline(sim, i_sim),
        "obs_plateau_days": plateau(obs, i_obs), "sim_plateau_days": plateau(sim, i_sim),
        "bias": float(np.mean(sim - obs)),
        "residual_sign_consistent": bool(np.all(sim - obs > 0) or np.all(sim - obs < 0)),
    }


def lai_shape(pairs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per-season curve features and their cross-season summary."""
    rows = [s for _, g in pairs.groupby(["location", "year"], sort=False)
            if (s := _season_shape(g)) is not None]
    seasons = pd.DataFrame(rows)
    if seasons.empty:
        return seasons, {"n_seasons": 0,
                         "note": "no location-season had >= 5 matched observations"}

    def pair(name, unit=""):
        o, s = seasons[f"obs_{name}"].astype(float), seasons[f"sim_{name}"].astype(float)
        ok = o.notna() & s.notna()
        if ok.sum() == 0:
            return {"n": 0}
        d = (s - o)[ok]
        return {"obs_median": float(o[ok].median()), "sim_median": float(s[ok].median()),
                "median_diff": float(d.median()),
                "iqr_diff": [float(d.quantile(0.25)), float(d.quantile(0.75))],
                "n": int(ok.sum()), "unit": unit}

    summary = {
        "n_seasons": int(len(seasons)),
        "peak_lai": pair("peak", "m2/m2"),
        "peak_timing": pair("peak_doy", "DOY"),
        "peak_stage": pair("peak_dvs", "DVS"),
        "early_canopy": pair("early", "m2/m2 (DVS<=0.3)"),
        "rise_rate": pair("rise", "LAI/day"),
        "decline_rate": pair("decline", "LAI/day"),
        "plateau_duration": pair("plateau_days", "days at >=80% of peak"),
        "seasons_with_sign_consistent_residual_pct":
            float(100 * seasons["residual_sign_consistent"].mean()),
        "season_bias_median": float(seasons["bias"].median()),
        "season_bias_iqr": [float(seasons["bias"].quantile(0.25)),
                            float(seasons["bias"].quantile(0.75))],
    }
    return seasons, summary


def lai_residual_structure(pairs: pd.DataFrame) -> dict:
    """Is the error a level offset, a phase error, or stress-driven?"""
    res = pairs["LAI_sim"] - pairs["LAI"]
    by_month = (pairs.assign(month=pairs["date"].dt.month, res=res)
                .groupby("month")["res"].agg(["mean", "count"]))
    by_loc = (pairs.assign(res=res).groupby("location")["res"].mean())
    return {
        "mean_residual": float(res.mean()),
        "residual_sd": float(res.std()),
        "residual_by_month": {int(m): {"mean": float(r["mean"]), "n": int(r["count"])}
                              for m, r in by_month.iterrows()},
        "residual_vs_dvs_slope": _slope(pairs["dvs"], res),
        "residual_vs_obs_slope": _slope(pairs["LAI"], res),
        "location_bias_spread_iqr": [float(by_loc.quantile(0.25)), float(by_loc.quantile(0.75))],
        "locations_biased_high_pct": float(100 * (by_loc > 0).mean()),
        "worst_locations": [{"location": int(k), "bias": float(by_loc[k])}
                            for k in by_loc.abs().nlargest(8).index],
    }


def lai_diagnostics(pairs: pd.DataFrame, bins: list[float]) -> dict:
    """The full LAI diagnostic bundle recorded with every iteration."""
    if pairs.empty:
        return {"error": "no matched observed/simulated LAI pairs"}
    seasons, shape = lai_shape(pairs)
    bin_table = lai_bin_table(pairs, bins)
    return _round({
        "overall": metrics(pairs["LAI"], pairs["LAI_sim"]),
        "n_pairs": int(len(pairs)),
        "n_locations": int(pairs["location"].nunique()),
        "years": sorted(int(y) for y in pairs["year"].unique()),
        "by_dvs_bin": bin_table.to_dict("records"),
        "curve_shape": shape,
        "residual_structure": lai_residual_structure(pairs),
    }), seasons


def lai_plots(pairs: pd.DataFrame, seasons: pd.DataFrame, bins: list[float],
              out_dir: Path, title: str, sim_daily: pd.DataFrame | None = None) -> list[str]:
    """Diagnostic figures for one LAI iteration. Returns the written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    # --- 1. mean trajectory along DVS + per-bin bias --------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    grid = np.linspace(0, 2, 41)
    band = pairs.assign(b=pd.cut(pairs["dvs"], grid))
    agg = band.groupby("b", observed=True).agg(
        x=("dvs", "mean"), o=("LAI", "median"), s=("LAI_sim", "median"),
        o_lo=("LAI", lambda v: v.quantile(0.25)), o_hi=("LAI", lambda v: v.quantile(0.75)),
        s_lo=("LAI_sim", lambda v: v.quantile(0.25)), s_hi=("LAI_sim", lambda v: v.quantile(0.75)),
    ).dropna()
    ax = axes[0]
    ax.fill_between(agg["x"], agg["o_lo"], agg["o_hi"], color=OBS, alpha=0.16, linewidth=0)
    ax.fill_between(agg["x"], agg["s_lo"], agg["s_hi"], color=SIM, alpha=0.16, linewidth=0)
    ax.plot(agg["x"], agg["o"], color=OBS, lw=2, label="observed (median)")
    ax.plot(agg["x"], agg["s"], color=SIM, lw=2, label="simulated (median)")
    for edge in bins[1:-1]:
        ax.axvline(edge, color=GRID, lw=0.8, zorder=0)
    ax.set(title="LAI along development stage", xlabel="DVS", ylabel="LAI (m2/m2)")
    ax.legend(loc="upper left")

    ax = axes[1]
    table = lai_bin_table(pairs, bins)
    y = np.arange(len(table))
    colors = [SIM if b > 0 else OBS for b in table["bias"]]
    ax.barh(y, table["bias"], color=colors, linewidth=0)
    ax.set_yticks(y, [s.replace("(", "").replace("]", "") for s in table["dvs_bin"]], fontsize=8)
    ax.axvline(0, color=INK2, lw=1)
    ax.invert_yaxis()
    ax.set(title="Bias by DVS bin  (sim - obs)   -> SLA element to move",
           xlabel="LAI bias (m2/m2)")
    fig.suptitle(title, x=0.006, ha="left", fontsize=11.5, color=INK, weight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = out_dir / "lai_trajectory_and_bias.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig); written.append(str(path))

    # --- 2. shape features: peak level and peak timing ------------------------
    if not seasons.empty:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        panels = [("obs_peak", "sim_peak", "Peak LAI", "LAI (m2/m2)"),
                  ("obs_peak_doy", "sim_peak_doy", "Peak timing", "day of year"),
                  ("obs_plateau_days", "sim_plateau_days", "Days at >=80% of peak", "days")]
        for ax, (o, s, name, unit) in zip(axes, panels):
            d = seasons[[o, s]].dropna()
            if d.empty:
                ax.set_axis_off(); continue
            lo = float(min(d[o].min(), d[s].min())); hi = float(max(d[o].max(), d[s].max()))
            pad = 0.05 * (hi - lo or 1)
            ax.scatter(d[o], d[s], s=16, color=SIM, alpha=0.55, linewidth=0)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK2, lw=1, ls="--")
            ax.set(title=name, xlabel=f"observed {unit}", ylabel=f"simulated {unit}")
            ax.text(0.03, 0.95, f"median diff {float((d[s] - d[o]).median()):+.3g}",
                    transform=ax.transAxes, va="top", fontsize=8.5, color=INK2)
        fig.suptitle(f"{title} — curve shape, one point per location-season",
                     x=0.006, ha="left", fontsize=11.5, color=INK, weight="semibold")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        path = out_dir / "lai_shape_features.png"
        fig.savefig(path, bbox_inches="tight"); plt.close(fig); written.append(str(path))

    # --- 3. residual structure ----------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    res = pairs["LAI_sim"] - pairs["LAI"]
    ax = axes[0]
    binned = pairs.assign(b=pd.cut(pairs["dvs"], bins, include_lowest=True), r=res)
    groups = [g["r"].to_numpy() for _, g in binned.groupby("b", observed=True)]
    labels = [str(k) for k, _ in binned.groupby("b", observed=True)]
    if groups:
        bp = ax.boxplot(groups, vert=True, showfliers=False, patch_artist=True, widths=0.6)
        for patch in bp["boxes"]:
            patch.set(facecolor=SIM, alpha=0.35, linewidth=0.8, edgecolor=INK2)
        for whisk in bp["medians"]:
            whisk.set(color=INK, linewidth=1.4)
        ax.set_xticks(range(1, len(labels) + 1), labels, rotation=40, ha="right", fontsize=7.5)
    ax.axhline(0, color=INK2, lw=1)
    ax.set(title="Residual by DVS bin", ylabel="LAI residual (sim - obs)")

    ax = axes[1]
    by_month = pairs.assign(month=pairs["date"].dt.month, r=res).groupby("month")["r"].mean()
    ax.bar(by_month.index, by_month.values, color=SIM, linewidth=0)
    ax.axhline(0, color=INK2, lw=1)
    ax.set(title="Mean residual by month  (phase error shows as a sign flip)",
           xlabel="month", ylabel="LAI residual")

    ax = axes[2]
    lo = float(min(pairs["LAI"].min(), pairs["LAI_sim"].min()))
    hi = float(max(pairs["LAI"].max(), pairs["LAI_sim"].max()))
    ax.hexbin(pairs["LAI"], pairs["LAI_sim"], gridsize=42, cmap="Blues", mincnt=1,
              linewidths=0, extent=(lo, hi, lo, hi))
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls="--")
    m = metrics(pairs["LAI"], pairs["LAI_sim"])
    ax.text(0.03, 0.95, f"RMSE {m['RMSE']:.3f}\nbias {m['bias']:+.3f}\nR2 {m.get('R2') or float('nan'):.2f}",
            transform=ax.transAxes, va="top", fontsize=8.5, color=INK2)
    ax.set(title="Observed vs simulated (daily pairs)", xlabel="observed LAI", ylabel="simulated LAI")
    fig.suptitle(f"{title} — residual structure", x=0.006, ha="left",
                 fontsize=11.5, color=INK, weight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = out_dir / "lai_residuals.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig); written.append(str(path))

    # --- 4. individual seasons: the full simulated curve vs the retrievals ----
    if sim_daily is not None and not sim_daily.empty:
        keys = (seasons.sort_values("n", ascending=False)[["location", "year"]].head(6).values
                if not seasons.empty else pairs[["location", "year"]].drop_duplicates().head(6).values)
        if len(keys):
            ncol = 3
            nrow = int(np.ceil(len(keys) / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.2 * nrow), squeeze=False)
            for ax, (loc, yr) in zip(axes.ravel(), keys):
                curve = sim_daily[(sim_daily["location"] == loc)].sort_values("date")
                pts = pairs[(pairs["location"] == loc) & (pairs["year"] == yr)]
                if not curve.empty:
                    ax.plot(curve["date"], curve["LAI_sim"], color=SIM, lw=1.6, label="simulated")
                ax.scatter(pts["date"], pts["LAI"], s=18, color=OBS, zorder=3, label="observed")
                ax.set(title=f"location {int(loc)} · {int(yr)}", ylabel="LAI")
                ax.tick_params(axis="x", rotation=30, labelsize=7.5)
            for ax in axes.ravel()[len(keys):]:
                ax.set_axis_off()
            axes[0][0].legend(loc="upper left")
            fig.suptitle(f"{title} — example seasons (full simulated curve, GLASS retrievals)",
                         x=0.006, ha="left", fontsize=11.5, color=INK, weight="semibold")
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            path = out_dir / "lai_example_seasons.png"
            fig.savefig(path, bbox_inches="tight"); plt.close(fig); written.append(str(path))

    return written


def load_lai_daily(run, limit: int | None = 150) -> pd.DataFrame:
    """Full simulated daily LAI (not only the matched dates), for the season plots."""
    try:
        sim = common.read_outputs(run.out_dir, "daily", limit=limit)
    except RuntimeError:
        return pd.DataFrame()
    sim = sim[sim["DevStage"] > 0].copy()
    sim["date"] = pd.to_datetime(sim["DATE"], format="%d.%m.%Y")
    return sim.rename(columns={"Location": "location", "Year": "year",
                               "DevStage": "dvs", "LAI": "LAI_sim"})[
        ["location", "year", "date", "dvs", "LAI_sim"]]


# ===========================================================================
# Yield
# ===========================================================================
ATTRIBUTION_COLUMNS = ["Location", "Year", "Yield_t_ha", "AGBiomass_t_ha", "maxLAI",
                       "Yield_translocated_t_ha", "TRANRF", "NNI", "NPKI"]


def yield_attribution_frame(run) -> pd.DataFrame:
    """District-year frame with the simulated state variables behind the yield."""
    raw = common.read_outputs(run.out_dir, "yearly")
    keep = [c for c in ATTRIBUTION_COLUMNS if c in raw.columns]
    sim = raw[keep].rename(columns={"Location": "location", "Year": "year"})

    points = (pd.read_csv(run.project_csv, sep=";",
                          usecols=["vLocationID", "vNUTS_ID", "vSTATE_NAME"])
              .drop_duplicates()
              .rename(columns={"vLocationID": "location", "vNUTS_ID": "NUTS_ID",
                               "vSTATE_NAME": "STATE_NAME"}))
    sim = sim.merge(points, on="location", how="inner")
    value_cols = [c for c in sim.columns if c not in ("location", "year", "NUTS_ID", "STATE_NAME")]
    sim = sim.groupby(["NUTS_ID", "STATE_NAME", "year"], as_index=False)[value_cols].mean()

    obs = pd.read_csv(run.crop_dir / "data_observed" / f"yield_{run.crop}.csv",
                      usecols=["NUTS_ID", "year", "yield"])
    obs = obs.assign(**{"yield": obs["yield"] * run.dm_fraction})
    return sim.merge(obs, on=["NUTS_ID", "year"], how="inner")


def yield_diagnostics(frame: pd.DataFrame, reference: dict | None = None) -> dict:
    """Yield error decomposed by year, by region, and along ``yield = AGB x HI``."""
    if frame.empty:
        return {"error": "no matched observed/simulated yield pairs"}

    frame = frame.copy()
    frame["residual"] = frame["yield_sim"] - frame["yield"] \
        if "yield_sim" in frame else frame["Yield_t_ha"] - frame["yield"]
    sim_col = "yield_sim" if "yield_sim" in frame else "Yield_t_ha"

    by_year = frame.groupby("year").agg(
        obs=("yield", "mean"), sim=(sim_col, "mean"), n=("yield", "size")).reset_index()
    by_year["bias"] = by_year["sim"] - by_year["obs"]
    by_state = frame.groupby("STATE_NAME").agg(
        obs=("yield", "mean"), sim=(sim_col, "mean"), n=("yield", "size")).reset_index()
    by_state["bias"] = by_state["sim"] - by_state["obs"]
    by_district = frame.groupby("NUTS_ID")["residual"].mean()

    out = {
        "overall": metrics(frame["yield"], frame[sim_col]),
        "n_district_years": int(len(frame)),
        "n_districts": int(frame["NUTS_ID"].nunique()),
        "years": [int(frame["year"].min()), int(frame["year"].max())],
        "temporal": {
            "RMSE_year_means": float(np.sqrt(np.mean((by_year["sim"] - by_year["obs"]) ** 2))),
            "residual_trend_per_year": _slope(by_year["year"], by_year["bias"]),
            "worst_years": by_year.reindex(by_year["bias"].abs().sort_values(ascending=False).index)
                                  .head(6).to_dict("records"),
            "by_year": by_year.to_dict("records"),
        },
        "spatial": {
            "RMSE_state_means": float(np.sqrt(np.mean((by_state["sim"] - by_state["obs"]) ** 2))),
            "state_bias_range": [float(by_state["bias"].min()), float(by_state["bias"].max())],
            "district_bias_iqr": [float(by_district.quantile(0.25)),
                                  float(by_district.quantile(0.75))],
            "by_state": by_state.sort_values("bias").to_dict("records"),
        },
    }

    # --- attribution: biomass or harvest index? ------------------------------
    if "AGBiomass_t_ha" in frame.columns:
        agb = frame["AGBiomass_t_ha"].replace(0, np.nan)
        hi_sim = (frame[sim_col] / agb)
        hi_needed = (frame["yield"] / agb)
        agb_needed = frame["yield"] / hi_sim.replace(0, np.nan)
        ref = reference or {}
        hi_range = ref.get("harvest_index")
        agb_range = ref.get("ag_biomass_t_ha")

        hi_med, hi_need_med = float(hi_sim.median()), float(hi_needed.median())
        agb_med, agb_need_med = float(agb.median()), float(agb_needed.median())

        def inside(value, span):
            return None if not span else bool(span[0] <= value <= span[1])

        hi_ok, agb_ok = inside(hi_med, hi_range), inside(agb_med, agb_range)
        hi_need_ok, agb_need_ok = inside(hi_need_med, hi_range), inside(agb_need_med, agb_range)

        if hi_ok is None or agb_ok is None:
            verdict = ("no crop reference ranges configured — compare hi_simulated and "
                       "ag_biomass_simulated against agronomic expectation manually")
        elif abs(out["overall"]["bias"]) < 0.25:
            verdict = "bias is small; no attribution needed"
        elif agb_ok and not hi_ok:
            verdict = ("biomass is plausible but the harvest index is not -> yield-specific "
                       "parameters: post-anthesis RUE, N translocation (TCNT/DVSNT/DVSNLT/"
                       "NMAXSO), FRTDM. Check whether the partitioning tables are a step "
                       "function first — if they are, they cannot be moved.")
        elif hi_ok and not agb_ok:
            verdict = ("harvest index is plausible but biomass is not -> biomass parameters "
                       "(RUETableRUE, KDIFTableK) or an unresolved LAI error")
        elif hi_need_ok and not agb_need_ok:
            verdict = ("closing the gap via the harvest index stays inside the plausible range "
                       "while doing it via biomass does not -> prefer partitioning")
        elif agb_need_ok and not hi_need_ok:
            verdict = ("closing the gap via biomass stays inside the plausible range while doing "
                       "it via the harvest index does not -> prefer RUE / KDIF")
        else:
            verdict = ("both routes are plausible -> use the residual correlations and the "
                       "spatial/temporal structure to choose")

        residual = frame["residual"]
        correlations = {}
        for name in ("maxLAI", "AGBiomass_t_ha", "TRANRF", "NNI", "NPKI"):
            if name in frame.columns:
                col = frame[name]
                ok = col.notna() & residual.notna()
                if ok.sum() > 3 and col[ok].std() > 0 and residual[ok].std() > 0:
                    correlations[name] = float(np.corrcoef(col[ok], residual[ok])[0, 1])

        out["attribution"] = {
            "hi_simulated_median": hi_med,
            "hi_required_to_match_obs": hi_need_med,
            "ag_biomass_simulated_median": agb_med,
            "ag_biomass_required_to_match_obs": agb_need_med,
            "reference_harvest_index": hi_range,
            "reference_ag_biomass_t_ha": agb_range,
            "verdict": verdict,
            "residual_correlations": correlations,
            "correlation_note": ("a strong negative correlation with TRANRF or NNI means the "
                                 "shortfall concentrates in stressed site-years, which points at "
                                 "the stress-response parameters rather than the potential ones"),
        }

    out["translocation"] = _translocation_diagnostics(frame)
    return _round(out)


def _translocation_diagnostics(frame: pd.DataFrame) -> dict | None:
    """Does the FRTDM translocation term move the model toward the observations?

    The solution reports two storage-organ weights: ``Yield_t_ha`` from the biomass
    component, and ``Yield_translocated_t_ha`` from ``BiomassTranslocation``, which
    adds the pre-anthesis reserves that FRTDM makes available for remobilisation.

    Scoring both against the same observed district yields is the only honest way
    to answer "is FRTDM worth having": if the translocated series is closer, the
    remobilisation term is doing real work and FRTDM is worth calibrating; if it is
    further away, raising FRTDM will make the fit worse no matter what the yield
    bias looks like.
    """
    sim_col = "yield_sim" if "yield_sim" in frame else "Yield_t_ha"
    if "Yield_translocated_t_ha" not in frame.columns or sim_col not in frame.columns:
        return None

    observed = frame["yield"]
    plain = frame[sim_col]
    translocated = frame["Yield_translocated_t_ha"]
    ok = observed.notna() & plain.notna() & translocated.notna()
    if ok.sum() < 5:
        return None

    observed, plain, translocated = observed[ok], plain[ok], translocated[ok]
    m_plain = metrics(observed, plain)
    m_trans = metrics(observed, translocated)
    delta = m_trans["RMSE"] - m_plain["RMSE"]

    contribution = float((translocated - plain).median())
    if abs(delta) < 0.05:
        verdict = ("the two yield definitions fit the observations equally well — the "
                   "translocation term is not currently distinguishable, so FRTDM is "
                   "not the parameter to move")
    elif delta < 0:
        verdict = (f"translocated yield fits better by {abs(delta):.3f} t/ha RMSE — the "
                   f"remobilisation term is doing real work; FRTDM is worth calibrating "
                   f"and the reported Yield_t_ha understates what the crop achieves")
    else:
        verdict = (f"translocated yield fits WORSE by {delta:.3f} t/ha RMSE — the "
                   f"remobilisation term overshoots; lower FRTDM rather than raising it")

    return {
        "yield_plain": m_plain,
        "yield_translocated": m_trans,
        "rmse_delta_translocated_minus_plain": float(delta),
        "median_translocation_contribution_t_ha": contribution,
        "mean_translocated_t_ha": float(translocated.mean()),
        "mean_plain_t_ha": float(plain.mean()),
        "verdict": verdict,
        "note": ("Yield_t_ha comes from the biomass component; Yield_translocated_t_ha "
                 "comes from BiomassTranslocation, which adds the pre-anthesis reserves "
                 "FRTDM makes available. The objective is scored on Yield_t_ha."),
    }


def yield_plots(frame: pd.DataFrame, out_dir: Path, title: str) -> list[str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    sim_col = "yield_sim" if "yield_sim" in frame else "Yield_t_ha"

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    ax = axes[0]
    lo = float(min(frame["yield"].min(), frame[sim_col].min()))
    hi = float(max(frame["yield"].max(), frame[sim_col].max()))
    ax.hexbin(frame["yield"], frame[sim_col], gridsize=40, cmap="Blues", mincnt=1,
              linewidths=0, extent=(lo, hi, lo, hi))
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls="--")
    m = metrics(frame["yield"], frame[sim_col])
    ax.text(0.03, 0.95, f"RMSE {m['RMSE']:.3f}\nbias {m['bias']:+.3f}",
            transform=ax.transAxes, va="top", fontsize=8.5, color=INK2)
    ax.set(title="District-year yield", xlabel="observed (t/ha DM)", ylabel="simulated (t/ha DM)")

    ax = axes[1]
    by_year = frame.groupby("year")[["yield", sim_col]].mean()
    ax.plot(by_year.index, by_year["yield"], color=OBS, lw=1.8, label="observed")
    ax.plot(by_year.index, by_year[sim_col], color=SIM, lw=1.8, label="simulated")
    ax.set(title="Yearly means — does the model track good and bad years?",
           xlabel="year", ylabel="yield (t/ha DM)")
    ax.legend(loc="best")

    ax = axes[2]
    by_state = frame.groupby("STATE_NAME")[["yield", sim_col]].mean().sort_values("yield")
    y = np.arange(len(by_state))
    ax.barh(y - 0.2, by_state["yield"], 0.38, color=OBS, linewidth=0, label="observed")
    ax.barh(y + 0.2, by_state[sim_col], 0.38, color=SIM, linewidth=0, label="simulated")
    ax.set_yticks(y, by_state.index, fontsize=7.5)
    ax.set(title="State means — the regional gradient", xlabel="yield (t/ha DM)")
    ax.legend(loc="lower right")
    fig.suptitle(title, x=0.006, ha="left", fontsize=11.5, color=INK, weight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = out_dir / "yield_overview.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig); written.append(str(path))

    # Attribution panel: what does the residual travel with?
    drivers = [c for c in ("AGBiomass_t_ha", "maxLAI", "TRANRF", "NNI") if c in frame.columns]
    if drivers:
        residual = frame[sim_col] - frame["yield"]
        fig, axes = plt.subplots(1, len(drivers), figsize=(3.6 * len(drivers), 3.6), squeeze=False)
        for ax, name in zip(axes[0], drivers):
            ax.scatter(frame[name], residual, s=8, color=SIM, alpha=0.35, linewidth=0)
            ax.axhline(0, color=INK2, lw=1)
            ok = frame[name].notna() & residual.notna()
            if ok.sum() > 3 and frame[name][ok].std() > 0:
                r = float(np.corrcoef(frame[name][ok], residual[ok])[0, 1])
                ax.text(0.03, 0.95, f"r = {r:+.2f}", transform=ax.transAxes,
                        va="top", fontsize=8.5, color=INK2)
            ax.set(title=name, xlabel=name, ylabel="yield residual (t/ha)")
        fig.suptitle(f"{title} — what the yield error travels with",
                     x=0.006, ha="left", fontsize=11.5, color=INK, weight="semibold")
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        path = out_dir / "yield_attribution.png"
        fig.savefig(path, bbox_inches="tight"); plt.close(fig); written.append(str(path))

    return written


# ===========================================================================
# History
# ===========================================================================
def history_plot(records: list[dict], out_path: Path, title: str) -> str | None:
    """Objective per iteration, with the running best and what changed each time."""
    done = [r for r in records if r.get("status") == "completed"]
    if len(done) < 1:
        return None
    done.sort(key=lambda r: r["iteration"])
    it = [r["iteration"] for r in done]
    obj = [r["objective"] for r in done]
    best = np.minimum.accumulate(obj)

    fig, ax = plt.subplots(figsize=(max(6.0, 0.55 * len(it) + 3), 4.0))
    ax.plot(it, obj, marker="o", ms=5, color=SIM, lw=1.4, label="iteration objective")
    ax.plot(it, best, color=OBS, lw=1.8, ls="--", label="best so far")
    for r, x, y in zip(done, it, obj):
        changed = ", ".join(sorted(r.get("parameters_changed", {}))) or "baseline"
        ax.annotate(changed, (x, y), textcoords="offset points", xytext=(0, 8),
                    fontsize=6.5, color=MUTED, ha="center", rotation=25)
    ax.set(title=title, xlabel="iteration", ylabel="objective")
    ax.set_xticks(it)
    ax.legend(loc="best")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)
