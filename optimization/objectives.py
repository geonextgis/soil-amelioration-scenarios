#!/usr/bin/env python
"""The three calibration losses — one per *view*: phenology, LAI, yield.

A view is one model run scored against one observation set. Each owns exactly
two functions:

    process_result(run_spec) -> DataFrame   join simulated output to observations
    loss_fn(frame)           -> (loss, metrics)

A calibration *stage* is one or more views plus the weights that combine them
(:func:`combine`). Stage 1 (phenology) has a single view; stage 2 (growth) scores
the LAI and yield views together, which is what makes the two interdependent
parameter sets calibratable in one loop.

Nothing here decides anything. These are pure scoring functions — the decision
layer (``calibrate.py``/``agents/``) reads the loss and the diagnostics derived
from the same frames and chooses the next parameter move.

    import objectives
    process_result, loss_fn = objectives.for_view("lai")
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


# ===========================================================================
# Phenology — anthesis and maturity date, in days
# ===========================================================================
#
# loss = ( RMSE(flowering_doy) + RMSE(maturity_doy) ) / 2
#
# Simulated ``AnthesisDOY`` / ``MaturityDOY`` (yearly output) against observed
# ``flowering_doy`` / ``maturity_doy``, joined on PointID + harvest_year.
#
# Autumn bolting is excluded. Some winter-crop location-years flower in the
# autumn of the establishment year instead of the following spring, landing at
# DOY ~280-365; that is a known weather-driven model artifact, and leaving those
# rows in would let a ~300-day error dominate the RMSE and steer the thermal-time
# parameters to compensate for something they do not cause.

PHENOLOGY_SIM_COLUMNS = {
    "Location": "PointID",
    "Year": "harvest_year",
    "AnthesisDate": "flowering_date_sim",
    "AnthesisDOY": "flowering_doy_sim",
    "MaturityDate": "maturity_date_sim",
    "MaturityDOY": "maturity_doy_sim",
}
PHENOLOGY_OBS_COLUMNS = ["PointID", "harvest_year", "flowering_doy", "maturity_doy"]


def phenology_process_result(spec: common.RunSpec) -> pd.DataFrame:
    """Join simulated and observed phenology.

    The number of rows dropped as autumn bolting is attached as
    ``frame.attrs["n_bolting"]`` — the diagnostics report it, and a sudden jump in
    it is itself a signal that a thermal-time change has gone wrong.
    """
    sim = common.read_outputs(spec.out_dir, "yearly")[list(PHENOLOGY_SIM_COLUMNS)]
    sim = sim.rename(columns=PHENOLOGY_SIM_COLUMNS)

    # Autumn bolting: simulated anthesis falls in the calendar year *before* harvest.
    anthesis_year = pd.to_datetime(
        sim["flowering_date_sim"], format="%d.%m.%Y", errors="coerce").dt.year
    bolting = anthesis_year < sim["harvest_year"]
    n_bolting = int(bolting.sum())
    sim = sim[~bolting.fillna(False)]

    obs = pd.read_csv(spec.crop_dir / "data_observed" / f"phenology_{spec.crop}.csv",
                      usecols=PHENOLOGY_OBS_COLUMNS)

    merged = sim.merge(obs, on=["PointID", "harvest_year"], how="inner")
    keep = ["flowering_doy", "flowering_doy_sim", "maturity_doy", "maturity_doy_sim"]
    merged = merged.dropna(subset=keep)
    merged.attrs["n_bolting"] = n_bolting
    return merged


def phenology_loss_fn(df: pd.DataFrame) -> tuple[float, dict]:
    """Mean of the flowering and maturity RMSEs, in days."""
    if df.empty:
        raise RuntimeError("no simulated/observed phenology pairs matched")

    flowering = rmse(df["flowering_doy"], df["flowering_doy_sim"])
    maturity = rmse(df["maturity_doy"], df["maturity_doy_sim"])
    loss = (flowering + maturity) / 2.0

    metrics = {
        "RMSE flowering (days)": round(flowering, 3),
        "RMSE maturity (days)": round(maturity, 3),
        "bias flowering (days)": round(
            float((df["flowering_doy_sim"] - df["flowering_doy"]).mean()), 3),
        "bias maturity (days)": round(
            float((df["maturity_doy_sim"] - df["maturity_doy"]).mean()), 3),
        "matched location-years": int(len(df)),
        "locations": int(df["PointID"].nunique()),
        "excluded (autumn bolting)": int(df.attrs.get("n_bolting", 0)),
    }
    return loss, metrics


# ===========================================================================
# LAI — canopy trajectory against GLASS retrievals
# ===========================================================================
#
# loss = RMSE( mean LAI_obs , mean LAI_sim )   over (DVS bin x year)
#
# The binning is deliberate. Daily GLASS-LAI retrievals are noisy and unevenly
# distributed over the season, so a raw daily RMSE is dominated by observation
# noise and by whichever growth stage happens to have the most retrievals.
# Averaging within development-stage bins weights each phase of the canopy
# trajectory equally, which is what the SLA/RGRLAI parameters actually control.
#
# The simulated series is truncated at physiological maturity (DVS >= 2) so
# post-maturity senescence does not enter the comparison.
#
# Note: this target reads the LAI calibration project table, whose point set and
# grid differ from the baseline table, so the run dir stages the matching
# ``*_LAI`` soil / location / fertilizer tables under the canonical names.

# Development-stage bin edges: emergence -> tillering -> stem elongation ->
# anthesis -> the three grain-filling phases -> maturity.
DVS_BINS = [0.0, 0.25, 0.5, 1.0, 1.25, 1.50, 1.75, 2.0]


def truncate_at_maturity(df: pd.DataFrame) -> pd.DataFrame:
    """Cut each location's series at the first day physiological maturity is reached.

    Expects ``df`` sorted by (Location, DATE) with a fresh RangeIndex.
    """
    pos = df.groupby("Location").cumcount()
    first_mature = pos.where(df["DevStage"] >= 2).groupby(df["Location"]).min()
    cutoff = df["Location"].map(first_mature)
    return df[cutoff.isna() | (pos <= cutoff)]


def lai_process_result(spec: common.RunSpec) -> pd.DataFrame:
    """Join simulated daily LAI to the observed GLASS retrievals on location + date."""
    sim = common.read_outputs(spec.out_dir, "daily")
    sim = sim[sim["DevStage"] > 0]
    sim["DATE"] = pd.to_datetime(sim["DATE"], format="%d.%m.%Y")

    # One season per location in the LAI calibration table, so a per-location
    # truncation is a per-season truncation.
    sim = (
        sim[["Location", "Year", "DATE", "DevStage", "LAI"]]
        .sort_values(["Location", "DATE"])
        .reset_index(drop=True)
        .pipe(truncate_at_maturity)
        .rename(columns={"Location": "location", "Year": "year",
                         "DATE": "date", "DevStage": "dvs", "LAI": "LAI_sim"})
    )

    obs = pd.read_csv(spec.crop_dir / "data_observed" / f"LAI_{spec.crop}.csv",
                      parse_dates=["date"])

    return sim.merge(obs, on=["location", "date"], how="inner")


def aggregate_by_dvs(df: pd.DataFrame, bins: list[float]) -> pd.DataFrame:
    df = df.assign(dvs_bin=pd.cut(df["dvs"], bins=bins, include_lowest=True))
    return (
        df.groupby(["dvs_bin", "year"], observed=True)
        .agg(LAI_obs_mean=("LAI", "mean"), LAI_sim_mean=("LAI_sim", "mean"))
        .dropna()
        .reset_index()
        .sort_values("year")
    )


def lai_loss_fn(df: pd.DataFrame) -> tuple[float, dict]:
    """RMSE over DVS-bin x year means of observed vs simulated LAI."""
    if df.empty:
        raise RuntimeError("no simulated/observed LAI pairs matched")

    agg = aggregate_by_dvs(df, DVS_BINS)
    if agg.empty:
        raise RuntimeError("DVS aggregation produced no groups")

    obs, sim = agg["LAI_obs_mean"].to_numpy(), agg["LAI_sim_mean"].to_numpy()
    loss = float(np.sqrt(np.mean((obs - sim) ** 2)))

    metrics = {
        "RMSE LAI (DVS-binned)": round(loss, 4),
        "bias LAI": round(float(np.mean(sim - obs)), 4),
        "RMSE LAI (raw daily)": round(
            float(np.sqrt(np.mean((df["LAI"] - df["LAI_sim"]) ** 2))), 4),
        "DVS bin x year groups": int(len(agg)),
        "matched daily pairs": int(len(df)),
        "locations": int(df["location"].nunique()),
    }
    return loss, metrics


# ===========================================================================
# Yield — district yields, temporal and spatial
# ===========================================================================
#
# loss = ( RMSE(yearly means) + RMSE(state means) ) / 2      [t/ha]
#
# Observed yield is district-level (one row per NUTS_ID + year) while the model
# runs per point, so simulated points are first averaged to NUTS-3 and joined on
# NUTS_ID + year. Scoring the year means and the state means separately — rather
# than the raw district-year pairs — keeps the loss from being dominated by
# district-level noise, and balances "does the model track good and bad years"
# against "does it get the regional yield gradient right". A parameter set that
# nails the national mean but flattens either dimension is penalised.
#
# Observed yield is multiplied by ``dm_fraction`` (config, per crop) to put it on
# the model's dry-matter basis.


# Carried alongside the yield itself so the loss can report the harvest index and
# the translocated-yield comparison without a second pass over the output files.
# Absent columns are tolerated: a crop whose solution predates BiomassTranslocation
# simply reports fewer metrics instead of failing the whole iteration.
YIELD_EXTRA_COLUMNS = {
    "AGBiomass_t_ha": "ag_biomass_sim",
    "Yield_translocated_t_ha": "yield_translocated_sim",
}


def yield_process_result(spec: common.RunSpec) -> pd.DataFrame:
    """Aggregate simulated point yields to NUTS-3 and join the observed districts.

    Also carries above-ground biomass and translocated yield through the same
    aggregation, so harvest index and the FRTDM translocation term are reported on
    exactly the district-years the objective is scored on.
    """
    raw = common.read_outputs(spec.out_dir, "yearly")
    extra = {src: dst for src, dst in YIELD_EXTRA_COLUMNS.items() if src in raw.columns}
    sim = raw[["Location", "Year", "Yield_t_ha", *extra]]
    sim = sim.rename(columns={"Location": "location", "Year": "year",
                              "Yield_t_ha": "yield_sim", **extra})

    # District/state lookup from this run's own project table.
    points = (
        pd.read_csv(spec.project_csv, sep=";",
                    usecols=["vLocationID", "vNUTS_ID", "vSTATE_NAME"])
        .drop_duplicates()
        .rename(columns={"vLocationID": "location", "vNUTS_ID": "NUTS_ID",
                         "vSTATE_NAME": "STATE_NAME"})
    )

    sim = sim.merge(points, on="location", how="inner")
    value_columns = ["yield_sim", *extra.values()]
    sim = (sim.groupby(["NUTS_ID", "STATE_NAME", "year"], as_index=False)[value_columns]
              .mean())

    obs = pd.read_csv(spec.crop_dir / "data_observed" / f"yield_{spec.crop}.csv",
                      usecols=["NUTS_ID", "year", "yield"])
    obs = obs.assign(**{"yield": obs["yield"] * spec.dm_fraction})

    return sim.merge(obs, on=["NUTS_ID", "year"], how="inner")


def yield_loss_fn(df: pd.DataFrame) -> tuple[float, dict]:
    """Mean of the yearly-mean RMSE (temporal) and state-mean RMSE (spatial).

    The **objective is unchanged** by the extra metrics below — harvest index and
    the translocated-yield comparison are reported, not scored. Adding either to
    the loss would silently invalidate every objective already in the ledger.
    """
    if df.empty:
        raise RuntimeError("no simulated/observed yield pairs matched")

    by_year = df.groupby("year")[["yield", "yield_sim"]].mean()
    by_state = df.groupby("STATE_NAME")[["yield", "yield_sim"]].mean()

    loss_year = rmse(by_year["yield"], by_year["yield_sim"])
    loss_state = rmse(by_state["yield"], by_state["yield_sim"])
    loss = (loss_year + loss_state) / 2.0

    metrics = {
        "RMSE yearly means (t/ha)": round(loss_year, 4),
        "RMSE state means (t/ha)": round(loss_state, 4),
        "RMSE district-year (t/ha)": round(rmse(df["yield"], df["yield_sim"]), 4),
        "bias (t/ha)": round(float((df["yield_sim"] - df["yield"]).mean()), 4),
        "mean obs (t/ha)": round(float(df["yield"].mean()), 4),
        "mean sim (t/ha)": round(float(df["yield_sim"].mean()), 4),
        "district-years": int(len(df)),
        "districts": int(df["NUTS_ID"].nunique()),
    }

    # --- harvest index -------------------------------------------------------
    if "ag_biomass_sim" in df:
        biomass = df["ag_biomass_sim"].to_numpy(float)
        usable = biomass > 0
        if usable.any():
            hi_sim = df["yield_sim"].to_numpy(float)[usable] / biomass[usable]
            # What HI would have to be, on the same biomass, to hit the observed
            # yield. The gap between the two is the partitioning question.
            hi_needed = df["yield"].to_numpy(float)[usable] / biomass[usable]
            metrics.update({
                "harvest index (sim)": round(float(np.median(hi_sim)), 4),
                "harvest index (required)": round(float(np.median(hi_needed)), 4),
                "HI bias": round(float(np.median(hi_sim) - np.median(hi_needed)), 4),
                "mean AG biomass (t/ha)": round(float(np.mean(biomass[usable])), 4),
            })

    # --- translocated yield (the FRTDM term) ---------------------------------
    if "yield_translocated_sim" in df:
        translocated = df["yield_translocated_sim"]
        metrics.update({
            "mean translocated yield (t/ha)": round(float(translocated.mean()), 4),
            "RMSE translocated vs obs (t/ha)": round(rmse(df["yield"], translocated), 4),
            # >1 means the translocation component adds to the storage organ; the
            # size of that addition is what FRTDM controls.
            "translocated / yield ratio": round(
                float((translocated.mean() / df["yield_sim"].mean())), 4)
            if df["yield_sim"].mean() else None,
        })

    return loss, metrics


# ===========================================================================
# Registry + combination
# ===========================================================================
VIEWS = {
    "phenology": (phenology_process_result, phenology_loss_fn),
    "lai": (lai_process_result, lai_loss_fn),
    "yield": (yield_process_result, yield_loss_fn),
}


def for_view(view: str):
    """``(process_result, loss_fn)`` for one view."""
    if view not in VIEWS:
        raise SystemExit(f"no loss registered for view {view!r}; have {sorted(VIEWS)}")
    return VIEWS[view]


def combine(components: dict, losses: dict) -> tuple[float, dict]:
    """Weighted mean of the per-view losses, each divided by its own scale.

        objective = sum( weight_i * loss_i / scale_i ) / sum( weight_i )

    ``scale`` is the loss value at which a component counts as 1.0 — its target.
    That makes components with different units (LAI in m2/m2, yield in t/ha)
    commensurable and gives the combined objective a fixed meaning: **1.0 means
    every component sits at its target on average**, and no component can be
    traded away silently because its contribution is recorded here.

    A single-view stage with ``scale: 1.0`` reduces to the raw loss, so the
    phenology objective is still an RMSE in days.
    """
    if not components:
        raise SystemExit("the objective declares no components")
    missing = [view for view in components if view not in losses]
    if missing:
        raise SystemExit(f"no loss computed for objective component(s): {missing}")

    breakdown, total, weights = {}, 0.0, 0.0
    for view, cfg in components.items():
        weight = float(cfg.get("weight", 1.0))
        scale = float(cfg.get("scale", 1.0))
        if scale <= 0:
            raise SystemExit(f"objective component {view!r} has a non-positive scale")
        scaled = float(losses[view]) / scale
        total += weight * scaled
        weights += weight
        breakdown[view] = {"loss": round(float(losses[view]), 6), "scale": scale,
                           "weight": weight, "scaled": round(scaled, 6),
                           "contribution": round(weight * scaled, 6)}
    if weights <= 0:
        raise SystemExit("objective component weights sum to zero")

    objective = total / weights
    for view in breakdown:
        breakdown[view]["share_of_objective"] = round(
            breakdown[view]["contribution"] / weights / objective, 4) if objective else None
    return float(objective), breakdown
