#!/usr/bin/env python
"""LAI calibration — one trial per invocation.

Loss
----
RMSE of leaf area index, computed on DVS-binned, per-year means rather than on
raw daily pairs:

    loss = RMSE( mean LAI_obs , mean LAI_sim )   over (DVS bin x year)

The binning is deliberate. Daily GLASS-LAI retrievals are noisy and unevenly
distributed over the season, so a raw daily RMSE is dominated by observation
noise and by whichever growth stage happens to have the most retrievals.
Averaging within development-stage bins weights each phase of the canopy
trajectory equally, which is what the SLA/RGRLAI parameters actually control.

The simulated series is truncated at physiological maturity (DVS >= 2) so
post-maturity senescence does not enter the comparison.

Note: this target reads the LAI calibration project table, whose point set and
grid differ from the baseline table, so the run dir stages the matching
``*_LAI`` soil / location / fertilizer tables under the canonical names.

Usage
-----
    python optimization/optimize_lai.py --crop winter_wheat
    python optimization/optimize_lai.py --crop winter_wheat --show-best
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

TARGET = "lai"

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


def process_result(spec: common.RunSpec) -> pd.DataFrame:
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


def loss_fn(df: pd.DataFrame) -> tuple[float, dict]:
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


def evaluate(spec: common.RunSpec) -> tuple[float, dict]:
    return loss_fn(process_result(spec))


if __name__ == "__main__":
    sys.exit(common.main(TARGET, __doc__, evaluate))
