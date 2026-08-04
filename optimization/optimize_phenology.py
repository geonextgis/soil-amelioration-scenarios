#!/usr/bin/env python
"""Phenology calibration — one trial per invocation.

Loss
----
Mean of two RMSEs, in days, over every matched location-year:

    loss = ( RMSE(flowering_doy) + RMSE(maturity_doy) ) / 2

Simulated ``AnthesisDOY`` / ``MaturityDOY`` (yearly output) against observed
``flowering_doy`` / ``maturity_doy``, joined on PointID + harvest_year.

Autumn bolting is excluded. Some winter-crop location-years flower in the autumn
of the establishment year instead of the following spring, landing at DOY ~280-365;
that is a known weather-driven model artifact, and leaving those rows in would let
a ~300-day error dominate the RMSE and steer the thermal-time parameters to
compensate for something they do not cause.

Usage
-----
    python optimization/optimize_phenology.py --crop winter_wheat
    python optimization/optimize_phenology.py --crop winter_wheat --show-best
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

TARGET = "phenology"

SIM_COLUMNS = {
    "Location": "PointID",
    "Year": "harvest_year",
    "AnthesisDate": "flowering_date_sim",
    "AnthesisDOY": "flowering_doy_sim",
    "MaturityDate": "maturity_date_sim",
    "MaturityDOY": "maturity_doy_sim",
}
OBS_COLUMNS = ["PointID", "harvest_year", "flowering_doy", "maturity_doy"]


def process_result(spec: common.RunSpec) -> tuple[pd.DataFrame, int]:
    """Join simulated and observed phenology. Returns (matched rows, n_bolting)."""
    sim = common.read_outputs(spec.out_dir, "yearly")[list(SIM_COLUMNS)]
    sim = sim.rename(columns=SIM_COLUMNS)

    # Autumn bolting: simulated anthesis falls in the calendar year *before* harvest.
    anthesis_year = pd.to_datetime(
        sim["flowering_date_sim"], format="%d.%m.%Y", errors="coerce").dt.year
    bolting = anthesis_year < sim["harvest_year"]
    n_bolting = int(bolting.sum())
    sim = sim[~bolting.fillna(False)]

    obs = pd.read_csv(spec.crop_dir / "data_observed" / f"phenology_{spec.crop}.csv",
                      usecols=OBS_COLUMNS)

    merged = sim.merge(obs, on=["PointID", "harvest_year"], how="inner")
    keep = ["flowering_doy", "flowering_doy_sim", "maturity_doy", "maturity_doy_sim"]
    return merged.dropna(subset=keep), n_bolting


def rmse(a: pd.Series, b: pd.Series) -> float:
    return float(np.sqrt(np.mean((a.to_numpy(float) - b.to_numpy(float)) ** 2)))


def loss_fn(df: pd.DataFrame) -> tuple[float, dict]:
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
    }
    return loss, metrics


def evaluate(spec: common.RunSpec) -> tuple[float, dict]:
    df, n_bolting = process_result(spec)
    loss, metrics = loss_fn(df)
    metrics["excluded (autumn bolting)"] = n_bolting
    return loss, metrics


if __name__ == "__main__":
    sys.exit(common.main(TARGET, __doc__, evaluate))
