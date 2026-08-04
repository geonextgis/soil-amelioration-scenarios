#!/usr/bin/env python
"""Yield calibration — one trial per invocation.

Loss
----
Mean of a temporal and a spatial RMSE, in t/ha:

    loss = ( RMSE(yearly means) + RMSE(state means) ) / 2

Observed yield is district-level (one row per NUTS_ID + year) while the model
runs per point, so simulated points are first averaged to NUTS-3 and joined on
NUTS_ID + year. Scoring the year means and the state means separately — rather
than the raw district-year pairs — keeps the loss from being dominated by
district-level noise, and balances "does the model track good and bad years"
against "does it get the regional yield gradient right". A parameter set that
nails the national mean but flattens either dimension is penalised.

Observed yield is multiplied by ``dm_fraction`` (config, per crop) to put it on
the model's dry-matter basis.

Usage
-----
    python optimization/optimize_yield.py --crop winter_wheat
    python optimization/optimize_yield.py --crop winter_wheat --show-best
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

TARGET = "yield"


def process_result(spec: common.RunSpec) -> pd.DataFrame:
    """Aggregate simulated point yields to NUTS-3 and join the observed districts."""
    sim = common.read_outputs(spec.out_dir, "yearly")[["Location", "Year", "Yield_t_ha"]]
    sim = sim.rename(columns={"Location": "location", "Year": "year",
                              "Yield_t_ha": "yield_sim"})

    # District/state lookup from this run's own project table.
    points = (
        pd.read_csv(spec.project_csv, sep=";",
                    usecols=["vLocationID", "vNUTS_ID", "vSTATE_NAME"])
        .drop_duplicates()
        .rename(columns={"vLocationID": "location", "vNUTS_ID": "NUTS_ID",
                         "vSTATE_NAME": "STATE_NAME"})
    )

    sim = sim.merge(points, on="location", how="inner")
    sim = (sim.groupby(["NUTS_ID", "STATE_NAME", "year"], as_index=False)["yield_sim"]
              .mean())

    obs = pd.read_csv(spec.crop_dir / "data_observed" / f"yield_{spec.crop}.csv",
                      usecols=["NUTS_ID", "year", "yield"])
    obs = obs.assign(**{"yield": obs["yield"] * spec.dm_fraction})

    return sim.merge(obs, on=["NUTS_ID", "year"], how="inner")


def rmse(a: pd.Series, b: pd.Series) -> float:
    return float(np.sqrt(np.mean((a.to_numpy(float) - b.to_numpy(float)) ** 2)))


def loss_fn(df: pd.DataFrame) -> tuple[float, dict]:
    """Mean of the yearly-mean RMSE (temporal) and state-mean RMSE (spatial)."""
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
    return loss, metrics


def evaluate(spec: common.RunSpec) -> tuple[float, dict]:
    return loss_fn(process_result(spec))


if __name__ == "__main__":
    sys.exit(common.main(TARGET, __doc__, evaluate))
