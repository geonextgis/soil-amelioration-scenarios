#!/usr/bin/env python
"""Build the historical CO2 forcing file that covers the full 1951-2014 period.

WHY
---
The observed CO2 file shipped with each crop (``co2_mm_observed.csv``) is the
NOAA/Scripps Mauna Loa monthly record, which begins **1958-03**. The historical
experiments start in **1951**, and the solution binds CO2 through

    <resource id="co2observed" ... frequence="DAILY">
      <res id="year"  key="CURRENT.YEAR"/>
      <res id="month" key="CURRENT.MONTH"/>

i.e. an exact (year, month) lookup on every simulated day. A month with no row
is not a zero and not a carry-forward — it is a missing key, and SIMPLACE dies
with a NullPointerException. Without this file, 1951-01 .. 1958-02 kills every
historical run that touches those years.

METHOD
------
Splice the pre-Mauna-Loa years onto the front of the observed record:

1. **Trend** from the Law Dome ice-core + firn + Cape Grim CO2 spline
   (MacFarling Meure et al. 2006; Etheridge et al. 1996, 1998), annual global
   mean, 20-year-attenuated spline. This is the standard reconstruction for
   pre-1958 atmospheric CO2 and is what CMIP6 historical concentrations are
   built on for this era. Annual values are placed at mid-year (Y+0.5) and
   interpolated to monthly.
2. **Seasonal cycle** from Mauna Loa itself: the mean deviation of each calendar
   month from a 12-month centred mean, estimated over the earliest complete
   years of the record (default 1959-1978). The seasonal amplitude at Mauna Loa
   has grown over time, so the *early* record is the right estimator for the
   1950s. Using an MLO-derived cycle also keeps the reconstructed segment on the
   same footing as the observed segment it is glued to.
3. **Offset match** at the splice: the Law Dome spline is a smoothed global mean
   and sits a few tenths of a ppm off the Mauna Loa scale. A single constant
   offset is fitted so the reconstructed trend agrees with the deseasonalised
   Mauna Loa trend over the first years of overlap, which makes the join
   continuous instead of stepped.

Observed months are copied through **verbatim** — this only ever prepends.

Usage:
  python orchestration/build_co2_historical.py            # write to all crops
  python orchestration/build_co2_historical.py --check    # verify, write nothing
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

LAW_DOME = "data/external/law_dome_co2_spline_annual.csv"
SRC_NAME = "co2_mm_observed.csv"       # per-crop Mauna Loa monthly record
OUT_NAME = "co2_mm_historical.csv"     # what this script writes

TARGET_START_YEAR = 1951               # earliest year any historical run needs
SEASON_YEARS = (1959, 1978)            # window used to estimate the MLO seasonal cycle
# Overlap window used to match the splice level. Kept short and adjacent to the
# join: the Law Dome spline grows more slowly than Mauna Loa over 1958-1978, so a
# constant offset fitted over a long window tilts the reconstruction (a 1958-1978
# fit puts 1958 ~0.7 ppm too high and reverses the 1958->1959 growth). Windows of
# 5-10 years converge on ~+0.7 ppm and give a continuous join.
OFFSET_YEARS = (1958, 1963)


def load_law_dome(repo: Path) -> pd.Series:
    """Annual global-mean CO2 spline, indexed by year."""
    path = repo / LAW_DOME
    if not path.exists():
        raise SystemExit(
            f"missing {path}\n"
            "  Regenerate it from the NOAA/NCEI Law Dome archive:\n"
            "    https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/law/law2006.txt\n"
            "  (columns 5 and 6 of the spline table: YearAD, CO2spl)"
        )
    df = pd.read_csv(path, comment="#")
    return df.set_index("year")["co2_ppm"].sort_index()


def seasonal_cycle(obs: pd.DataFrame, lo: int, hi: int) -> pd.Series:
    """Mean deviation of each calendar month from a 12-month centred mean."""
    s = obs.set_index(pd.to_datetime(dict(year=obs.year, month=obs.month, day=15)))["average"]
    detr = s - s.rolling(12, center=True, min_periods=12).mean()
    win = detr[(detr.index.year >= lo) & (detr.index.year <= hi)]
    cyc = win.groupby(win.index.month).mean()
    return cyc - cyc.mean()          # force it to be a pure anomaly (sums to zero)


def deseasonalised(obs: pd.DataFrame, cyc: pd.Series) -> pd.Series:
    """Observed record with the estimated seasonal cycle removed, on decimal years."""
    dec = obs["year"] + (obs["month"] - 0.5) / 12.0
    return pd.Series((obs["average"] - obs["month"].map(cyc)).values, index=dec.values)


def build(repo: Path, crop_dir: Path) -> pd.DataFrame:
    obs = pd.read_csv(crop_dir / "data" / "co2" / SRC_NAME)
    obs = obs.sort_values(["year", "month"]).reset_index(drop=True)

    law = load_law_dome(repo)
    cyc = seasonal_cycle(obs, *SEASON_YEARS)
    des = deseasonalised(obs, cyc)

    # Law Dome annual means represent the year as a whole -> place at mid-year.
    law_x, law_y = law.index.values + 0.5, law.values

    # Constant offset that puts the reconstruction on the Mauna Loa scale.
    ov = des[(des.index >= OFFSET_YEARS[0]) & (des.index < OFFSET_YEARS[1] + 1)]
    offset = float((ov - np.interp(ov.index.values, law_x, law_y)).mean())

    first = obs.iloc[0]
    need = pd.period_range(f"{TARGET_START_YEAR}-01",
                           f"{int(first.year)}-{int(first.month)}", freq="M")[:-1]
    if len(need) == 0:
        return obs.copy()

    dec = np.array([p.year + (p.month - 0.5) / 12.0 for p in need])
    trend = np.interp(dec, law_x, law_y) + offset
    vals = trend + np.array([cyc[p.month] for p in need])

    back = pd.DataFrame({"year": [p.year for p in need],
                         "month": [p.month for p in need],
                         "average": np.round(vals, 2)})
    out = pd.concat([back, obs], ignore_index=True)
    out.attrs["offset"] = offset
    out.attrs["n_reconstructed"] = len(back)
    return out


def verify(df: pd.DataFrame, label: str) -> list[str]:
    """A missing or duplicated (year, month) is a guaranteed NullPointerException."""
    problems = []
    per = pd.PeriodIndex([f"{y}-{m:02d}" for y, m in zip(df.year, df.month)], freq="M")
    if per.duplicated().any():
        problems.append(f"{label}: duplicate months {list(per[per.duplicated()][:5])}")
    if not per.is_monotonic_increasing:
        problems.append(f"{label}: months not sorted")
    full = pd.period_range(per.min(), per.max(), freq="M")
    gaps = full.difference(per)
    if len(gaps):
        problems.append(f"{label}: {len(gaps)} missing months, first {gaps[0]}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(Path(__file__).with_name("experiments.yaml")))
    ap.add_argument("--check", action="store_true",
                    help="report what would be written, change nothing")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    repo = REPO_ROOT
    problems: list[str] = []

    for crop in cfg["crops"]:
        crop_dir = repo / "simplace" / crop
        out_path = crop_dir / "data" / "co2" / OUT_NAME
        df = build(repo, crop_dir)
        problems += verify(df, crop)

        lo = f"{df.year.iloc[0]}-{df.month.iloc[0]:02d}"
        hi = f"{df.year.iloc[-1]}-{df.month.iloc[-1]:02d}"
        note = (f"{crop:16s} {lo} .. {hi}  rows={len(df):,}  "
                f"reconstructed={df.attrs['n_reconstructed']}  "
                f"offset={df.attrs['offset']:+.3f} ppm")
        if args.check:
            print(f"  [would write] {note}")
        else:
            df.to_csv(out_path, index=False, float_format="%.2f")
            print(f"  [written]     {note}  -> {out_path.relative_to(repo)}")

    # Every SSP file must also cover the whole future period with no holes.
    fut = cfg["periods"]["future"]
    for name in ("co2_mm_ssp126_future.csv", "co2_mm_ssp370_future.csv"):
        f = pd.read_csv(repo / "simplace" / cfg["crops"][0] / "data" / "co2" / name)
        problems += verify(f, name)
        if f.year.min() > fut["start"] or f.year.max() < fut["end"]:
            problems.append(f"{name}: covers {f.year.min()}-{f.year.max()}, "
                            f"needs {fut['start']}-{fut['end']}")

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  ! {p}")
        return 1
    print("\nAll CO2 series are gap-free and cover their periods.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
