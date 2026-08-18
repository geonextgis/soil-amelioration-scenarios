#!/usr/bin/env python
"""Verification time series for the finished scenario simulations.

Reads the per-experiment annual aggregates produced from
`simplace/<crop>/runs/<exp_id>/out/<exp_id>/yearly/*_yearly.csv` and draws, for
every crop, the three quantities that say fastest whether a run is sane:

    mean yield (t ha-1 DM)   mean max LAI (m2 m-2)   mean harvest date (DOY)

each averaged over the ~3,086 simulated locations, one point per harvest year.

Layout is one figure per crop: rows are the three variables, columns are the
climate scenarios (baseline / historical / future, left to right), and the five
soil amelioration scenarios are the series inside each panel. The y-scale is
shared along a row so panels are comparable at a glance, which is the whole
point of the figure — a climate or soil scenario that sits off the others'
range is the thing being looked for.

Colour: S1 is the *reference* soil and is drawn in neutral ink, not a hue; the
four amelioration variants take categorical slots 1-4 (blue, orange, aqua,
violet), which clear the all-pairs CVD and normal-vision floors for small
multiples. The accompanying CSV is the table view that the aqua slot's
sub-3:1 contrast obliges.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "CLAUDE.md").exists() and (p / "simplace").is_dir():
            return p
    raise SystemExit("repo root not found")


REPO = Path(__file__).resolve()
REPO = _find_repo_root(REPO.parent)
FIGDIR = REPO / "notebooks" / "04_evaluation" / "figures"
TABLE = REPO / "data" / "processed" / "scenario_verification" / "annual_means.csv"

# --- design tokens ---------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"

# S1 is the baseline soil -> neutral ink. The four amelioration variants take
# validated categorical slots (validate_palette.js --pairs all: PASS).
# The five scenarios track each other closely, so S1 is drawn first as a wide
# neutral reference band and the four variants ride on top of it as thin hues:
# where they coincide you see one thin colour centred in grey, where they part
# you see both.
SOIL_STYLE = {
    "S1":  {"color": "#6b6a66", "lw": 2.6, "z": 3, "label": "S1 — BZE baseline (reference)"},
    "S2":  {"color": "#2a78d6", "lw": 1.0, "z": 4, "label": "S2 — amelioration"},
    "S3":  {"color": "#eb6834", "lw": 1.0, "z": 5, "label": "S3 — amelioration"},
    "DL":  {"color": "#1baf7a", "lw": 1.0, "z": 6, "label": "DL — deep loosening"},
    "DLB": {"color": "#4a3aa7", "lw": 1.0, "z": 7, "label": "DLB — deep loosening + B"},
}
SOIL_ORDER = ["S1", "S2", "S3", "DL", "DLB"]

VARIABLES = [
    ("yield_mean",  "Yield", "t ha$^{-1}$ (dry matter)"),
    ("lai_mean",    "Maximum LAI", "m$^2$ m$^{-2}$"),
    ("hdoy_mean",   "Harvest date", "day of year"),
    # Not a model output but the denominator of the three above: SIMPLACE only
    # writes a yearly row when HarvestManagement.DoHarvest fires, so a location
    # whose crop never matured is silently absent from that year's mean. Without
    # this row a rising mean cannot be told apart from a shrinking sample.
    ("n_locations", "Locations harvested", "count, of 3,086"),
]
HDOY_ROW = 2

# Month starts, for reading the harvest-date row as dates.
MONTHS = [(152, "1 Jun"), (182, "1 Jul"), (213, "1 Aug"), (244, "1 Sep"), (274, "1 Oct")]


def climate_group(clim: str) -> str:
    if clim == "DWD":
        return "Baseline (DWD obs.)"
    if clim.endswith("_historical") or clim == "HYRAS_OBS":
        return "Historical"
    return "Future"


def climate_sort_key(clim: str) -> tuple:
    order = {"Baseline (DWD obs.)": 0, "Historical": 1, "Future": 2}
    return (order[climate_group(clim)], clim != "HYRAS_OBS", clim)


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=3)


def draw_crop_figure(df: pd.DataFrame, crop: str, out: Path) -> None:
    climates = sorted(df["climate"].unique(), key=climate_sort_key)
    ncol = len(climates)
    fig, axes = plt.subplots(
        len(VARIABLES), ncol, figsize=(2.35 * ncol + 1.6, 8.2),
        sharey="row", squeeze=False,
    )
    fig.patch.set_facecolor(SURFACE)

    for r, (col, name, unit) in enumerate(VARIABLES):
        for c, clim in enumerate(climates):
            ax = axes[r][c]
            style_axes(ax)
            sub = df[df["climate"] == clim]
            for soil in SOIL_ORDER:
                s = sub[sub["soil"] == soil].sort_values("year")
                if s.empty:
                    continue
                st = SOIL_STYLE[soil]
                ax.plot(s["year"], s[col], color=st["color"], lw=st["lw"],
                        zorder=st["z"], solid_capstyle="round")
            if r == 0:
                ax.set_title(clim, fontsize=8.5, color=INK, pad=14)
            if r == len(VARIABLES) - 1:
                ax.set_xlabel("harvest year", fontsize=8, color=INK_2)
            if c == 0:
                ax.set_ylabel(f"{name}\n({unit})", fontsize=8.5, color=INK)

    # A three-location wobble in the sample row must not autoscale into a cliff:
    # give that row at least a 300-location window so "essentially complete"
    # reads as flat.
    nrow = len(VARIABLES) - 1
    lo, hi = axes[nrow][0].get_ylim()
    if hi - lo < 300:
        mid = (hi + lo) / 2
        lo, hi = min(mid - 150, 3086 - 300), min(max(mid + 150, lo + 300), 3120)
        for ax in axes[nrow]:
            ax.set_ylim(lo, hi)

    # Month reference lines, added after autoscaling so they never widen the row.
    ylim = axes[HDOY_ROW][0].get_ylim()
    for ax in axes[HDOY_ROW]:
        for doy, _ in MONTHS:
            if ylim[0] < doy < ylim[1]:
                ax.axhline(doy, color=GRID, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.set_ylim(ylim)

    # Month names for the harvest-date row, once, on a right-hand axis.
    tw = axes[HDOY_ROW][-1].twinx()
    tw.set_ylim(ylim)
    tw.set_yticks([d for d, _ in MONTHS if ylim[0] < d < ylim[1]],
                  [lab for d, lab in MONTHS if ylim[0] < d < ylim[1]])
    tw.tick_params(colors=INK_2, labelsize=7.5, length=0)
    for side in ("top", "right", "left", "bottom"):
        tw.spines[side].set_visible(False)

    # Group bands over the column titles.
    for c, clim in enumerate(climates):
        grp = climate_group(clim)
        prev = climate_group(climates[c - 1]) if c else None
        if grp != prev:
            axes[0][c].annotate(grp, (0.0, 1.30), xycoords="axes fraction",
                                fontsize=9, color=INK, fontweight="bold")

    handles = [Line2D([], [], color=SOIL_STYLE[s]["color"], lw=SOIL_STYLE[s]["lw"] + 0.4,
                      label=SOIL_STYLE[s]["label"]) for s in SOIL_ORDER]
    fig.legend(handles=handles, loc="lower center", ncols=5, frameon=False,
               fontsize=8.5, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.004))

    lo, hi = int(df["n_locations"].min()), int(df["n_locations"].max())
    n = f"{lo:,}" if lo == hi else f"{lo:,}–{hi:,}"
    fig.suptitle(
        f"{crop.replace('_', ' ')} — annual mean over {n} locations, by climate and soil scenario",
        fontsize=12.5, color=INK, x=0.008, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0.028, 1, 0.955))
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def draw_overview(df: pd.DataFrame, out: Path) -> None:
    """One climate (the DWD baseline), all crops side by side."""
    base = df[df["climate"] == "DWD"]
    crops = sorted(base["crop"].unique())
    fig, axes = plt.subplots(len(VARIABLES), len(crops),
                             figsize=(2.9 * len(crops) + 1.4, 10.0),
                             sharey="row", squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for r, (col, name, unit) in enumerate(VARIABLES):
        for c, crop in enumerate(crops):
            ax = axes[r][c]
            style_axes(ax)
            sub = base[base["crop"] == crop]
            for soil in SOIL_ORDER:
                s = sub[sub["soil"] == soil].sort_values("year")
                if s.empty:
                    continue
                st = SOIL_STYLE[soil]
                ax.plot(s["year"], s[col], color=st["color"], lw=st["lw"], zorder=st["z"])
            if r == 0:
                ax.set_title(crop.replace("_", " "), fontsize=9.5, color=INK)
            if r == len(VARIABLES) - 1:
                ax.set_xlabel("harvest year", fontsize=8, color=INK_2)
            if c == 0:
                ax.set_ylabel(f"{name}\n({unit})", fontsize=8.5, color=INK)
    handles = [Line2D([], [], color=SOIL_STYLE[s]["color"], lw=SOIL_STYLE[s]["lw"] + 0.4,
                      label=SOIL_STYLE[s]["label"]) for s in SOIL_ORDER]
    fig.legend(handles=handles, loc="lower center", ncols=5, frameon=False,
               fontsize=8.5, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.004))
    fig.suptitle("DWD baseline (1978–2023) — all crops, all soil scenarios",
                 fontsize=12.5, color=INK, x=0.008, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0.028, 1, 0.962))
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str]) -> int:
    src = Path(argv[1]) if len(argv) > 1 else TABLE
    df = pd.read_csv(src)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for crop, sub in df.groupby("crop"):
        out = FIGDIR / f"verification_{crop}.png"
        draw_crop_figure(sub, crop, out)
        print(f"  wrote {out.relative_to(REPO)}")
    out = FIGDIR / "verification_overview_baseline.png"
    draw_overview(df, out)
    print(f"  wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
