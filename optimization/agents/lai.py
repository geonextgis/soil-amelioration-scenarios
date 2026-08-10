#!/usr/bin/env python
"""LAI calibration agent — canopy development against GLASS-LAI retrievals.

Stage 1 of the calibration order. Runs on the phenology-optimized parameter set
with every phenology parameter frozen, and its result is what the yield stage
starts from (``calibrate.py handoff``).
"""
from __future__ import annotations

from .base import CalibrationAgent


class LAIAgent(CalibrationAgent):
    target = "lai"
    agent_key = "lai"
    prompt_file = "lai.md"
