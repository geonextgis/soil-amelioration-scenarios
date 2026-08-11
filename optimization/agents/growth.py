#!/usr/bin/env python
"""Growth calibration agent — canopy and yield, calibrated jointly.

Stage 2. Starts from the phenology-calibrated ``crop.xml`` (via ``calibrate.py
handoff``) with the thermal-time parameters frozen. One iteration runs both views
— GLASS-LAI and district yield — from the same parameter set and is scored on the
combined objective, so a change that buys yield by wrecking the canopy shows up
immediately in the number the agent is judged on.
"""
from __future__ import annotations

from .base import CalibrationAgent


class GrowthAgent(CalibrationAgent):
    target = "growth"
    agent_key = "growth"
    prompt_file = "growth.md"
