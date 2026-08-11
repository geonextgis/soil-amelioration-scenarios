#!/usr/bin/env python
"""Phenology calibration agent — stage 1, calibrated from scratch.

Owns the thermal-time parameters and nothing else. Its result is frozen for the
joint growth stage, which is seeded from it by ``calibrate.py handoff``.
"""
from __future__ import annotations

from .base import CalibrationAgent


class PhenologyAgent(CalibrationAgent):
    target = "phenology"
    agent_key = "phenology"
    prompt_file = "phenology.md"
