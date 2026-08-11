"""Local LLM calibration agents.

Three agents, all talking to a local Ollama server and all going through
``optimization/calibrate.py`` for anything that touches a file:

    PhenologyAgent   stage 1 — thermal time against DWD observations
    GrowthAgent      stage 2 — canopy and yield, calibrated jointly
    AnalystAgent     read-only review of a calibration in progress

Driven by ``python optimization/agentic.py``.
"""
from .analyst import AnalystAgent
from .base import AgentError, CalibrationAgent, Decision
from .growth import GrowthAgent
from .llm import LLMError, LLMUnavailable, MockBackend, OllamaBackend, build_backend
from .phenology import PhenologyAgent

#: target -> agent class, for the CLI and the tests
AGENTS = {
    "phenology": PhenologyAgent,
    "growth": GrowthAgent,
}

__all__ = [
    "AGENTS", "AgentError", "AnalystAgent", "CalibrationAgent", "Decision",
    "GrowthAgent", "LLMError", "LLMUnavailable", "MockBackend", "OllamaBackend",
    "PhenologyAgent", "build_backend",
]
