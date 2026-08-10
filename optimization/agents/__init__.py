"""Local LLM calibration agents.

Four agents, one per job, all talking to a local Ollama server and all going
through ``optimization/calibrate.py`` for anything that touches a file:

    PhenologyAgent   validate the optimized phenology; recalibrate on request
    LAIAgent         stage 1 — canopy development against GLASS-LAI
    YieldAgent       stage 2 — district yields, LAI parameters frozen
    AnalystAgent     read-only review of a calibration in progress

Driven by ``python optimization/agentic.py``.
"""
from .analyst import AnalystAgent
from .base import AgentError, CalibrationAgent, Decision
from .lai import LAIAgent
from .llm import LLMError, LLMUnavailable, MockBackend, OllamaBackend, build_backend
from .phenology import PhenologyAgent
from .yield_ import YieldAgent

#: target -> agent class, for the CLI and the tests
AGENTS = {
    "phenology": PhenologyAgent,
    "lai": LAIAgent,
    "yield": YieldAgent,
}

__all__ = [
    "AGENTS", "AgentError", "AnalystAgent", "CalibrationAgent", "Decision",
    "LAIAgent", "LLMError", "LLMUnavailable", "MockBackend", "OllamaBackend",
    "PhenologyAgent", "YieldAgent", "build_backend",
]
