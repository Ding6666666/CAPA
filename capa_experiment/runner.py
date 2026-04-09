from __future__ import annotations

from .config import CAPA5Config
from .mixins import (
    AdaptationMixin,
    CoreMixin,
    EvaluationMixin,
    GuardianMixin,
    ReportingMixin,
    ScoringMixin,
)


class CAPAExperimentRunner(
    CoreMixin,
    GuardianMixin,
    ScoringMixin,
    AdaptationMixin,
    EvaluationMixin,
    ReportingMixin,
):
    """Structured CAPA experiment runner composed from concern-focused mixins."""


CAPA5NotebookRunner = CAPAExperimentRunner

__all__ = ["CAPAExperimentRunner", "CAPA5NotebookRunner", "CAPA5Config"]
