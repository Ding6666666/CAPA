from __future__ import annotations

__all__ = ["CAPA5Config", "CAPA5NotebookRunner", "CAPAExperimentRunner"]


def __getattr__(name: str):
    if name == "CAPA5Config":
        from .config import CAPA5Config
        return CAPA5Config
    if name in {"CAPAExperimentRunner", "CAPA5NotebookRunner"}:
        from .runner import CAPA5NotebookRunner, CAPAExperimentRunner
        return {
            "CAPAExperimentRunner": CAPAExperimentRunner,
            "CAPA5NotebookRunner": CAPA5NotebookRunner,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
