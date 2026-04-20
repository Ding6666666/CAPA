from __future__ import annotations

import runpy


def main() -> int:
    # Keep the package CLI as a thin shim over the vendored standalone mainline
    # so code/ stays aligned with experiment_3line.py without duplicated parsers.
    runpy.run_module("capa_experiment.mainline", run_name="__main__")
    return 0


__all__ = ["main"]
