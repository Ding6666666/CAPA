from __future__ import annotations

import os
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")


def configure_runtime() -> None:
    """Apply lightweight runtime guards before heavy scientific imports."""
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    try:
        from sklearn.exceptions import UndefinedMetricWarning
    except Exception:
        UndefinedMetricWarning = None
    if UndefinedMetricWarning is not None:
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


configure_runtime()
