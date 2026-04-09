from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..runtime import configure_runtime

configure_runtime()

import copy
import os
import pickle
import random
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, auc, f1_score, roc_auc_score, roc_curve
from tqdm import tqdm

class GuardianMixin:
    def _is_go_guardian_enabled(self) -> bool:
        return bool(getattr(self.config, "ENABLE_GO_GUARDIAN", False))

    def _guardian_is_frozen(self) -> bool:
        return str(self.guardian_status) == "frozen"

    def _guardian_hist_from_values(self, values: np.ndarray, bins: np.ndarray) -> np.ndarray:
        hist, _ = np.histogram(values, bins=bins)
        dist = hist.astype(np.float64)
        eps = 1e-6
        dist = (dist + eps) / (dist.sum() + eps * len(dist))
        return dist

    def _compute_psi_from_dists(self, expected_dist: np.ndarray, actual_dist: np.ndarray) -> float:
        e = np.asarray(expected_dist, dtype=np.float64)
        a = np.asarray(actual_dist, dtype=np.float64)
        eps = 1e-6
        return float(np.sum((a - e) * np.log((a + eps) / (e + eps))))

    def _guardian_init_baseline(self):
        if not self._is_go_guardian_enabled():
            return
        vals = np.asarray(self.guardian_baseline_values, dtype=np.float64)
        vals = np.clip(vals, 0.0, 1.0)
        n_bins = max(5, int(getattr(self.config, "GO_PSI_BINS", 10)))
        if vals.size < n_bins:
            return
        bins = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)
        baseline_hist = self._guardian_hist_from_values(vals, bins)

        self.guardian_psi_bin_edges = bins
        self.guardian_psi_baseline_hist = baseline_hist
        self.guardian_last_psi = np.nan
        self.guardian_psi_history = []
        self.guardian_resume_streak = 0
        self.guardian_num_alarms = 0
        self.guardian_last_alarm_step = -1
        self.guardian_window_values = []
        self._log(
            f"[GO] Baseline ready: n={int(vals.size)}, bins={n_bins}, "
            f"psi_thr={float(self.config.GO_PSI_THR):.3f}, resume={float(self.config.GO_TAU_RESUME):.3f}x{int(self.config.GO_RESUME_WINDOWS)}",
            always=True,
        )

    def _guardian_collect_batch_scalars(self, probs: torch.Tensor):
        if (not self._is_go_guardian_enabled()) or probs is None:
            return
        scalar_name = str(getattr(self.config, "GO_GUARDIAN_SCALAR", "top1_conf")).strip().lower()
        if scalar_name == "top1_conf":
            vals = probs.max(dim=1).values.detach().cpu().numpy()
        else:
            vals = probs.max(dim=1).values.detach().cpu().numpy()
        vals = np.clip(np.asarray(vals, dtype=np.float64), 0.0, 1.0)
        if vals.size <= 0:
            return
        max_base = max(128, int(getattr(self.config, "GO_PSI_BASELINE_MAX", 5000)))
        if str(self.guardian_status) == "baseline_collect":
            self.guardian_baseline_values.extend([float(v) for v in vals.tolist()])
            if len(self.guardian_baseline_values) > max_base:
                self.guardian_baseline_values = self.guardian_baseline_values[-max_base:]
        else:
            self.guardian_window_values.extend([float(v) for v in vals.tolist()])

    def _guardian_window_psi(self) -> Optional[float]:
        if self.guardian_psi_baseline_hist is None or self.guardian_psi_bin_edges is None:
            return None
        win = max(16, int(getattr(self.config, "GO_PSI_WINDOW", 512)))
        if len(self.guardian_window_values) < win:
            return None
        curr = np.asarray(self.guardian_window_values[:win], dtype=np.float64)
        self.guardian_window_values = self.guardian_window_values[win:]
        curr_hist = self._guardian_hist_from_values(curr, self.guardian_psi_bin_edges)
        return self._compute_psi_from_dists(self.guardian_psi_baseline_hist, curr_hist)

    def _guardian_update_from_window(self, step: int, probs: torch.Tensor) -> Optional[Dict[str, object]]:
        if not self._is_go_guardian_enabled():
            return None
        warm = max(0, int(getattr(self.config, "GO_WARMUP_STEPS", 50)))
        collect = max(0, int(getattr(self.config, "GO_BASELINE_COLLECT_STEPS", 50)))
        end_collect = warm + collect
        prev_status = str(self.guardian_status)

        if int(step) < warm:
            self.guardian_status = "off"
            return None

        if int(step) < end_collect:
            if prev_status != "baseline_collect":
                self.guardian_baseline_values = []
                self.guardian_window_values = []
                self.guardian_psi_baseline_hist = None
                self.guardian_psi_bin_edges = None
            self.guardian_status = "baseline_collect"
            self._guardian_collect_batch_scalars(probs)
            if int(step) == end_collect - 1:
                self._guardian_init_baseline()
                if self.guardian_psi_baseline_hist is not None:
                    self.guardian_status = "normal"
                    if isinstance(self.current_R, torch.Tensor):
                        self.R_last_good = self.current_R.detach().clone()
            return {"step": int(step), "status": str(self.guardian_status), "phase": "baseline_collect"}

        if self.guardian_psi_baseline_hist is None or self.guardian_psi_bin_edges is None:
            self.guardian_status = "baseline_collect"
            self._guardian_collect_batch_scalars(probs)
            self._guardian_init_baseline()
            return {"step": int(step), "status": str(self.guardian_status), "phase": "baseline_collect_fallback"}

        if str(self.guardian_status) not in ("normal", "frozen"):
            self.guardian_status = "normal"
            self.guardian_resume_streak = 0
            if isinstance(self.current_R, torch.Tensor):
                self.R_last_good = self.current_R.detach().clone()

        self._guardian_collect_batch_scalars(probs)
        psi = self._guardian_window_psi()
        if psi is None:
            return None

        psi = float(psi)
        self.guardian_last_psi = psi
        self.guardian_psi_history.append(psi)
        changed = False
        dry_run = bool(getattr(self.config, "GO_DRY_RUN", False))
        psi_thr = float(getattr(self.config, "GO_PSI_THR", 2.0))
        tau_resume = float(getattr(self.config, "GO_TAU_RESUME", 1.0))
        resume_windows = max(1, int(getattr(self.config, "GO_RESUME_WINDOWS", 3)))

        if self.guardian_status == "normal":
            if psi > psi_thr:
                self.guardian_last_alarm_step = int(step)
                self.guardian_num_alarms += 1
                self.guardian_resume_streak = 0
                if not dry_run:
                    self.guardian_status = "frozen"
                    changed = True
        elif self.guardian_status == "frozen":
            if psi < tau_resume:
                self.guardian_resume_streak += 1
                if self.guardian_resume_streak >= resume_windows:
                    self.guardian_status = "normal"
                    self.guardian_resume_streak = 0
                    changed = True
            else:
                self.guardian_resume_streak = 0
                if psi > psi_thr:
                    self.guardian_last_alarm_step = int(step)
                    self.guardian_num_alarms += 1

        if self._guardian_is_frozen() and isinstance(self.R_last_good, torch.Tensor):
            self.current_R = self.R_last_good.clone()
            self.R_frozen = self.R_last_good.clone()

        return {
            "step": int(step),
            "psi": psi,
            "status": str(self.guardian_status),
            "changed": bool(changed),
            "frozen": bool(self._guardian_is_frozen()),
            "dry_run": bool(dry_run),
        }
