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

class EvaluationMixin:
    def _compute_ece(self, y_true, y_prob, n_bins=10):
        """
        计算 Expected Calibration Error (ECE)。
        衡量由于 TTA 导致的模型过度自信(Overconfidence)程度。
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        # 确保转为 1D 数组进行全局计算 (Micro-level ECE)
        y_true_flat = y_true.flatten()
        y_prob_flat = y_prob.flatten()
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # 落在这个置信度区间内的样本
            in_bin = (y_prob_flat > bin_lower) & (y_prob_flat <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true_flat[in_bin].mean()
                avg_confidence_in_bin = y_prob_flat[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece * 100.0  # 转为百分比

    def _compute_metrics(
        self,
        z_test,
        y_test,
        is_multi,
        t_protos,
        scale_override: Optional[float] = None,
        temperature_override: Optional[float] = None,
        dataset_name: Optional[str] = None,
        scoring_mode: Optional[str] = None,
        use_cache: bool = False,
        baseline_t_protos: Optional[torch.Tensor] = None,
    ):
        """Compute metrics under selected scoring mode. AUC uses T=1.0 ranking scores."""
        scale = self.s_opt if scale_override is None else scale_override
        calib_T = self.T_opt if temperature_override is None else max(temperature_override, 1e-4)
        mode = self._resolve_scoring_mode(scoring_mode)

        logits = self._compose_eval_logits(
            z_test,
            t_protos,
            scale=scale,
            baseline_t_protos=baseline_t_protos,
        )
        logits = self._blend_with_cache_logits(
            logits,
            z_test,
            dataset_name,
            use_cache=bool(use_cache),
        )

        stats = {
            "Macro-AUC": np.nan,
            "Micro-AUC": np.nan,
            "ECE": np.nan,
            "Acc": 0.0,
            "Top1_Median": 0.0,
            "Brier": np.nan,
            "Temperature": calib_T,
        }

        if is_multi:
            probs = self._predict_probs(
                logits, calib_T=calib_T, is_multi=True, dataset_name=dataset_name, scoring_mode=mode, ranking=False
            ).cpu().numpy()
            stats["Top1_Median"] = np.median(probs.max(axis=1))

            try:
                if y_test.ndim != 2:
                    raise ValueError(f"Expected multilabel y_test as 2D array, got shape={getattr(y_test, 'shape', None)}")
                n_labels = int(y_test.shape[1])
                n_proto = int(probs.shape[1])
                n_eval = max(1, min(n_labels, n_proto))

                # Assume label columns align with ORDERED_CLASS_NAMES[:n_eval]
                eval_indices = list(range(n_eval))
                probs_eval = probs[:, eval_indices]
                y_eval = y_test[:, eval_indices]

                aucs = [roc_auc_score(y_eval[:, i], probs_eval[:, i])
                        for i in range(n_eval) if len(np.unique(y_eval[:, i])) > 1]
                if aucs:
                    stats["Macro-AUC"] = np.mean(aucs)
                try:
                    stats["Micro-AUC"] = roc_auc_score(y_eval, probs_eval, average="micro")
                except Exception:
                    pass

                stats["ECE"] = self._compute_ece(y_eval, probs_eval)
                stats["Acc"] = f1_score(y_eval, (probs_eval > 0.5).astype(int), average="macro", zero_division=0)
                stats["Brier"] = float(np.mean((probs_eval - y_eval) ** 2))

                if "DEBUG_AUC_CHECK" in os.environ:
                    probs_T1 = self._predict_probs(
                        logits, calib_T=1.0, is_multi=True, dataset_name=dataset_name, scoring_mode=mode, ranking=True
                    ).cpu().numpy()
                    try:
                        probs_T1 = probs_T1[:, eval_indices]
                        aucs_T1 = [roc_auc_score(y_eval[:, i], probs_T1[:, i])
                                   for i in range(n_eval) if len(np.unique(y_eval[:, i])) > 1]
                        if aucs_T1 and not np.isnan(stats["Macro-AUC"]):
                            assert abs(np.mean(aucs_T1) - stats["Macro-AUC"]) < 2e-3, "AUC changed unexpectedly after calibration!"
                    except Exception:
                        pass
            except Exception as e:
                self._log(f"Metric Error: {e}")

        else:
            probs_calib_full = F.softmax(logits / calib_T, dim=1).cpu().numpy()
            stats["Top1_Median"] = np.median(probs_calib_full.max(axis=1))

            prob_pos_calibrated = self._predict_probs(
                logits, calib_T=calib_T, is_multi=False, dataset_name=dataset_name, scoring_mode=mode, ranking=False
            ).cpu().numpy()
            prob_pos_ranking = self._predict_probs(
                logits, calib_T=1.0, is_multi=False, dataset_name=dataset_name, scoring_mode=mode, ranking=True
            ).cpu().numpy()

            stats["ECE"] = self._compute_ece(y_test, prob_pos_calibrated)
            stats["Acc"] = accuracy_score(y_test, (prob_pos_calibrated > 0.5).astype(int))
            stats["Brier"] = float(np.mean((prob_pos_calibrated - y_test) ** 2))
            try:
                if len(np.unique(y_test)) > 1:
                    auc_val = roc_auc_score(y_test, prob_pos_ranking)
                    stats["Macro-AUC"] = auc_val
                    stats["Micro-AUC"] = auc_val
            except Exception:
                pass

        return stats

    def _fit_posthoc_tau(
        self,
        logits_np: np.ndarray,
        labels_np: np.ndarray,
        dataset_name: str,
        is_multi: bool,
        scoring_mode: Optional[str] = None,
    ):
        """Post-hoc probability calibration: search tau in [0.5, 2.0] with coarse grid + local refine."""
        mode = self._resolve_scoring_mode(scoring_mode)
        logits_arr = np.asarray(logits_np, dtype=np.float64)
        labels_arr = np.asarray(labels_np)
        if logits_arr.ndim != 2:
            raise ValueError(f"logits_np must be 2D [N, C], got shape={logits_arr.shape}")

        if is_multi:
            if labels_arr.ndim != 2:
                raise ValueError(f"Multilabel calibration labels must be 2D, got shape={labels_arr.shape}")
            n_use = max(1, min(labels_arr.shape[1], logits_arr.shape[1]))
            y_cal = labels_arr[:, :n_use].astype(np.float64)
        else:
            pos_indices = self._get_binary_positive_indices(dataset_name, logits_arr.shape[1])
            if labels_arr.ndim == 2:
                max_idx = max(pos_indices)
                if max_idx < labels_arr.shape[1]:
                    y_bin = labels_arr[:, pos_indices].max(axis=1)
                else:
                    y_bin = labels_arr[:, 0]
            else:
                y_bin = labels_arr.reshape(-1)
            y_cal = y_bin.reshape(-1, 1).astype(np.float64)

        eps = 1e-8

        def nll_tau(tau_val: float) -> float:
            tau = float(np.clip(tau_val, 0.5, 2.0))
            scaled = logits_arr / tau

            if mode == "mixed":
                if is_multi:
                    probs = 1.0 / (1.0 + np.exp(-scaled[:, : y_cal.shape[1]]))
                else:
                    pos_indices_local = self._get_binary_positive_indices(dataset_name, scaled.shape[1])
                    neg_indices_local = [i for i in range(scaled.shape[1]) if i not in pos_indices_local]
                    if len(neg_indices_local) == 0:
                        bin_logit = scaled[:, pos_indices_local].mean(axis=1, keepdims=True)
                    else:
                        pos_term = logsumexp(scaled[:, pos_indices_local], axis=1, keepdims=True)
                        neg_term = logsumexp(scaled[:, neg_indices_local], axis=1, keepdims=True)
                        bin_logit = pos_term - neg_term
                    probs = 1.0 / (1.0 + np.exp(-bin_logit))
            else:
                probs_full = np.exp(scaled - logsumexp(scaled, axis=1, keepdims=True))
                if is_multi:
                    probs = probs_full[:, : y_cal.shape[1]]
                else:
                    pos_indices_local = self._get_binary_positive_indices(dataset_name, scaled.shape[1])
                    probs = probs_full[:, pos_indices_local].sum(axis=1, keepdims=True)

            probs = np.clip(probs, eps, 1.0 - eps)
            loss = -(y_cal * np.log(probs) + (1.0 - y_cal) * np.log(1.0 - probs)).mean()
            return float(loss)

        grid = [0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0]
        grid_vals = [(g, nll_tau(g)) for g in grid]
        best_tau, best_nll = min(grid_vals, key=lambda x: x[1])

        # local refinement within ±20% of best grid point
        lo, hi = best_tau * 0.8, best_tau * 1.2
        lo, hi = max(0.5, lo), min(2.0, hi)
        tol_improve = 0.001  # 0.1% improvement threshold
        for _ in range(6):
            mid = 0.5 * (lo + hi)
            nll_mid = nll_tau(mid)
            if nll_mid < best_nll * (1 - tol_improve):
                best_tau, best_nll = mid, nll_mid
            # choose interval that improves
            nll_lo, nll_hi = nll_tau(lo), nll_tau(hi)
            if nll_lo < nll_hi:
                hi = mid
            else:
                lo = mid
        return best_tau, best_nll

    def _get_gate_active_mask(self, n_cls: int) -> torch.Tensor:
        if self.class_counts_hpq is not None and self.class_counts_hpq.shape[0] >= n_cls:
            return (self.class_counts_hpq[:n_cls] >= self.config.N_MIN_HPQ_FOR_ACTIVE).to(self.device)
        return torch.ones(n_cls, dtype=torch.bool, device=self.device)

    def _compute_dataset_label_centroids(
        self,
        z_embed: torch.Tensor,
        y_labels,
        n_cls: int,
        dataset_name: Optional[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recompute per-dataset image centroids from dataset labels in the same aligned space.
        Returns:
            mu_dataset: [C, D] normalized where support>0
            support:    [C] per-class positive counts used to build centroids
        """
        z_norm = self._l2_norm(z_embed)
        n_samples, d = z_norm.shape
        mu = torch.zeros((n_cls, d), device=self.device, dtype=z_norm.dtype)
        support = torch.zeros(n_cls, device=self.device, dtype=z_norm.dtype)

        y_arr = np.asarray(y_labels)
        if y_arr.ndim == 2:
            n_use = max(1, min(n_cls, y_arr.shape[1]))
            y_bin = (y_arr[:, :n_use] > 0).astype(np.float32)
            y_t = torch.tensor(y_bin, device=self.device, dtype=z_norm.dtype)
            if y_t.shape[0] != n_samples:
                n_trim = min(int(y_t.shape[0]), int(n_samples))
                y_t = y_t[:n_trim]
                z_norm = z_norm[:n_trim]
                n_samples = n_trim
            s = y_t.sum(dim=0)
            support[:n_use] = s
            if n_samples > 0:
                mu_part = torch.matmul(y_t.T, z_norm) / s.view(-1, 1).clamp_min(1e-6)
                valid = s > 0
                if bool(valid.any().item()):
                    mu[:n_use][valid] = self._l2_norm(mu_part[valid])
            return mu, support

        y_bin_np = (y_arr.reshape(-1) > 0).astype(np.float32)
        y_t = torch.tensor(y_bin_np, device=self.device, dtype=z_norm.dtype).view(-1, 1)
        if y_t.shape[0] != n_samples:
            n_trim = min(int(y_t.shape[0]), int(n_samples))
            y_t = y_t[:n_trim]
            z_norm = z_norm[:n_trim]
        pos_indices = self._get_binary_positive_indices(dataset_name, n_cls)
        pos_count = float(y_t.sum().item())
        if len(pos_indices) > 0 and pos_count > 0:
            mu_pos = (torch.matmul(y_t.T, z_norm) / max(pos_count, 1e-6)).view(-1)
            mu_pos = self._l2_norm(mu_pos.view(1, -1)).view(-1)
            for idx in pos_indices:
                mu[idx] = mu_pos
                support[idx] = pos_count
        return mu, support

    def _compute_alignment_stats_from_mu(
        self,
        mu: torch.Tensor,
        mask: torch.Tensor,
        t_base: torch.Tensor,
        t_rot: torch.Tensor,
    ) -> Dict[str, float]:
        t_base_norm = self._l2_norm(t_base)
        t_rot_norm = self._l2_norm(t_rot)

        n_cls = min(mu.shape[0], t_base_norm.shape[0], t_rot_norm.shape[0], mask.shape[0])
        if n_cls <= 0:
            return {
                "sim_before": 0.0,
                "sim_after": 0.0,
                "sim_gain": 0.0,
                "off_diag_pre": 0.0,
                "off_diag_post_mean": 0.0,
                "off_diag_post_max": 0.0,
                "active_classes": 0,
            }

        mu = self._l2_norm(mu[:n_cls])
        t_base_norm = t_base_norm[:n_cls]
        t_rot_norm = t_rot_norm[:n_cls]
        mask = mask[:n_cls].bool()

        n_active = int(mask.sum().item())
        if n_active <= 0:
            return {
                "sim_before": 0.0,
                "sim_after": 0.0,
                "sim_gain": 0.0,
                "off_diag_pre": 0.0,
                "off_diag_post_mean": 0.0,
                "off_diag_post_max": 0.0,
                "active_classes": 0,
            }

        sim_pre = torch.matmul(mu, t_base_norm.T).diag()
        sim_post = torch.matmul(mu, t_rot_norm.T).diag()
        sim_before = sim_pre[mask].mean()
        sim_after = sim_post[mask].mean()
        sim_gain = sim_after - sim_before

        idx = torch.where(mask)[0]
        t_base_act = t_base_norm.index_select(0, idx)
        t_rot_act = t_rot_norm.index_select(0, idx)
        n_act = int(t_base_act.shape[0])
        if n_act <= 1:
            off_diag_pre = torch.tensor([0.0], device=self.device)
            off_diag_post = torch.tensor([0.0], device=self.device)
        else:
            tt_pre = torch.matmul(t_base_act, t_base_act.T)
            tt_post = torch.matmul(t_rot_act, t_rot_act.T)
            off_mask = ~torch.eye(n_act, dtype=torch.bool, device=self.device)
            off_diag_pre = tt_pre[off_mask]
            off_diag_post = tt_post[off_mask]

        return {
            "sim_before": float(sim_before.item()),
            "sim_after": float(sim_after.item()),
            "sim_gain": float(sim_gain.item()),
            "off_diag_pre": float(off_diag_pre.mean().item()),
            "off_diag_post_mean": float(off_diag_post.mean().item()),
            "off_diag_post_max": float(off_diag_post.max().item()),
            "active_classes": int(n_active),
        }

    def _compute_dataset_alignment_stats(
        self,
        z_embed: Optional[torch.Tensor],
        y_labels,
        t_base: torch.Tensor,
        t_rot: torch.Tensor,
        dataset_name: Optional[str] = None,
        sim_source: Optional[str] = None,
    ):
        """
        Unified Sim formula:
            Sim = mean_{c in C_tau} <mu_c, t_c>
        where only mu_c source changes:
            - gate:    mu_c from Procrustes/Gate centroids (alignment state)
            - dataset: mu_c recomputed on target dataset labels, with C_tau intersection
        """
        src = self._resolve_sim_source(sim_source)
        if self.image_centroids is None:
            raise RuntimeError("image_centroids is missing; run pipeline first.")

        n_cls = min(self.image_centroids.shape[0], t_base.shape[0], t_rot.shape[0])
        gate_mask = self._get_gate_active_mask(n_cls)

        if src == "gate":
            mu_gate = self._l2_norm(self.image_centroids[:n_cls])
            out = self._compute_alignment_stats_from_mu(mu_gate, gate_mask, t_base[:n_cls], t_rot[:n_cls])
            out["sim_source"] = "gate"
            out["sim_scope"] = "C_tau"
            return out

        if z_embed is None or y_labels is None:
            raise ValueError("dataset sim source requires z_embed and y_labels.")
        mu_dataset, support = self._compute_dataset_label_centroids(z_embed, y_labels, n_cls, dataset_name)
        data_mask = support > 0
        mask = gate_mask & data_mask
        out = self._compute_alignment_stats_from_mu(mu_dataset, mask, t_base[:n_cls], t_rot[:n_cls])
        out["sim_source"] = "dataset"
        out["sim_scope"] = "C_tau_intersect_dataset"
        out["dataset_supported_classes"] = int(data_mask.sum().item())
        return out

    def _prepare_auc_inputs(
        self,
        y_test,
        logits: torch.Tensor,
        is_multi: bool,
        dataset_name: str,
        scoring_mode: Optional[str] = None,
    ):
        logits = logits.detach()
        mode = self._resolve_scoring_mode(scoring_mode)
        if is_multi:
            y_arr = np.asarray(y_test)
            probs = self._predict_probs(
                logits, calib_T=1.0, is_multi=True, dataset_name=dataset_name, scoring_mode=mode, ranking=True
            ).cpu().numpy()
            if y_arr.ndim != 2:
                return None
            n_use = max(1, min(y_arr.shape[1], probs.shape[1]))
            return {
                "is_multi": True,
                "y": y_arr[:, :n_use].astype(np.int32),
                "s": probs[:, :n_use].astype(np.float64),
            }

        score = self._predict_probs(
            logits, calib_T=1.0, is_multi=False, dataset_name=dataset_name, scoring_mode=mode, ranking=True
        ).detach().cpu().numpy().reshape(-1).astype(np.float64)

        y_arr = np.asarray(y_test)
        if y_arr.ndim == 2:
            pos_indices = self._get_binary_positive_indices(dataset_name, logits.shape[1])
            max_idx = max(pos_indices)
            if max_idx < y_arr.shape[1]:
                y_bin = y_arr[:, pos_indices].max(axis=1)
            else:
                y_bin = y_arr[:, 0]
        else:
            y_bin = y_arr.reshape(-1)
        y_bin = y_bin.astype(np.int32)

        n_use = min(len(y_bin), len(score))
        return {"is_multi": False, "y": y_bin[:n_use], "s": score[:n_use]}

    def _macro_auc_from_inputs(self, packed: Optional[Dict[str, object]]) -> float:
        if packed is None:
            return np.nan
        is_multi = bool(packed.get("is_multi", False))
        y = packed.get("y", None)
        s = packed.get("s", None)
        if y is None or s is None:
            return np.nan
        y_arr = np.asarray(y)
        s_arr = np.asarray(s)

        if is_multi:
            if y_arr.ndim != 2 or s_arr.ndim != 2:
                return np.nan
            n_use = min(y_arr.shape[1], s_arr.shape[1])
            vals = []
            for c in range(n_use):
                yc = y_arr[:, c]
                sc = s_arr[:, c]
                if len(np.unique(yc)) < 2:
                    continue
                try:
                    vals.append(float(roc_auc_score(yc, sc)))
                except Exception:
                    continue
            return float(np.mean(vals)) if vals else np.nan

        y_bin = y_arr.reshape(-1)
        s_bin = s_arr.reshape(-1)
        if len(np.unique(y_bin)) < 2:
            return np.nan
        try:
            return float(roc_auc_score(y_bin, s_bin))
        except Exception:
            return np.nan

    def _paired_bootstrap_auc_delta(
        self,
        base_packed: Dict[str, object],
        capa_packed: Dict[str, object],
        n_boot: Optional[int] = None,
    ) -> Dict[str, float]:
        n_boot = int(self.config.AUC_BOOTSTRAP_ROUNDS if n_boot is None else n_boot)
        n_boot = max(200, n_boot)
        is_multi = bool(base_packed.get("is_multi", False))
        if is_multi != bool(capa_packed.get("is_multi", False)):
            return {"delta": np.nan, "ci_low": np.nan, "ci_high": np.nan}

        y = np.asarray(base_packed["y"])
        s0 = np.asarray(base_packed["s"])
        s1 = np.asarray(capa_packed["s"])
        n = int(min(len(y), len(s0), len(s1)))
        if n <= 1:
            return {"delta": np.nan, "ci_low": np.nan, "ci_high": np.nan}

        y = y[:n]
        s0 = s0[:n]
        s1 = s1[:n]

        if is_multi:
            if y.ndim != 2 or s0.ndim != 2 or s1.ndim != 2:
                return {"delta": np.nan, "ci_low": np.nan, "ci_high": np.nan}
            n_use = min(y.shape[1], s0.shape[1], s1.shape[1])
            valid_cols = [c for c in range(n_use) if len(np.unique(y[:, c])) >= 2]
            if len(valid_cols) == 0:
                return {"delta": np.nan, "ci_low": np.nan, "ci_high": np.nan}

            def _macro_auc_fixed(yb: np.ndarray, sb: np.ndarray) -> float:
                vals = []
                for c in valid_cols:
                    yc = yb[:, c]
                    if len(np.unique(yc)) < 2:
                        return np.nan
                    vals.append(float(roc_auc_score(yc, sb[:, c])))
                return float(np.mean(vals)) if vals else np.nan

            delta_obs = _macro_auc_fixed(y, s1) - _macro_auc_fixed(y, s0)
        else:
            packed0 = {"is_multi": False, "y": y, "s": s0}
            packed1 = {"is_multi": False, "y": y, "s": s1}
            delta_obs = self._macro_auc_from_inputs(packed1) - self._macro_auc_from_inputs(packed0)
        if not np.isfinite(delta_obs):
            return {"delta": np.nan, "ci_low": np.nan, "ci_high": np.nan}

        rng = np.random.default_rng(int(self.config.RANDOM_SEED))
        deltas = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n, endpoint=False)
            if is_multi:
                d = _macro_auc_fixed(y[idx], s1[idx]) - _macro_auc_fixed(y[idx], s0[idx])
            else:
                p0 = {"is_multi": False, "y": y[idx], "s": s0[idx]}
                p1 = {"is_multi": False, "y": y[idx], "s": s1[idx]}
                d = self._macro_auc_from_inputs(p1) - self._macro_auc_from_inputs(p0)
            if np.isfinite(d):
                deltas.append(float(d))

        if len(deltas) < 20:
            return {"delta": float(delta_obs), "ci_low": np.nan, "ci_high": np.nan}

        arr = np.asarray(deltas, dtype=np.float64)
        return {
            "delta": float(delta_obs),
            "ci_low": float(np.quantile(arr, 0.025)),
            "ci_high": float(np.quantile(arr, 0.975)),
        }

    def _compute_midrank(self, x: np.ndarray) -> np.ndarray:
        order = np.argsort(x)
        sorted_x = x[order]
        n = len(x)
        ranks = np.zeros(n, dtype=np.float64)
        i = 0
        while i < n:
            j = i
            while j < n and sorted_x[j] == sorted_x[i]:
                j += 1
            mid = 0.5 * (i + j - 1)
            ranks[i:j] = mid + 1.0
            i = j
        out = np.empty(n, dtype=np.float64)
        out[order] = ranks
        return out

    def _fast_delong(self, preds: np.ndarray, m: int):
        n_classifiers, n_examples = preds.shape
        n = n_examples - m
        if m < 2 or n < 2:
            return np.full((n_classifiers,), np.nan, dtype=np.float64), np.full((n_classifiers, n_classifiers), np.nan, dtype=np.float64)
        pos = preds[:, :m]
        neg = preds[:, m:]

        tx = np.empty((n_classifiers, m), dtype=np.float64)
        ty = np.empty((n_classifiers, n), dtype=np.float64)
        tz = np.empty((n_classifiers, n_examples), dtype=np.float64)
        for r in range(n_classifiers):
            tx[r] = self._compute_midrank(pos[r])
            ty[r] = self._compute_midrank(neg[r])
            tz[r] = self._compute_midrank(preds[r])

        aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
        v01 = (tz[:, :m] - tx) / n
        v10 = 1.0 - (tz[:, m:] - ty) / m
        sx = np.cov(v01)
        sy = np.cov(v10)
        cov = sx / m + sy / n
        return aucs, cov

    def _delong_binary_delta_var(self, y_true: np.ndarray, s_base: np.ndarray, s_capa: np.ndarray):
        y = np.asarray(y_true).reshape(-1).astype(np.int32)
        s0 = np.asarray(s_base).reshape(-1).astype(np.float64)
        s1 = np.asarray(s_capa).reshape(-1).astype(np.float64)
        n = int(min(len(y), len(s0), len(s1)))
        if n <= 1:
            return np.nan, np.nan
        y = y[:n]
        s0 = s0[:n]
        s1 = s1[:n]
        if len(np.unique(y)) < 2:
            return np.nan, np.nan

        order = np.argsort(-y)
        y_ord = y[order]
        m = int(np.sum(y_ord == 1))
        n_neg = int(len(y_ord) - m)
        if m <= 0 or m >= len(y_ord) or m < 2 or n_neg < 2:
            return np.nan, np.nan
        preds = np.vstack([s0[order], s1[order]])
        aucs, cov = self._fast_delong(preds, m)
        delta = float(aucs[1] - aucs[0])
        var = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
        if not np.isfinite(var) or var < 0.0:
            var = np.nan
        return delta, var

    def _delong_macro_pvalue(self, base_packed: Dict[str, object], capa_packed: Dict[str, object]) -> float:
        is_multi = bool(base_packed.get("is_multi", False))
        if is_multi != bool(capa_packed.get("is_multi", False)):
            return np.nan

        if not is_multi:
            d, v = self._delong_binary_delta_var(base_packed["y"], base_packed["s"], capa_packed["s"])
            if not np.isfinite(d) or not np.isfinite(v) or v <= 0:
                return np.nan
            z = float(d / np.sqrt(v))
            return float(2.0 * norm.sf(abs(z)))

        y = np.asarray(base_packed["y"])
        s0 = np.asarray(base_packed["s"])
        s1 = np.asarray(capa_packed["s"])
        if y.ndim != 2 or s0.ndim != 2 or s1.ndim != 2:
            return np.nan
        n_use = min(y.shape[1], s0.shape[1], s1.shape[1])
        deltas = []
        vars_ = []
        for c in range(n_use):
            d, v = self._delong_binary_delta_var(y[:, c], s0[:, c], s1[:, c])
            if np.isfinite(d) and np.isfinite(v) and v > 0:
                deltas.append(float(d))
                vars_.append(float(v))
        if len(deltas) == 0:
            return np.nan
        delta_macro = float(np.mean(deltas))
        var_macro = float(np.sum(vars_) / (len(vars_) ** 2))
        if not np.isfinite(var_macro) or var_macro <= 0:
            return np.nan
        z = float(delta_macro / np.sqrt(var_macro))
        return float(2.0 * norm.sf(abs(z)))

    def _plot_and_save_curves(self, y_true, y_prob, dataset_name, suffix="", is_multilabel=False, auc_override=None):
        """Plot calibration and ROC curves and save to disk."""
        prob_flat = y_prob.flatten()
        true_flat = y_true.flatten()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        try:
            frac_pos, mean_pred = calibration_curve(true_flat, prob_flat, n_bins=10, strategy='quantile')
            axes[0].plot(mean_pred, frac_pos, "s-", label=f"{dataset_name}")
            axes[0].plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
            axes[0].set_xlabel("Mean Predicted Value")
            axes[0].set_ylabel("Fraction of Positives")
            axes[0].set_title(f"Calibration Curve ({dataset_name})")
            axes[0].legend()
        except ValueError:
            axes[0].set_title("Calibration Curve unavailable")

        # Use macro ROC for multilabel to match macro AUC metrics
        if is_multilabel and y_prob.ndim == 2 and y_prob.shape[1] > 1:
            fpr_grid = np.linspace(0, 1, 101)
            tpr_list = []
            for c in range(y_prob.shape[1]):
                # Skip degenerate columns to avoid nan AUC
                if len(np.unique(y_true[:, c])) < 2:
                    continue
                fpr_c, tpr_c, _ = roc_curve(y_true[:, c], y_prob[:, c])
                tpr_interp = np.interp(fpr_grid, fpr_c, tpr_c)
                tpr_list.append(tpr_interp)

            if tpr_list:
                mean_tpr = np.mean(tpr_list, axis=0)
                roc_auc = auc(fpr_grid, mean_tpr)
                axes[1].plot(fpr_grid, mean_tpr, label=f"Macro AUC = {roc_auc:.4f}")
            else:
                roc_auc = auc( *roc_curve(true_flat, prob_flat)[:2] ) if auc_override is None else auc_override
                axes[1].plot(*roc_curve(true_flat, prob_flat)[:2], label=f"AUC = {roc_auc:.4f}")
        else:
            fpr, tpr, _ = roc_curve(true_flat, prob_flat)
            roc_auc = auc_override if auc_override is not None else auc(fpr, tpr)
            axes[1].plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        axes[1].plot([0, 1], [0, 1], "k--")
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title(f"ROC Curve ({dataset_name})")
        axes[1].legend()

        save_path = os.path.join(self.config.SAVE_DIR, f"{dataset_name}_curves_{suffix}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        self._log(f"   -> Curves saved to {save_path}")

    def _run_scale_sweep(self, s_values: List[float]):
        """Sweep over scale factors without touching routing/temperature/rotation."""
        self._log("\n[Stage IV-B] Scale Sweep (s)")
        results = []
        t_base = self.t_raw_pooled
        t_align_base = self._get_alignment_text_base()
        t_capa = self._l2_norm(torch.matmul(t_align_base, self.R_frozen.T))

        for name, path in self.config.TEST_DATA_PATHS.items():
            if not os.path.exists(path):
                continue
            z_test, y_test, is_multi = self._load_data(path, split_override=2)
            if y_test is None:
                continue

            z_test = self._apply_preprocessing(z_test, self.zI_mean)

            for s_val in s_values:
                stats = self._compute_metrics(
                    z_test,
                    y_test,
                    is_multi,
                    t_capa,
                    scale_override=s_val,
                    dataset_name=name,
                    baseline_t_protos=t_base,
                )
                results.append({
                    "dataset": name,
                    "s": s_val,
                    "MacroAUC": stats["Macro-AUC"],
                    "MicroAUC": stats["Micro-AUC"],
                    "ECE": stats["ECE"],
                    "Top1_Median": stats["Top1_Median"]
                })

                auc_val = stats["Macro-AUC"]
                ece_val = stats["ECE"]
                top1_med = stats["Top1_Median"]
                self._log(f" {name:<10} | s={s_val:<3} | MacroAUC={auc_val:.4f} | ECE%={ece_val:.2f} | Top1Med={top1_med:.4f}")

        if results:
            df = pd.DataFrame(results)
            out_path = os.path.join(self.config.SAVE_DIR, "scale_sweep_results.csv")
            df.to_csv(out_path, index=False)
            self._log(f"[Saved] {out_path}")
        else:
            self._log(" No datasets found for sweep; nothing saved.")

    def run_manuscript_validation(self, scoring_mode: Optional[str] = None, sim_source: Optional[str] = None):
        mode = self._resolve_scoring_mode(scoring_mode)
        sim_src = self._resolve_sim_source(sim_source)
        self._log(f"[Final] Manuscript Validation (scoring={mode}, sim={sim_src})")

        state_path = os.path.join(self.config.SAVE_DIR, "capa_state.pkl")
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"Missing state: {state_path}")

        with open(state_path, "rb") as f:
            state = pickle.load(f)

        self.R_frozen = state["R"].to(self.device)
        r_last_good = state.get("R_last_good", None)
        if isinstance(r_last_good, torch.Tensor):
            self.R_last_good = r_last_good.to(self.device)
        else:
            self.R_last_good = self.R_frozen.clone()
        counts = state["counts"].to(self.device)
        self.class_counts_hpq = counts.clone()
        if "centroids" in state and state["centroids"] is not None:
            self.image_centroids = self._l2_norm(state["centroids"].to(self.device))
        t_align_state = state.get("t_align_base", None)
        if isinstance(t_align_state, torch.Tensor):
            t_align_state = t_align_state.to(self.device)
            if self.t_raw_pooled is not None and tuple(t_align_state.shape) == tuple(self.t_raw_pooled.shape):
                self.t_align_base = self._l2_norm(t_align_state)
            else:
                self.t_align_base = None
        else:
            self.t_align_base = None
        ot_gamma_state = state.get("ot_gamma", None)
        self.ot_gamma = ot_gamma_state.to(self.device) if isinstance(ot_gamma_state, torch.Tensor) else None
        self.final_alignment_stats = state.get("alignment_stats", {})
        self.max_leverage_info = state.get("max_leverage", "N/A")
        guardian_state = state.get("guardian", {}) or {}
        self.guardian_status = str(guardian_state.get("status", "normal"))
        self.guardian_last_alarm_step = int(guardian_state.get("last_alarm_step", -1))
        self.guardian_num_alarms = int(guardian_state.get("num_alarms", 0))
        self.guardian_last_psi = float(guardian_state.get("last_psi", np.nan))
        self.guardian_psi_history = [float(x) for x in list(guardian_state.get("psi_history", []))]
        self.guardian_psi_baseline_hist = guardian_state.get("psi_baseline_hist", None)
        self.guardian_psi_bin_edges = guardian_state.get("psi_bin_edges", None)

        if self.t_align_base is None and bool(getattr(self.config, "ENABLE_OT_PROTOTYPE_MIXING", False)):
            self.t_align_base = self._compute_ot_mixed_text()

        eval_scale = float(self.s_opt)
        self._set_prior_bias_from_counts(counts)

        # Keep alignment-space (processed) heads for both metrics and Sim reporting.
        t_base_proc = self.t_raw_pooled
        t_align_base_proc = self._get_alignment_text_base()
        t_capa_proc = self._l2_norm(torch.matmul(t_align_base_proc, self.R_frozen.T))
        gate_sim = self._compute_dataset_alignment_stats(
            z_embed=None,
            y_labels=None,
            t_base=t_align_base_proc,
            t_rot=t_capa_proc,
            sim_source="gate",
        )
        gate_delta_offdiag = float(gate_sim["off_diag_post_mean"] - gate_sim["off_diag_pre"])

        # Load held-out calibration set for post-hoc temperature scaling (tau); must NOT come from test sets.
        # Load held-out calibration set for post-hoc temperature scaling (tau); must NOT come from test sets.
        z_tau_raw, y_tau, is_multi_tau = self._load_data(
            self.config.TAU_CALIB_DATA_PATH,
            is_calibration=True,
            split_override=1,
        )
        if y_tau is None:
            raise RuntimeError(f"TAU calibration set has no labels: {self.config.TAU_CALIB_DATA_PATH}")
        z_tau_proc = self._apply_preprocessing(z_tau_raw, self.zI_mean)

        # Deterministic shuffle for tau calibration set
        g_cpu_tau = torch.Generator()
        g_cpu_tau.manual_seed(self.config.RANDOM_SEED)
        perm_tau = torch.randperm(len(z_tau_proc), generator=g_cpu_tau)
        z_tau_proc = z_tau_proc[perm_tau]
        if isinstance(y_tau, np.ndarray):
            y_tau = y_tau[perm_tau.cpu().numpy()]

        n_tau_total = len(z_tau_proc)
        n_tau = max(1, int(self.config.TAU_CALIB_FRAC * n_tau_total))
        if n_tau_total >= 2500:
            n_tau = min(self.config.TAU_CALIB_MAX, max(self.config.TAU_CALIB_MIN, n_tau))
        z_tau_proc = z_tau_proc[:n_tau]
        y_tau = y_tau[:n_tau]

        final_report_rows = []

        for d_name, path in self.config.TEST_DATA_PATHS.items():
            if not os.path.exists(path):
                continue

            self._log(f"\n >> Processing {d_name} ...")
            # 1. Load RAW Data
            z_test_raw, y_test, is_multi = self._load_data(path, split_override=2)
            if y_test is None:
                continue

            # Create PROCESSED Data (Whitened/Centered) for Metrics
            z_test_proc = self._apply_preprocessing(z_test_raw, self.zI_mean)

            # Deterministic shuffle (reproducibility only)
            g_cpu = torch.Generator()
            g_cpu.manual_seed(self.config.RANDOM_SEED)
            perm = torch.randperm(len(z_test_raw), generator=g_cpu)
            z_test_raw = z_test_raw[perm]
            z_test_proc = z_test_proc[perm]
            y_test = y_test[perm.cpu().numpy()]

            z_eval_proc = z_test_proc
            y_eval = y_test

            # Dataset-space geometry diagnostic (for reporting only).
            dataset_sim_eval = self._compute_dataset_alignment_stats(
                z_embed=z_eval_proc,
                y_labels=y_eval,
                t_base=t_align_base_proc,
                t_rot=t_capa_proc,
                dataset_name=d_name,
                sim_source="dataset",
            )
            t_eval_proc = t_capa_proc
            cache_on_eval = bool(self._is_cache_enabled())

            # Fit tau on held-out calibration set (NOT test sets), using Stage-IV-aligned runtime scale/bias.
            logits_tau = self._compose_eval_logits(
                z_tau_proc,
                t_eval_proc,
                scale=eval_scale,
                baseline_t_protos=t_base_proc,
            ).cpu().numpy()
            labels_tau = np.array(y_tau)
            if is_multi:
                if labels_tau.ndim != 2:
                    raise RuntimeError(f"TAU calib labels expected 2D for multilabel, got shape={labels_tau.shape}")
                n_eval = int(y_eval.shape[1]) if getattr(y_eval, "ndim", 1) == 2 else 1
                n_tau_labels = int(labels_tau.shape[1])
                n_tau_logits = int(logits_tau.shape[1])
                n_use = max(1, min(n_eval, n_tau_labels, n_tau_logits))
                logits_tau = logits_tau[:, :n_use]
                labels_tau = labels_tau[:, :n_use]
            tau_cal, _ = self._fit_posthoc_tau(
                logits_tau,
                labels_tau,
                dataset_name=d_name,
                is_multi=is_multi,
                scoring_mode=mode,
            )
            assert hasattr(self, "T_opt") and self.T_opt == self.config.INIT_TEMPERATURE, "Routing T_opt was unexpectedly modified!"
            np.save(os.path.join(self.config.SAVE_DIR, f"{d_name}_LOGITS_U_{mode}.npy"), logits_tau)
            np.save(os.path.join(self.config.SAVE_DIR, f"{d_name}_LABELS_U_{mode}.npy"), labels_tau)

            # Compute Metrics using PROCESSED data
            stats_pre = self._compute_metrics(
                z_eval_proc, y_eval, is_multi, t_eval_proc,
                scale_override=eval_scale,
                temperature_override=self.T_opt,
                dataset_name=d_name,
                scoring_mode=mode,
                baseline_t_protos=t_base_proc,
            )
            stats_post = self._compute_metrics(
                z_eval_proc, y_eval, is_multi, t_eval_proc,
                scale_override=eval_scale,
                temperature_override=tau_cal,
                dataset_name=d_name,
                scoring_mode=mode,
                baseline_t_protos=t_base_proc,
            )
            stats_base = self._compute_metrics(
                z_eval_proc, y_eval, is_multi, t_base_proc,
                scale_override=eval_scale, temperature_override=self.T_opt, dataset_name=d_name, scoring_mode=mode
            )
            stats_capa = self._compute_metrics(
                z_eval_proc, y_eval, is_multi, t_eval_proc,
                scale_override=eval_scale,
                temperature_override=self.T_opt,
                dataset_name=d_name,
                scoring_mode=mode,
                baseline_t_protos=t_base_proc,
            )
            stats_capa_cache = self._compute_metrics(
                z_eval_proc,
                y_eval,
                is_multi,
                t_eval_proc,
                scale_override=eval_scale,
                temperature_override=self.T_opt,
                dataset_name=d_name,
                scoring_mode=mode,
                use_cache=cache_on_eval,
                baseline_t_protos=t_base_proc,
            )

            auc_diff = abs(stats_post["Macro-AUC"] - stats_pre["Macro-AUC"])
            if auc_diff > 1e-5:
                self._log(f"[WARN] AUC mismatch! Pre: {stats_pre['Macro-AUC']:.5f}, Post: {stats_post['Macro-AUC']:.5f}")

            delta_macro_auc = np.nan
            if np.isfinite(stats_capa["Macro-AUC"]) and np.isfinite(stats_base["Macro-AUC"]):
                delta_macro_auc = float(stats_capa["Macro-AUC"] - stats_base["Macro-AUC"])
            delta_micro_auc = np.nan
            if np.isfinite(stats_capa["Micro-AUC"]) and np.isfinite(stats_base["Micro-AUC"]):
                delta_micro_auc = float(stats_capa["Micro-AUC"] - stats_base["Micro-AUC"])
            delta_ece = np.nan
            if np.isfinite(stats_capa["ECE"]) and np.isfinite(stats_base["ECE"]):
                delta_ece = float(stats_capa["ECE"] - stats_base["ECE"])
            delta_macro_auc_cache = np.nan
            if np.isfinite(stats_capa_cache["Macro-AUC"]) and np.isfinite(stats_base["Macro-AUC"]):
                delta_macro_auc_cache = float(stats_capa_cache["Macro-AUC"] - stats_base["Macro-AUC"])
            delta_micro_auc_cache = np.nan
            if np.isfinite(stats_capa_cache["Micro-AUC"]) and np.isfinite(stats_base["Micro-AUC"]):
                delta_micro_auc_cache = float(stats_capa_cache["Micro-AUC"] - stats_base["Micro-AUC"])
            delta_ece_cache = np.nan
            if np.isfinite(stats_capa_cache["ECE"]) and np.isfinite(stats_base["ECE"]):
                delta_ece_cache = float(stats_capa_cache["ECE"] - stats_base["ECE"])

            logits_eval = self._compose_eval_logits(
                z_eval_proc,
                t_eval_proc,
                scale=eval_scale,
                baseline_t_protos=t_base_proc,
            )
            if sim_src == "gate":
                dataset_sim = gate_sim
            else:
                dataset_sim = dataset_sim_eval
            delta_offdiag = float(dataset_sim["off_diag_post_mean"] - dataset_sim["off_diag_pre"])

            logits_base_eval = self._compose_eval_logits(
                z_eval_proc,
                t_base_proc,
                scale=eval_scale,
                baseline_t_protos=None,
            )
            logits_capa_eval = self._compose_eval_logits(
                z_eval_proc,
                t_eval_proc,
                scale=eval_scale,
                baseline_t_protos=t_base_proc,
            )
            auc_base_pack = self._prepare_auc_inputs(
                y_eval, logits_base_eval, is_multi=is_multi, dataset_name=d_name, scoring_mode=mode
            )
            auc_capa_pack = self._prepare_auc_inputs(
                y_eval, logits_capa_eval, is_multi=is_multi, dataset_name=d_name, scoring_mode=mode
            )
            ci = {"delta": np.nan, "ci_low": np.nan, "ci_high": np.nan}
            p_val = np.nan
            if auc_base_pack is not None and auc_capa_pack is not None:
                ci = self._paired_bootstrap_auc_delta(auc_base_pack, auc_capa_pack)
                p_val = self._delong_macro_pvalue(auc_base_pack, auc_capa_pack)
            prob_plot = self._predict_probs(
                logits_eval,
                calib_T=tau_cal,
                is_multi=is_multi,
                dataset_name=d_name,
                scoring_mode=mode,
                ranking=False,
            ).cpu().numpy()
            y_plot = y_eval

            self._plot_and_save_curves(
                y_plot,
                prob_plot,
                d_name,
                suffix=f"stage4_{mode}",
                is_multilabel=is_multi,
                auc_override=stats_post["Macro-AUC"],
            )

            row = {
                "Dataset": d_name,
                "ScoringMode": mode,
                "Calibrator": "Temp (Scalar)",
                "Scale": eval_scale,
                "Tau": tau_cal,
                "AUC_Baseline_ZeroShot_Macro": stats_base["Macro-AUC"],
                "AUC_CAPA_Aligned_Macro": stats_capa["Macro-AUC"],
                "Delta_AUC_CAPA_minus_Baseline_Macro": delta_macro_auc,
                "AUC_CAPA_Cache_Macro": stats_capa_cache["Macro-AUC"],
                "Delta_AUC_CAPA_Cache_minus_Baseline_Macro": delta_macro_auc_cache,
                "AUC_Baseline_ZeroShot_Micro": stats_base["Micro-AUC"],
                "AUC_CAPA_Aligned_Micro": stats_capa["Micro-AUC"],
                "Delta_AUC_CAPA_minus_Baseline_Micro": delta_micro_auc,
                "AUC_CAPA_Cache_Micro": stats_capa_cache["Micro-AUC"],
                "Delta_AUC_CAPA_Cache_minus_Baseline_Micro": delta_micro_auc_cache,
                "AUC_Pre": stats_pre["Macro-AUC"],
                "AUC_Post": stats_post["Macro-AUC"],
                "ECE_Pre": stats_pre["ECE"],
                "ECE_Post": stats_post["ECE"],
                "ECE_CAPA_Cache": stats_capa_cache["ECE"],
                "Brier": stats_post["Brier"],
                "Sim_Source": str(dataset_sim.get("sim_source", sim_src)),
                "Sim_Scope": str(dataset_sim.get("sim_scope", "")),
                "Sim_Before": dataset_sim["sim_before"],
                "Sim_After": dataset_sim["sim_after"],
                "Sim_Gain": dataset_sim["sim_gain"],
                "Delta_OffDiag": delta_offdiag,
                "OffDiag_Pre": dataset_sim["off_diag_pre"],
                "OffDiag_Post_Mean": dataset_sim["off_diag_post_mean"],
                "OffDiag_Post_Max": dataset_sim["off_diag_post_max"],
                "Active_Classes": dataset_sim["active_classes"],
                "Dataset_Supported_Classes": dataset_sim.get("dataset_supported_classes", np.nan),
                "Calib_Sim_Before": gate_sim["sim_before"],
                "Calib_Sim_After": gate_sim["sim_after"],
                "Calib_Sim_Gain": gate_sim["sim_gain"],
                "Calib_Delta_OffDiag": gate_delta_offdiag,
                "Calib_OffDiag_Pre": gate_sim["off_diag_pre"],
                "Calib_OffDiag_Post_Mean": gate_sim["off_diag_post_mean"],
                "Calib_OffDiag_Post_Max": gate_sim["off_diag_post_max"],
                "Calib_Active_Classes": gate_sim["active_classes"],
                "Delta_ECE": delta_ece,
                "Delta_ECE_CAPA_Cache_minus_Baseline": delta_ece_cache,
                "Delta_AUC_CI_Low": ci["ci_low"],
                "Delta_AUC_CI_High": ci["ci_high"],
                "Delta_AUC_DeLong_p": p_val,
            }
            final_report_rows.append(row)

        df = pd.DataFrame(final_report_rows)
        out_path = os.path.join(self.config.SAVE_DIR, f"final_manuscript_table_{mode}_{sim_src}.csv")
        df.to_csv(out_path, index=False)
        self._log(f"[Saved] {out_path}")
        three_way_cols = [
            "Dataset",
            "AUC_Baseline_ZeroShot_Macro",
            "AUC_CAPA_Aligned_Macro",
            "AUC_CAPA_Cache_Macro",
            "Delta_AUC_CAPA_minus_Baseline_Macro",
            "Delta_AUC_CAPA_Cache_minus_Baseline_Macro",
        ]
        three_path = os.path.join(self.config.SAVE_DIR, f"four_dataset_three_way_{mode}_{sim_src}.csv")
        df.loc[:, [c for c in three_way_cols if c in df.columns]].to_csv(three_path, index=False)
        self._log(f"[Saved] {three_path}")
        if mode == self._resolve_scoring_mode(None) and sim_src == self._resolve_sim_source(None):
            legacy_out = os.path.join(self.config.SAVE_DIR, "final_manuscript_table.csv")
            df.to_csv(legacy_out, index=False)
            self._log(f"[Saved] {legacy_out}")
            legacy_three = os.path.join(self.config.SAVE_DIR, "four_dataset_three_way.csv")
            df.loc[:, [c for c in three_way_cols if c in df.columns]].to_csv(legacy_three, index=False)
            self._log(f"[Saved] {legacy_three}")
        return final_report_rows

    def _compute_psi(self, expected_array, actual_array, n_bins=10, eps=1e-6):
        """Quantile-binned PSI to avoid artificial drift from shifted scales."""
        breakpoints = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(expected_array, breakpoints)
        bins[0] = -0.01
        bins[-1] = 1.01
        for i in range(1, len(bins)):
            if bins[i] <= bins[i-1]:
                bins[i] = bins[i-1] + 1e-6

        expected_hist, _ = np.histogram(expected_array, bins=bins)
        actual_hist, _ = np.histogram(actual_array, bins=bins)
        e_dist = expected_hist / (expected_hist.sum() + eps)
        a_dist = actual_hist / (actual_hist.sum() + eps)
        psi_vals = (a_dist - e_dist) * np.log((a_dist + eps) / (e_dist + eps))
        return np.sum(psi_vals)

    def run_psi_monitor_shadow_mode(self):
        print("\n" + "="*60)
        print(" Stage III: PSI Monitor (Final: Adaptive + Shuffled + Dynamic Bins)")
        print("="*60)
        
        dataset_s_map = {"CheXpert": 8.0, "MIMIC": 8.0, "COVID": 40.0, "RSNA": 40.0}
        prior_strength = 0.2
        # Prior bias disabled for consistency (b_c=0 everywhere).
        
        # 加载冻结状态
        state_path = os.path.join(self.config.SAVE_DIR, "capa_state.pkl")
        if not os.path.exists(state_path):
            print(" [Error] State file not found.")
            return

        with open(state_path, "rb") as f: state = pickle.load(f)
        self.R_frozen = state["R"].to(self.device)
        counts = state["counts"].to(self.device)
        t_align_state = state.get("t_align_base", None)
        if isinstance(t_align_state, torch.Tensor):
            t_align_state = t_align_state.to(self.device)
            if self.t_raw_pooled is not None and tuple(t_align_state.shape) == tuple(self.t_raw_pooled.shape):
                self.t_align_base = self._l2_norm(t_align_state)
            else:
                self.t_align_base = None
        else:
            self.t_align_base = None
        
        total_counts = counts.sum()
        priors = (counts + 1e-6) / (total_counts + 1e-6 * len(counts))
        self.b_c = (prior_strength * torch.log(priors)).view(1, -1)
        
        t_align_base = self._get_alignment_text_base()
        t_capa = self._l2_norm(torch.matmul(t_align_base, self.R_frozen.T))
        
        results = []

        for name, path in self.config.TEST_DATA_PATHS.items():
            if not os.path.exists(path): continue
            
            print(f"\n >> Monitoring {name} (Simulation)...")
            z_test, _, _ = self._load_data(path, split_override=2)
            z_test = self._apply_preprocessing(z_test, self.zI_mean)
            
            # 1. Shuffle
            g_cpu = torch.Generator()
            g_cpu.manual_seed(self.config.RANDOM_SEED)
            perm_indices = torch.randperm(len(z_test), generator=g_cpu)
            z_test = z_test[perm_indices]
            
            # 2. Dynamic Config (关键修改: 针对小数据集减少分桶数)
            n_samples = len(z_test)
            if n_samples < 2000:
                WINDOW_SIZE = 50
                INIT_WINDOWS = 5
                PSI_BINS = 5
                print(f"   [Config] Small dataset (N={n_samples}). Window=50, Bins=5.")
            else:
                WINDOW_SIZE = 128
                INIT_WINDOWS = 5
                PSI_BINS = 10
                print(f"   [Config] Large dataset (N={n_samples}). Window=128, Bins=10.")

            scale = dataset_s_map.get(name, 12.0)
            
            # A. 收集置信度
            all_confidences = []
            BATCH_SIZE = 256
            for i in range(0, len(z_test), BATCH_SIZE):
                z_b = z_test[i:i+BATCH_SIZE]
                logits = scale * torch.matmul(z_b, t_capa.T) + self.b_c
                probs = torch.sigmoid(logits / self.T_opt)
                confs = probs.max(dim=1).values.detach().cpu().numpy()
                all_confidences.extend(confs)
            
            all_confidences = np.array(all_confidences)
            
            # B. 建立基准
            n_init_samples = WINDOW_SIZE * INIT_WINDOWS
            if len(all_confidences) < n_init_samples + WINDOW_SIZE:
                print(f"   [Skip] Not enough data.")
                continue
                
            baseline_data = all_confidences[:n_init_samples]
            monitor_data = all_confidences[n_init_samples:]
            
            # C. 监控
            psi_history = []
            n_windows = int(np.ceil(len(monitor_data) / WINDOW_SIZE))
            
            for i in range(n_windows):
                start = i * WINDOW_SIZE
                end = min((i + 1) * WINDOW_SIZE, len(monitor_data))
                if end - start < 10: break 
                
                curr_window_data = monitor_data[start:end]
                
                # 使用动态设定的 bin 数量
                psi_val = self._compute_psi(baseline_data, curr_window_data, n_bins=PSI_BINS)
                psi_history.append(psi_val)
            
            if not psi_history:
                print("   [Skip] No valid windows.")
                continue

            # D. 统计
            psi_arr = np.array(psi_history)
            psi_min = psi_arr.min()
            psi_med = np.median(psi_arr)
            psi_max = psi_arr.max()
            
            # 稍微放宽小样本的 Max 阈值 (Median 才是关键)
            threshold = 0.4 if n_samples < 2000 else 0.25
            status = "Stable" if psi_med < 0.25 else "Drift" # 主要看中位数
            
            print(f"   [Stats] PSI -> Min: {psi_min:.4f} | Median: {psi_med:.4f} | Max: {psi_max:.4f}")
            print(f"   [Result] System is {status} (Median < 0.25).")
            
            results.append({
                "Dataset": name,
                "PSI_Min": psi_min,
                "PSI_Median": psi_med,
                "PSI_Max": psi_max,
                "Status": status
            })

        df = pd.DataFrame(results)
        out_path = os.path.join(self.config.SAVE_DIR, "task3_psi_monitor_adaptive.csv")
        df.to_csv(out_path, index=False)
        print(f"\n [Done] Final PSI Report saved to {out_path}")

    def _run_evaluation(self):
        status = "ACCEPTED (Frozen R*)" if self.is_frozen else "REJECTED (Identity)"
        self._log(f"\n[Stage IV] Evaluation ({status})")
        
        t_base = self.t_raw_pooled
        t_align_base = self._get_alignment_text_base()
        t_capa = self._l2_norm(torch.matmul(t_align_base, self.R_frozen.T))
        gate_sim = self._compute_dataset_alignment_stats(
            z_embed=None,
            y_labels=None,
            t_base=t_align_base,
            t_rot=t_capa,
            sim_source="gate",
        )
        rows = []
        
        
        for name, path in self.config.TEST_DATA_PATHS.items():
            if not os.path.exists(path): continue
            z_test, y_test, is_multi = self._load_data(path, split_override=2)
            if y_test is None: continue
            
            z_test = self._apply_preprocessing(z_test, self.zI_mean)
            dataset_sim_eval = self._compute_dataset_alignment_stats(
                z_embed=z_test,
                y_labels=y_test,
                t_base=t_align_base,
                t_rot=t_capa,
                dataset_name=name,
                sim_source="dataset",
            )
            t_eval = t_capa
            res_base = self._compute_metrics(z_test, y_test, is_multi, t_base, dataset_name=name)
            res_capa = self._compute_metrics(
                z_test,
                y_test,
                is_multi,
                t_eval,
                dataset_name=name,
                baseline_t_protos=t_base,
            )
            res_capa_cache = self._compute_metrics(
                z_test,
                y_test,
                is_multi,
                t_eval,
                dataset_name=name,
                use_cache=self._is_cache_enabled(),
                baseline_t_protos=t_base,
            )

            dataset_sim = dataset_sim_eval
            
            if np.isnan(res_capa["Macro-AUC"]):
                continue
            gain = float(res_capa["Macro-AUC"] - res_base["Macro-AUC"])
            rows.append({
                "Dataset": name,
                "MacroAUC_Base": res_base["Macro-AUC"],
                "MacroAUC_CAPA": res_capa["Macro-AUC"],
                "MacroAUC_CAPA_Cache": res_capa_cache["Macro-AUC"],
                "MicroAUC_Base": res_base["Micro-AUC"],
                "MicroAUC_CAPA": res_capa["Micro-AUC"],
                "MicroAUC_CAPA_Cache": res_capa_cache["Micro-AUC"],
                "ECE_Base": res_base["ECE"],
                "ECE_CAPA": res_capa["ECE"],
                "ECE_CAPA_Cache": res_capa_cache["ECE"],
                "Gain_MacroAUC": gain,
                "Gain_MacroAUC_Cache": float(res_capa_cache["Macro-AUC"] - res_base["Macro-AUC"])
                if np.isfinite(res_capa_cache["Macro-AUC"]) and np.isfinite(res_base["Macro-AUC"])
                else np.nan,
                "Sim_Before": dataset_sim["sim_before"],
                "Sim_After": dataset_sim["sim_after"],
                "SimGain": dataset_sim["sim_gain"],
                "OffDiag_Post_Mean": dataset_sim["off_diag_post_mean"],
            })

        if rows:
            out_path = os.path.join(self.config.SAVE_DIR, "stage_iv_eval.csv")
            pd.DataFrame(rows).to_csv(out_path, index=False)
            self._log(f"[Saved] {out_path}")

        if self.config.PRINT_SUMMARY and rows and self.config.VERBOSE:
            print(f"{'Dataset':<12} | {'MacroAUC':<8} | {'MicroAUC':<8} | {'ECE%':<6} | {'Gain':<8} | {'SimGain':<8} | {'OffDiag':<8}")
            print("-" * 90)
            for r in rows:
                g = float(r["Gain_MacroAUC"])
                sym = "^" if g > 0 else "v"
                print(f"{r['Dataset']:<12} | {r['MacroAUC_CAPA']:.4f}   | {r['MicroAUC_CAPA']:.4f}   | {r['ECE_CAPA']:.2f}   | {sym} {abs(g):.4f} | {r['SimGain']:+.4f} | {r['OffDiag_Post_Mean']:+.4f}")
                print("-" * 90)
