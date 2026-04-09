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

class ScoringMixin:
    def _resolve_scoring_mode(self, scoring_mode: Optional[str] = None) -> str:
        mode = str(scoring_mode or getattr(self.config, "SCORING_MODE", "mixed")).strip().lower()
        if mode not in ("mixed", "softmax"):
            raise ValueError(f"Unsupported scoring mode: {mode}. Use 'mixed' or 'softmax'.")
        return mode

    def _resolve_sim_source(self, sim_source: Optional[str] = None) -> str:
        src = str(sim_source or getattr(self.config, "SIM_SOURCE", "gate")).strip().lower()
        if src not in ("gate", "dataset"):
            raise ValueError(f"Unsupported sim source: {src}. Use 'gate' or 'dataset'.")
        return src

    def _get_binary_positive_indices(self, dataset_name: Optional[str], n_proto: int) -> List[int]:
        key = dataset_name if dataset_name in self.config.BINARY_POSITIVE_CLASS_MAP else "default"
        class_names = self.config.BINARY_POSITIVE_CLASS_MAP.get(key, [])
        if not class_names:
            class_names = self.config.BINARY_POSITIVE_CLASS_MAP.get("default", ["Pneumonia"])
        idxs: List[int] = []
        for cls in class_names:
            if cls in self.config.ORDERED_CLASS_NAMES:
                idx = self.config.ORDERED_CLASS_NAMES.index(cls)
                if idx < n_proto:
                    idxs.append(idx)
        if not idxs and n_proto > 0:
            fallback = 0
            if "Pneumonia" in self.config.ORDERED_CLASS_NAMES:
                fallback = min(self.config.ORDERED_CLASS_NAMES.index("Pneumonia"), n_proto - 1)
            idxs = [fallback]
        return sorted(set(idxs))

    def _binary_logit_from_multiclass(self, logits: torch.Tensor, pos_indices: List[int]) -> torch.Tensor:
        n_cls = logits.shape[1]
        pos_mask = torch.zeros(n_cls, dtype=torch.bool, device=logits.device)
        pos_mask[pos_indices] = True
        neg_mask = ~pos_mask
        if not neg_mask.any():
            return logits[:, pos_indices].mean(dim=1)
        pos_term = torch.logsumexp(logits[:, pos_mask], dim=1)
        neg_term = torch.logsumexp(logits[:, neg_mask], dim=1)
        return pos_term - neg_term

    def _is_cache_enabled(self) -> bool:
        return bool(getattr(self.config, "ENABLE_CAPA_CACHE", False))

    def _labels_to_cache_matrix(
        self,
        y_labels,
        *,
        is_multi: bool,
        dataset_name: Optional[str],
        n_cls: int,
    ) -> Optional[torch.Tensor]:
        if y_labels is None:
            return None
        y_arr = np.asarray(y_labels)
        if y_arr.ndim == 0:
            return None
        n = int(y_arr.shape[0])
        if n <= 0:
            return None
        out = np.zeros((n, n_cls), dtype=np.float32)
        if is_multi:
            if y_arr.ndim == 1:
                y_arr = y_arr.reshape(-1, 1)
            n_use = max(1, min(n_cls, int(y_arr.shape[1])))
            out[:, :n_use] = (y_arr[:, :n_use] > 0).astype(np.float32)
        else:
            y_bin = (y_arr.reshape(-1) > 0).astype(np.float32)
            pos_indices = self._get_binary_positive_indices(dataset_name, n_cls)
            for idx in pos_indices:
                out[:, idx] = y_bin
        return torch.tensor(out, dtype=torch.float32)

    def _build_global_cache(self) -> bool:
        if self.cache_ready and self.cache_keys is not None and self.cache_labels is not None:
            return True
        if not self._is_cache_enabled():
            return False
        try:
            z_raw, y_train, is_multi = self._load_data(
                self.config.TRAIN_DATA_PATH,
                split_override=0,
            )
        except Exception as e:
            self._log(f"[Cache] build skipped (load error): {e}")
            return False
        if y_train is None or len(z_raw) == 0:
            return False
        z_proc = self._apply_preprocessing(z_raw, self.zI_mean).detach().cpu()
        y_cache = self._labels_to_cache_matrix(
            y_train,
            is_multi=is_multi,
            dataset_name="default",
            n_cls=int(self.t_raw_pooled.shape[0]),
        )
        if y_cache is None:
            return False
        y_cache = y_cache.detach().cpu()
        if z_proc.shape[0] != y_cache.shape[0]:
            n = min(int(z_proc.shape[0]), int(y_cache.shape[0]))
            z_proc = z_proc[:n]
            y_cache = y_cache[:n]
        if int(z_proc.shape[0]) <= 0:
            return False
        self.cache_keys = z_proc
        self.cache_labels = y_cache
        self.cache_is_multi = bool(is_multi)
        self.cache_ready = True
        self._log(
            f"[Cache] built keys={int(z_proc.shape[0])}, dim={int(z_proc.shape[1])}, n_cls={int(y_cache.shape[1])}",
            always=True,
        )
        return True

    def _compute_cache_logits(self, z_embed: torch.Tensor, dataset_name: Optional[str]) -> Optional[torch.Tensor]:
        if not self._is_cache_enabled():
            return None
        if not self._build_global_cache():
            return None
        keys = self.cache_keys
        labels = self.cache_labels
        if keys is None or labels is None:
            return None
        if keys.numel() == 0 or labels.numel() == 0:
            return None

        cache_topk = max(1, min(int(getattr(self.config, "CACHE_TOPK", 32)), int(keys.shape[0])))
        cache_temp = max(float(getattr(self.config, "CACHE_TEMP", 0.07)), 1e-4)
        chunk_size = max(16, int(getattr(self.config, "CACHE_CHUNK", 512)))
        self_match_cos = float(getattr(self.config, "CACHE_SELF_MATCH_COS", 0.999999))

        z_cpu = z_embed.detach().float().cpu()
        out_chunks: List[torch.Tensor] = []
        for s in range(0, int(z_cpu.shape[0]), chunk_size):
            e = min(s + chunk_size, int(z_cpu.shape[0]))
            z_part = z_cpu[s:e]
            sims = torch.matmul(z_part, keys.T)
            self_mask = sims >= self_match_cos
            if bool(self_mask.any().item()):
                sims = sims.masked_fill(self_mask, -1e9)
            topv, topi = torch.topk(sims, k=cache_topk, dim=1, largest=True, sorted=False)
            w = F.softmax(topv / cache_temp, dim=1)
            neigh = labels[topi]
            p = (w.unsqueeze(-1) * neigh).sum(dim=1).clamp(1e-4, 1.0 - 1e-4)
            l = torch.log(p / (1.0 - p))
            out_chunks.append(l)
        cache_logits = torch.cat(out_chunks, dim=0)
        return cache_logits.to(z_embed.device)

    def _blend_with_cache_logits(
        self,
        logits: torch.Tensor,
        z_embed: torch.Tensor,
        dataset_name: Optional[str],
        *,
        use_cache: bool,
    ) -> torch.Tensor:
        if not use_cache:
            return logits
        if not self._is_cache_enabled():
            return logits
        alpha = max(0.0, min(1.0, float(getattr(self.config, "CACHE_ALPHA", 0.0))))
        if alpha <= 0.0:
            return logits
        cache_logits = self._compute_cache_logits(z_embed, dataset_name)
        if cache_logits is None:
            return logits
        n_use = min(int(logits.shape[1]), int(cache_logits.shape[1]))
        if n_use <= 0:
            return logits
        out = logits.clone()
        out[:, :n_use] = (1.0 - alpha) * out[:, :n_use] + alpha * cache_logits[:, :n_use]
        return out

    def _is_soft_fusion_enabled(self) -> bool:
        if not bool(getattr(self.config, "ENABLE_CAPA_BASELINE_SOFT_FUSION", False)):
            return False
        lam = float(getattr(self.config, "CAPA_BASELINE_FUSION_LAMBDA", 1.0))
        return 0.0 <= lam < 1.0

    def _fuse_with_baseline_logits(
        self,
        logits_capa: torch.Tensor,
        z_embed: torch.Tensor,
        *,
        scale: float,
        baseline_t_protos: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if baseline_t_protos is None:
            return logits_capa
        if not self._is_soft_fusion_enabled():
            return logits_capa
        lam = min(1.0, max(0.0, float(getattr(self.config, "CAPA_BASELINE_FUSION_LAMBDA", 1.0))))
        logits_base = scale * torch.matmul(z_embed, baseline_t_protos.T) + self.b_c
        n_use = min(int(logits_capa.shape[1]), int(logits_base.shape[1]))
        if n_use <= 0:
            return logits_capa
        out = logits_capa.clone()
        out[:, :n_use] = lam * out[:, :n_use] + (1.0 - lam) * logits_base[:, :n_use]
        return out

    def _compose_eval_logits(
        self,
        z_embed: torch.Tensor,
        t_protos: torch.Tensor,
        *,
        scale: float,
        baseline_t_protos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = scale * torch.matmul(z_embed, t_protos.T) + self.b_c
        logits = self._fuse_with_baseline_logits(
            logits,
            z_embed,
            scale=scale,
            baseline_t_protos=baseline_t_protos,
        )
        return logits

    def _predict_probs(
        self,
        logits: torch.Tensor,
        *,
        calib_T: float,
        is_multi: bool,
        dataset_name: Optional[str],
        scoring_mode: Optional[str] = None,
        ranking: bool = False,
    ) -> torch.Tensor:
        mode = self._resolve_scoring_mode(scoring_mode)
        T = 1.0 if ranking else max(float(calib_T), 1e-4)

        if is_multi:
            if mode == "mixed":
                return torch.sigmoid(logits / T)
            return F.softmax(logits / T, dim=1)

        pos_indices = self._get_binary_positive_indices(dataset_name, logits.shape[1])
        if mode == "mixed":
            bin_logit = self._binary_logit_from_multiclass(logits / T, pos_indices)
            return torch.sigmoid(bin_logit)

        probs_full = F.softmax(logits / T, dim=1)
        return probs_full[:, pos_indices].sum(dim=1).clamp(0.0, 1.0)
