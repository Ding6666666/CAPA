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

class CoreMixin:
    def __init__(self, config: CAPA5Config):
        self.config = config
        self.device = torch.device(self.config.DEVICE)
        self.config.VERBOSE = bool(self.config.VERBOSE) or bool(getattr(self.config, "DEBUG", False))
        self._log(f"[CAPA] Init on {self.device} (verbose={self.config.VERBOSE})")
        self._init_seed()
        self._init_clip_model()
        self._init_state()
        if not os.path.exists(self.config.SAVE_DIR):
            os.makedirs(self.config.SAVE_DIR)

    def _log(self, msg: str, *, always: bool = False):
        if always or self.config.VERBOSE:
            print(msg)

    def _init_seed(self):
        random.seed(self.config.RANDOM_SEED)
        np.random.seed(self.config.RANDOM_SEED)
        torch.manual_seed(self.config.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _init_clip_model(self):
        model_dir = self.config.LOCAL_MODEL_PATH
        if not os.path.exists(model_dir): os.makedirs(model_dir)
        os.environ["HF_HOME"] = model_dir
        try:
            self.clip_model, _, _ = open_clip.create_model_and_transforms(
                self.config.MODEL_NAME, cache_dir=model_dir
            )
            self.tokenizer = open_clip.get_tokenizer(self.config.MODEL_NAME)
            self.clip_model = self.clip_model.to(self.device).eval()
            self._log(" BioMedCLIP Loaded.")
        except Exception as e:
            print(f" Model Load Error: {e}")
            raise e

    def _init_state(self):
        self.zI_mean = None; self.zT_mean = None; self.W_zca = None
        self.T_opt = self.config.INIT_TEMPERATURE
        self.s_opt = self.config.INIT_SCALE_FACTOR
        self.t_raw_pooled = None
        self.t_raw_pooled_raw = None  # store unwhitened text prototypes for raw-space geometry reporting
        self.t_align_base = None  # alignment-side text prototypes (may be OT-mixed)
        self.t_paraphrases = []
        self.current_R = None; self.image_centroids = None
        self.class_counts_hpq = None; self.b_c = None
        self.is_frozen = False; self.R_frozen = None
        self.ot_gamma = None
        self.cache_keys = None
        self.cache_labels = None
        self.cache_is_multi = False
        self.cache_ready = False
        self.final_alignment_stats = {}
        self.max_leverage_info = "N/A"
        self.R_last_good = None

        # GO Guardian runtime state.
        self.guardian_status = "off"  # off | baseline_collect | normal | frozen
        self.guardian_last_alarm_step = -1
        self.guardian_num_alarms = 0
        self.guardian_resume_streak = 0
        self.guardian_last_psi = np.nan
        self.guardian_psi_history: List[float] = []
        self.guardian_psi_baseline_hist: Optional[np.ndarray] = None
        self.guardian_psi_bin_edges: Optional[np.ndarray] = None
        self.guardian_baseline_values: List[float] = []
        self.guardian_window_values: List[float] = []

        # GO multi-label tau cache by active-label cardinality.
        self.go_ml_tau_cache: Dict[int, float] = {}

    def _l2_norm(self, x, dim=-1):
        return F.normalize(x, p=2, dim=dim, eps=1e-8)

    def _encode_text(self, texts: List[str]):
        with torch.no_grad():
            text_inputs = self.tokenizer(texts).to(self.device)
            return self._l2_norm(self.clip_model.encode_text(text_inputs))

    def _find_embedding_col(self, cols):
        for c in cols:
            if ('img' in c.lower() or 'visual' in c.lower()) and ('emb' in c.lower() or 'feat' in c.lower()): return c
            if 'embedding' in c.lower(): return c
        return None

    def _canonical_dataset_name(self, dataset_name: Optional[str]) -> str:
        if dataset_name is None:
            return ""
        raw = str(dataset_name).strip()
        low = raw.lower()
        if "chexpert" in low:
            return "CheXpert"
        if "mimic" in low:
            return "MIMIC"
        if "covid" in low:
            return "COVID"
        if "rsna" in low:
            return "RSNA"
        return raw

    def _resolve_legacy_data_path(self, path: str) -> str:
        p = str(path)
        if os.path.isfile(p):
            return p
        if not os.path.exists(p):
            return p
        if os.path.isdir(p):
            low = p.lower().replace("/", "\\")
            data_root = str(getattr(self.config, "DATA_ROOT", "")).rstrip("\\/")
            if "chexpert-v1.0-small" in low:
                mapped = os.path.join(data_root, "chexpert_small_full.pkl")
                return mapped
            if "covid19-radiography-database" in low:
                mapped = os.path.join(data_root, "COVID_3616x2.pkl")
                return mapped
            if low.endswith("\\rsna") or ("\\raw_data\\rsna" in low):
                mapped = os.path.join(data_root, "RSNA_4243x2.pkl")
                return mapped
        return p

    def _load_data(self, path, is_calibration=False, split_override: Optional[int] = None):
        src = self._resolve_legacy_data_path(path)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Missing: {path} (resolved={src})")
        data = None
        try:
            with open(src, "rb") as f:
                data = pickle.load(f)
        except Exception:
            # Support torch.save payloads (e.g., chexpert_small_full.pkl built by build_train_pkl.py).
            data = torch.load(src, map_location="cpu", weights_only=False)

        # Support dict payload generated by build_train_pkl.py (embeddings/labels/split/class_names)
        if isinstance(data, dict) and ("embeddings" in data):
            emb = np.asarray(data.get("embeddings", []), dtype=np.float32)
            if emb.ndim != 2:
                raise RuntimeError(f"Invalid embeddings shape in {src}: {emb.shape}")
            y = np.asarray(data.get("labels", [])) if ("labels" in data) else None
            split = np.asarray(data.get("split", []), dtype=np.int8).reshape(-1) if ("split" in data) else None

            mask = np.ones(emb.shape[0], dtype=bool)
            if split is not None and split.shape[0] == emb.shape[0]:
                use_split = None
                if split_override is not None:
                    use_split = int(split_override)
                elif src == self.config.TRAIN_DATA_PATH:
                    use_split = 0
                elif is_calibration or src in (self.config.CALIB_DATA_PATH, self.config.TAU_CALIB_DATA_PATH):
                    use_split = 1
                elif src in [self._resolve_legacy_data_path(v) for v in list(self.config.TEST_DATA_PATHS.values())]:
                    use_split = 2
                if use_split is not None:
                    mask = split == int(use_split)
                    if int(mask.sum()) == 0:
                        # Fallback: keep all if requested split does not exist.
                        mask = np.ones(emb.shape[0], dtype=bool)

            emb = emb[mask]
            z = torch.tensor(emb, dtype=torch.float32)
            if y is not None and y.size > 0:
                y = y[mask]
                is_multi = bool(getattr(y, "ndim", 1) == 2 and y.shape[1] > 1)
                if y.ndim == 2 and y.shape[1] == 1:
                    y = y[:, 0]
                    is_multi = False
                return z.to(self.device), y.astype(np.int32), is_multi
            return z.to(self.device), None, False

        col = self._find_embedding_col(data.columns)
        if col is None:
            raise RuntimeError(f"Cannot find embedding column in {src}. Columns: {list(data.columns)}")
        vals = data[col].values
        z = torch.stack(list(vals)) if isinstance(vals[0], torch.Tensor) else torch.tensor(np.stack(list(vals)), dtype=torch.float32)
        if is_calibration and "chexpert_mimic" in src.lower():
            z = z[:1000]
        
        y, is_multi = None, False
        lower_cols = {c.lower(): c for c in data.columns}
        cand_cols = ['labels', 'label', 'finding', 'target', 'class', 'studylabel', 'pneumonia', 'covid']
        found_col = next((lower_cols[c] for c in cand_cols if c in lower_cols), None)
        
        if found_col:
            raw = data[found_col].values
            if isinstance(raw[0], (list, tuple, np.ndarray)):
                if len(raw[0]) > 1: 
                    y = np.stack(raw)  # 直接 stack 成 (N, C) 的 2D 数组
                    is_multi = True
                else: 
                    y = np.array([int(x[0]) if len(x)>0 else 0 for x in raw])
            else: 
                y = np.array([int(x) for x in raw])

        if y is not None and len(y) != len(z):
            n_trim = min(int(len(z)), int(len(y)))
            z = z[:n_trim]
            y = y[:n_trim]
        return z.to(self.device), y, is_multi

    def _apply_preprocessing(self, z, mean_vec):
        if mean_vec is None: return self._l2_norm(z)
        z_centered = z - mean_vec
        if self.config.USE_ZCA_WHITEN and self.W_zca is not None:
            z_centered = torch.matmul(z_centered, self.W_zca)
        return self._l2_norm(z_centered)

    def _get_alignment_text_base(self) -> torch.Tensor:
        if (
            self.t_align_base is not None
            and self.t_raw_pooled is not None
            and isinstance(self.t_align_base, torch.Tensor)
            and tuple(self.t_align_base.shape) == tuple(self.t_raw_pooled.shape)
        ):
            return self.t_align_base
        return self.t_raw_pooled

    def _sinkhorn_plan(self, cost: torch.Tensor, eps: float, n_iters: int) -> torch.Tensor:
        if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
            raise ValueError(f"cost must be square [C,C], got {tuple(cost.shape)}")
        c = int(cost.shape[0])
        if c <= 0:
            raise ValueError("cost has zero size.")
        eps = max(float(eps), 1e-4)
        n_iters = max(1, int(n_iters))

        K = torch.exp(-cost / eps).clamp_min(1e-12)
        a = torch.ones((c,), device=cost.device, dtype=cost.dtype)
        b = torch.ones((c,), device=cost.device, dtype=cost.dtype)
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        for _ in range(n_iters):
            u = a / torch.matmul(K, v).clamp_min(1e-12)
            v = b / torch.matmul(K.T, u).clamp_min(1e-12)
        gamma = (u.view(-1, 1) * K) * v.view(1, -1)
        return gamma

    def _compute_ot_mixed_text(self) -> torch.Tensor:
        t_base = self._l2_norm(self.t_raw_pooled)
        self.ot_gamma = torch.eye(int(t_base.shape[0]), device=self.device, dtype=t_base.dtype)
        if not bool(getattr(self.config, "ENABLE_OT_PROTOTYPE_MIXING", False)):
            return t_base
        if self.image_centroids is None:
            return t_base

        c = int(t_base.shape[0])
        if c <= 1:
            return t_base

        mu = self._l2_norm(self.image_centroids[:c])
        use_active_only = bool(getattr(self.config, "OT_ACTIVE_ONLY", True))
        if use_active_only and self.class_counts_hpq is not None and int(self.class_counts_hpq.shape[0]) >= c:
            mask = self.class_counts_hpq[:c] >= self.config.N_MIN_HPQ_FOR_ACTIVE
        else:
            mask = torch.ones((c,), dtype=torch.bool, device=self.device)

        idx = torch.where(mask)[0]
        min_active = max(2, int(getattr(self.config, "OT_MIN_ACTIVE_CLASSES", 2)))
        if int(idx.numel()) < min_active:
            return t_base

        mu_act = mu.index_select(0, idx)
        t_act = t_base.index_select(0, idx)
        sim = torch.matmul(mu_act, t_act.T)
        cost = (1.0 - sim).clamp_min(0.0)

        gamma_act = self._sinkhorn_plan(
            cost,
            eps=float(getattr(self.config, "OT_SINKHORN_EPS", 0.05)),
            n_iters=int(getattr(self.config, "OT_SINKHORN_ITERS", 100)),
        )
        gamma_act = gamma_act / gamma_act.sum(dim=1, keepdim=True).clamp_min(1e-8)
        t_ot = self._l2_norm(torch.matmul(gamma_act, t_act))
        alpha = max(0.0, min(1.0, float(getattr(self.config, "OT_IDENTITY_BLEND", 0.0))))
        t_act_mixed = self._l2_norm((1.0 - alpha) * t_ot + alpha * t_act) if alpha > 0.0 else t_ot

        t_mixed = t_base.clone()
        t_mixed[idx] = t_act_mixed

        gamma_full = torch.eye(c, device=self.device, dtype=t_base.dtype)
        for i in range(int(idx.numel())):
            ii = int(idx[i].item())
            gamma_full[ii, idx] = gamma_act[i]
        self.ot_gamma = gamma_full
        return self._l2_norm(t_mixed)

    def _build_prototypes(self):
        classes = self.config.ORDERED_CLASS_NAMES
        t_reshaped_list = []
        for cls_name in classes:
            syns = self.config.MEDICAL_SYNONYM_MAP.get(cls_name, [cls_name])
            templates = self.config.TEMPLATES_PI
            cls_texts = [templates[i % len(templates)].replace("{finding}", syns[i % len(syns)]) for i in range(self.config.M)]
            t_reshaped_list.append(self._encode_text(cls_texts))
        t_reshaped = torch.stack(t_reshaped_list)  # shape [C, M, D]

        # 1) Raw pooled prototypes (raw space) — store for geometry reporting
        self.t_raw_pooled_raw = self._l2_norm(t_reshaped.mean(dim=1))  # [C, D]

        # 2) Compute text-mean on raw pooled (this will serve as zT_mean)
        self.zT_mean = self.t_raw_pooled_raw.mean(dim=0, keepdim=True)

        # 3) Now generate processed/whitened prototypes for scoring
        t_all_flat = t_reshaped.view(-1, t_reshaped.shape[-1])
        t_all_flat_proc = self._apply_preprocessing(t_all_flat, self.zT_mean)
        t_reshaped_proc = t_all_flat_proc.view(len(classes), self.config.M, -1)
        self.t_raw_pooled = self._l2_norm(t_reshaped_proc.mean(dim=1))
        self.t_align_base = self.t_raw_pooled.clone()
        self.ot_gamma = torch.eye(int(self.t_raw_pooled.shape[0]), device=self.device, dtype=self.t_raw_pooled.dtype)
        self.cache_keys = None
        self.cache_labels = None
        self.cache_is_multi = False
        self.cache_ready = False
        self.t_paraphrases = [self._l2_norm(t_reshaped_proc[:, m, :]) for m in range(self.config.M)]

    def _set_prior_bias_from_counts(self, counts: torch.Tensor):
        tau_prior = float(getattr(self.config, "TAU_PRIOR", 0.0))
        if tau_prior <= 0.0:
            self.b_c = torch.zeros((1, self.t_raw_pooled.shape[0]), device=self.device)
            return
        cnt = counts.detach().to(self.device).float()
        priors = (cnt + 1.0) / (cnt.sum() + float(len(cnt)))
        self.b_c = (tau_prior * torch.log(priors.clamp_min(1e-8))).view(1, -1)
