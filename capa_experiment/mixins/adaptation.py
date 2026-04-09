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

class AdaptationMixin:
    def _run_adaptive_scs(self, z, top_preds, max_probs, R):
        B = z.shape[0]

        t_vars = [self._l2_norm(torch.matmul(t, R.T)) for t in self.t_paraphrases[:3]]

        max_budget = torch.ones(B, dtype=torch.long, device=self.device) * 3
        max_budget[max_probs >= self.config.CONF_MED] = 2
        max_budget[max_probs >= self.config.CONF_HIGH] = 1
        
        votes = torch.zeros(B, device=self.device)
        actual_checks = max_budget.float() 

        preds_1 = torch.argmax(self.s_opt * torch.matmul(z, t_vars[0].T), dim=1)
        votes += (preds_1 == top_preds).float()

        mask_2 = max_budget >= 2
        if mask_2.any():
            preds_2 = torch.argmax(self.s_opt * torch.matmul(z[mask_2], t_vars[1].T), dim=1)
            votes[mask_2] += (preds_2 == top_preds[mask_2]).float()

        mask_3 = max_budget >= 3
        if mask_3.any():
            preds_3 = torch.argmax(self.s_opt * torch.matmul(z[mask_3], t_vars[2].T), dim=1)
            votes[mask_3] += (preds_3 == top_preds[mask_3]).float()
            
        scs_scores = votes / actual_checks
        return scs_scores, actual_checks

    def _go_ml_tau(self, T_sub: torch.Tensor) -> float:
        # Condition-number-constrained tau selection, cached by |Y|.
        k = int(T_sub.shape[0])
        if k <= 1:
            return float(getattr(self.config, "GO_ML_TAU_BASE", 1e-2))
        if k in self.go_ml_tau_cache:
            return float(self.go_ml_tau_cache[k])

        tau_base = max(1e-8, float(getattr(self.config, "GO_ML_TAU_BASE", 1e-2)))
        cond_target = max(1.01, float(getattr(self.config, "GO_ML_COND_TARGET", 1e3)))
        G = torch.matmul(T_sub, T_sub.T)
        try:
            eigvals = torch.linalg.eigvalsh(G).real
            lam_max = float(torch.max(eigvals).item())
            lam_min = float(torch.min(eigvals).item())
        except RuntimeError:
            tau = tau_base
            self.go_ml_tau_cache[k] = tau
            return tau

        # Need (lam_max + tau) / (lam_min + tau) <= cond_target.
        num = lam_max - cond_target * lam_min
        den = cond_target - 1.0
        tau_needed = max(0.0, num / den) if den > 0 else 0.0
        tau = max(tau_base, tau_needed + 1e-8)
        self.go_ml_tau_cache[k] = float(tau)
        return float(tau)

    def _compute_multilabel_residual(self, z_img, active_indices, target_c_idx, t_aligned):
        other_indices = [idx for idx in active_indices if idx != target_c_idx]
        if not other_indices:
            return z_img, 1.0

        T_sub = t_aligned[other_indices]
        K = int(T_sub.shape[0])
        G = torch.matmul(T_sub, T_sub.T)
        use_go_ml = bool(getattr(self.config, "ENABLE_GO_MULTILABEL_PROJECTION", False))
        tau = self._go_ml_tau(T_sub) if use_go_ml else float(self.config.MULTI_LABEL_RIDGE)
        Reg = tau * torch.eye(K, device=self.device, dtype=T_sub.dtype)
        rhs = torch.matmul(T_sub, z_img)
        try:
            alpha = torch.linalg.solve(G + Reg, rhs)
        except RuntimeError:
            return z_img, 1.0

        resid_raw = z_img - torch.matmul(alpha, T_sub)
        resid_norm = float(torch.norm(resid_raw).item())
        if not np.isfinite(resid_norm) or resid_norm <= 1e-8:
            return z_img, 1.0
        return self._l2_norm(resid_raw), resid_norm

    def _update_centroids_ema_multilabel(self, z_batch, probs, scs_scores, t_aligned, hpq_threshold):
        B = z_batch.shape[0]
        active_mask = probs > self.config.PROB_THRESHOLD
        
        update_count = 0 
        rejected_labels = 0  # 统计 LQ (Low Quality) 标签数量
        hpq_labels = 0  # 统计 HPQ (High-Quality) 标签数量
        total_candidates = 0 # 统计总共尝试判断的标签数量
        
        for i in range(B):
            active_indices = torch.where(active_mask[i])[0].tolist()
            if not active_indices: continue
            
            img_vec = z_batch[i]
            scs_val = scs_scores[i].item()
            
            for c_idx in active_indices:
                total_candidates += 1
                prob_c = probs[i, c_idx]
                
                # === LQ 判断逻辑 ===
                # 如果置信度不够高 或者 SCS一致性太差,就算作 Low Quality
                if prob_c < hpq_threshold or scs_val < self.config.SCS_THRESH: 
                    rejected_labels += 1 # 计数 +1
                    continue
                
                # 下面是正常的更新逻辑(HPQ样本)
                hpq_labels += 1  # 统计 HPQ 样本
                z_resid, resid_norm = self._compute_multilabel_residual(img_vec, active_indices, c_idx, t_aligned)
                self.class_counts_hpq[c_idx] += 1
                eta = 1.0 / (self.config.KAPPA_EMA + self.class_counts_hpq[c_idx])
                weight = scs_val * (prob_c ** self.config.GAMMA_S)
                if bool(getattr(self.config, "ENABLE_GO_MULTILABEL_PROJECTION", False)) and bool(
                    getattr(self.config, "GO_ML_USE_RESIDUAL_NORM_WEIGHT", True)
                ):
                    weight *= max(1e-4, float(resid_norm))
                current_mu = self.image_centroids[c_idx]
                # vMF shrunk EMA target: blend weighted image signal with text anchor t_c.
                kappa0 = float(self.config.KAPPA0)
                anchor = t_aligned[c_idx]
                signal_vec = img_vec
                if bool(getattr(self.config, "ENABLE_GO_MULTILABEL_PROJECTION", False)) and not bool(
                    getattr(self.config, "GO_ML_SIGNAL_USE_ORIGINAL", True)
                ):
                    signal_vec = z_resid
                target = (kappa0 * anchor + weight * signal_vec) / (kappa0 + weight + 1e-8)
                self.image_centroids[c_idx] = self._l2_norm((1.0 - eta) * current_mu + eta * target)
                update_count += 1
                
        return update_count, rejected_labels, hpq_labels # 返回 LQ 和 HPQ 计数

    def _solve_procrustes(self):
        N = self.class_counts_hpq
        mask = N >= self.config.N_MIN_HPQ_FOR_ACTIVE
        if mask.sum() == 0:
            return self.current_R

        t_ref = self._get_alignment_text_base()
        n_cap = max(1, int(getattr(self.config, "N_CAP", 500)))
        N_capped = torch.clamp(N, max=float(n_cap))
        w = (torch.pow(N_capped, self.config.GAMMA_WEIGHT) + self.config.ALPHA_WEIGHT)
        sim = (self.image_centroids * t_ref).sum(dim=1)
        w *= (1.0 + self.config.BETA_WEIGHT * torch.clamp(self.config.RHO - sim, min=0))
        if mask.any():
            w_med = torch.median(w[mask])
            w_cap = self.config.PROCRUSTES_WEIGHT_CAP_MULT * (w_med + 1e-6)
            w = torch.minimum(w, torch.full_like(w, w_cap))

        w = w * mask.float()
        w = w / w.sum().clamp_min(1e-8)
        mu = self._l2_norm(self.image_centroids)
        t_raw = self._l2_norm(t_ref)
        M_pos = torch.matmul((w.view(-1, 1) * mu).T, t_raw)
        M_cov = M_pos

        use_hard_neg = bool(getattr(self.config, "ENABLE_HARD_NEG_PROCRUSTES", False))
        beta = max(0.0, float(getattr(self.config, "HARD_NEG_BETA", 0.0)))
        topk = int(getattr(self.config, "HARD_NEG_TOPK", 0))
        temp = max(float(getattr(self.config, "HARD_NEG_TEMP", 0.07)), 1e-4)
        if use_hard_neg and beta > 0.0 and topk > 0:
            active_idx = torch.where(mask)[0]
            n_act = int(active_idx.numel())
            if n_act > 1:
                mu_act = mu.index_select(0, active_idx)
                t_act = t_raw.index_select(0, active_idx)
                w_act = w.index_select(0, active_idx)
                sim_cross = torch.matmul(mu_act, t_act.T)
                sim_cross = sim_cross.masked_fill(
                    torch.eye(n_act, dtype=torch.bool, device=self.device),
                    -1e9,
                )
                k_eff = max(1, min(topk, n_act - 1))
                topv, topi = torch.topk(sim_cross, k=k_eff, dim=1, largest=True, sorted=False)
                a = F.softmax(topv / temp, dim=1)
                t_hard = (a.unsqueeze(-1) * t_act[topi]).sum(dim=1)
                M_neg = torch.matmul((w_act.view(-1, 1) * mu_act).T, t_hard)
                M_cov = M_pos - beta * M_neg

        U, _, Vh = torch.linalg.svd(M_cov)
        diag = torch.ones(M_cov.shape[1], device=self.device)
        if torch.det(torch.matmul(U, Vh)) < 0: diag[-1] = -1
        return torch.matmul(torch.matmul(U, torch.diag(diag)), Vh)

    def _diagnose_and_log(self, R_cand, step, phase="Adapt"):
        mask = self.class_counts_hpq >= self.config.N_MIN_HPQ_FOR_ACTIVE
        n_act = mask.sum().item()
        if n_act == 0: return {'passed': False, 'dS': 0.0}

        T = self._get_alignment_text_base()
        Mu = self.image_centroids
        T_rot = torch.matmul(T, R_cand.T)
        
        # 1. Norms & Orthogonality
        mu_norm = torch.norm(Mu, dim=1).mean().item()
        t_rot_norm = torch.norm(T_rot, dim=1).mean().item()
        I = torch.eye(R_cand.shape[0], device=self.device)
        ortho_err = torch.norm(torch.matmul(R_cand, R_cand.T) - I).item()

        # 2. Geometry / Similarities
        # Sim matrix [N_class x N_class] in rotated and pre-rotated spaces.
        Sim_post = torch.matmul(Mu, T_rot.T)
        Sim_pre = torch.matmul(Mu, T.T)

        # Valid: Diagonal elements (matching classes)
        sim_before_vec = Sim_pre.diag()[mask]
        sim_before = sim_before_vec.mean().item()
        valid_sim = Sim_post.diag()[mask].mean().item()

        # Off-diagonal statistics (compression check) on C_tau subset.
        idx = torch.where(mask)[0]
        if int(idx.numel()) > 1:
            Sim_pre_act = Sim_pre.index_select(0, idx).index_select(1, idx)
            Sim_post_act = Sim_post.index_select(0, idx).index_select(1, idx)
            off_diag_mask = ~torch.eye(int(idx.numel()), dtype=torch.bool, device=self.device)
            off_diag_pre_mean = Sim_pre_act[off_diag_mask].mean().item()
            off_diag_post_mean = Sim_post_act[off_diag_mask].mean().item()
            off_diag_delta = off_diag_post_mean - off_diag_pre_mean
        else:
            off_diag_pre_mean = 0.0
            off_diag_post_mean = 0.0
            off_diag_delta = 0.0

        # 3. Gate calculation
        delta = valid_sim - sim_before
        if bool(getattr(self.config, "GATE_USE_RHO_QUANTILE", False)) and int(sim_before_vec.numel()) > 0:
            q = min(1.0, max(0.0, float(getattr(self.config, "RHO_QUANTILE", 0.70))))
            rho_eff = float(torch.quantile(sim_before_vec.detach(), q).item())
        else:
            rho_eff = float(self.config.RHO)
        if not np.isfinite(rho_eff):
            rho_eff = float(self.config.RHO)

        if bool(getattr(self.config, "GATE_REQUIRE_OFFDIAG_IMPROVEMENT", True)):
            offdiag_ok = off_diag_delta <= float(getattr(self.config, "GATE_MAX_OFFDIAG_DELTA", 0.0))
        else:
            offdiag_ok = True

        passed = (delta >= self.config.EPSILON) and (valid_sim <= rho_eff) and offdiag_ok

        if self.config.VERBOSE:
            tqdm.write(
                f" Diagnostic Step {step} ({phase}) | Active {n_act}/{len(mask)} "
                f"| Δs={delta:+.4f} | dOffDiag={off_diag_delta:+.4f} | rho*={rho_eff:.4f} "
                f"| OrthoErr={ortho_err:.2e} | Passed={passed}"
            )
        
        return {
            "passed": passed,
            "dS": delta,
            "s_after": valid_sim,
            "n_act": n_act,
            "dOffDiag": off_diag_delta,
            "rho_eff": rho_eff,
        }

    def _save_and_report_per_class(self):
        self._log("\n[Report] Saving per-class stats...")
        # 1. Safety Check (Compression)
        T_base = self._get_alignment_text_base()
        T_rot = torch.matmul(T_base, self.R_frozen.T)
        
        Sim_Pre = torch.matmul(self.image_centroids, T_base.T)
        Sim_Post = torch.matmul(self.image_centroids, T_rot.T)
        
        off_mask = ~torch.eye(Sim_Post.shape[0], dtype=torch.bool, device=self.device)
        off_mean_pre = Sim_Pre[off_mask].mean().item()
        off_mean_post = Sim_Post[off_mask].mean().item()
        
        self._log(f" Safety Check: Off-Diagonal Change = {off_mean_post - off_mean_pre:+.6f}")
        
        if off_mean_post > off_mean_pre + 0.01:
            self._log("[WARN] Compression risk: off-diagonal similarity increased.")

        run_tag = getattr(self, "config_name", f"seed{self.config.RANDOM_SEED}")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        csv_path = os.path.join(self.config.SAVE_DIR, f"per_class_stats_{run_tag}_{timestamp}.csv")
        self._log(f"[Saved] {csv_path}")

        # 2. Per-Class Snapshot Table
        N = self.class_counts_hpq
        mask = N >= self.config.N_MIN_HPQ_FOR_ACTIVE
        n_cap = max(1, int(getattr(self.config, "N_CAP", 500)))
        N_capped = torch.clamp(N, max=float(n_cap))
        w_raw = (torch.pow(N_capped, self.config.GAMMA_WEIGHT) + self.config.ALPHA_WEIGHT)
        
        sim_raw = (self.image_centroids * T_base).sum(dim=1)
        sim_rot = (self.image_centroids * T_rot).sum(dim=1) 
        
        w_final = w_raw * (1.0 + self.config.BETA_WEIGHT * torch.clamp(self.config.RHO - sim_raw, min=0))
        w_median_val = w_final[mask].median().item() if mask.any() else 0.0
        leverage = torch.zeros_like(w_final)
        if mask.any():
            leverage[mask] = w_final[mask] / (w_median_val + 1e-6)
        
        rows = []
        for i, name in enumerate(self.config.ORDERED_CLASS_NAMES):
            if not mask[i]: continue
            rows.append({
                "name": name,
                "n": int(N[i].item()),
                "sim_pre": sim_raw[i].item(),
                "sim_post": sim_rot[i].item(),
                "gain": sim_rot[i].item() - sim_raw[i].item(),
                "weight": w_final[i].item(),
                "leverage": leverage[i].item()
            })
        
        rows.sort(key=lambda x: x["leverage"], reverse=True)
        if rows:
            non_support = [r for r in rows if r["name"] != "Support Devices"]
            preferred = non_support[0] if non_support else rows[0]
            self.max_leverage_info = f"{preferred['name']} ({preferred['leverage']:.2f})"
            self._log(f" [Info] Max leverage class: {self.max_leverage_info}")
        
        if self.config.VERBOSE:
            self._log("\n Per-Class Snapshot (Sorted by Leverage - Impact on Rotation):")
            header = f"{'Class Name':>40}  {'Support (N)':>11}  {'Sim (Before)':>12}  {'Sim (After)':>12}   {'Gain':>6}  {'Weight (w_c)':>12}  {'Leverage':>10}"
            self._log(header)
            for r in rows:
                self._log(f"{r['name']:>40}  {r['n']:11d}  {r['sim_pre']:12.4f}  {r['sim_post']:12.4f} {r['gain']:+.4f}  {r['weight']:12.2f}  {r['leverage']:10.2f}")
        
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        # 3. Save State (after leverage is computed)
        state_path = os.path.join(self.config.SAVE_DIR, "capa_state.pkl")
        guardian_blob = {
            "enabled": bool(self._is_go_guardian_enabled()),
            "status": str(self.guardian_status),
            "psi_thr": float(getattr(self.config, "GO_PSI_THR", 2.0)),
            "tau_resume": float(getattr(self.config, "GO_TAU_RESUME", 1.0)),
            "resume_windows": int(getattr(self.config, "GO_RESUME_WINDOWS", 3)),
            "warmup_steps": int(getattr(self.config, "GO_WARMUP_STEPS", 50)),
            "baseline_collect_steps": int(getattr(self.config, "GO_BASELINE_COLLECT_STEPS", 50)),
            "dry_run": bool(getattr(self.config, "GO_DRY_RUN", False)),
            "last_psi": float(self.guardian_last_psi) if np.isfinite(self.guardian_last_psi) else np.nan,
            "num_alarms": int(self.guardian_num_alarms),
            "last_alarm_step": int(self.guardian_last_alarm_step),
            "psi_history": list(self.guardian_psi_history[-512:]),
            "psi_baseline_hist": self.guardian_psi_baseline_hist,
            "psi_bin_edges": self.guardian_psi_bin_edges,
            "baseline_samples": int(len(self.guardian_baseline_values)),
        }
        state_dict = {
            "centroids": self.image_centroids.cpu(),
            "R": self.R_frozen.cpu(),
            "R_last_good": self.R_last_good.cpu() if isinstance(self.R_last_good, torch.Tensor) else None,
            "counts": self.class_counts_hpq.cpu(),
            "alignment_stats": self.final_alignment_stats,
            "max_leverage": self.max_leverage_info,
            "t_align_base": T_base.detach().cpu(),
            "t_mixed": T_base.detach().cpu(),
            "t_raw_pooled": self.t_raw_pooled.detach().cpu(),
            "ot_gamma": self.ot_gamma.detach().cpu() if isinstance(self.ot_gamma, torch.Tensor) else None,
            "guardian": guardian_blob,
        }
        with open(state_path, "wb") as f:
            pickle.dump(state_dict, f)
        self._log(f"[Saved] {state_path}")

    def _get_curriculum_threshold(self, step_idx):
        if step_idx >= self.config.WARMUP_BATCHES:
            return self.config.HPQ_CONF_LO
        alpha = step_idx / float(self.config.WARMUP_BATCHES) 
        thresh = self.config.CURRICULUM_THRESH_START - alpha * (self.config.CURRICULUM_THRESH_START - self.config.HPQ_CONF_LO)
        return thresh

    def run_pipeline(self):
        self._log("[Stage I] Calibration")
        z_cal, _, _ = self._load_data(
            self.config.CALIB_DATA_PATH,
            is_calibration=True,
            split_override=1,
        )
        self.zI_mean = z_cal.mean(dim=0, keepdim=True)
        d_feat = int(z_cal.shape[1])
        n_cal = int(len(z_cal))
        min_cov_n = max(2, d_feat + 1)
        if self.config.USE_ZCA_WHITEN and n_cal >= min_cov_n:
            cov = torch.matmul((z_cal - self.zI_mean).T, (z_cal - self.zI_mean)) / max(1, (n_cal - 1))
            try:
                U, S, Vh = torch.linalg.svd(cov)
                self.W_zca = torch.matmul(torch.matmul(U, torch.diag(1.0 / torch.sqrt(S + 1e-5))), Vh)
            except RuntimeError as e:
                self.W_zca = torch.eye(d_feat, device=self.device)
                self._log(f"[Stage I] Whitening SVD failed ({e}); fallback to identity.", always=True)
        else:
            self.W_zca = torch.eye(d_feat, device=self.device)
            if self.config.USE_ZCA_WHITEN:
                self._log(
                    f"[Stage I] Whitening skipped: n={n_cal} < d+1={min_cov_n}; fallback to identity.",
                    always=True,
                )
        self._build_prototypes()

        d = self.t_raw_pooled.shape[1]
        self.current_R = torch.eye(d, device=self.device)
        self.R_frozen = torch.eye(d, device=self.device)
        self.image_centroids = self.t_raw_pooled.clone()
        self.class_counts_hpq = torch.zeros(self.t_raw_pooled.shape[0], device=self.device)
        self.t_align_base = self._compute_ot_mixed_text()
        self._set_prior_bias_from_counts(self.class_counts_hpq)
        self.R_last_good = self.current_R.clone()

        # GO Guardian baseline is collected online after GO warmup steps.
        self.guardian_status = "off"
        self.guardian_last_alarm_step = -1
        self.guardian_num_alarms = 0
        self.guardian_resume_streak = 0
        self.guardian_last_psi = np.nan
        self.guardian_psi_history = []
        self.guardian_psi_baseline_hist = None
        self.guardian_psi_bin_edges = None
        self.guardian_baseline_values = []
        self.guardian_window_values = []
        saved_state = False
        
        self._log(f"[Stage III] Curriculum Adaptation (warmup={self.config.WARMUP_BATCHES} batches)")
        z_train, _, _ = self._load_data(self.config.TRAIN_DATA_PATH, split_override=0)
        z_train = self._apply_preprocessing(z_train, self.zI_mean)
        
        B = self.config.TRAIN_BATCH_SIZE
        
        pbar = tqdm(range(0, len(z_train), B), desc=" Init Warmup", disable=(not self.config.VERBOSE),
                   bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
        
        for i in pbar:
            if self.is_frozen: break
            z_b = z_train[i:i+B]
            step = i // B
            
            is_warmup = step < self.config.WARMUP_BATCHES
            curr_thresh = self._get_curriculum_threshold(step)
            
            if is_warmup:
                R_inference = torch.eye(d, device=self.device)
                phase_name = "WarmUp"
            else:
                R_inference = self.current_R
                phase_name = "Adapt"
                
            t_base = self._get_alignment_text_base()
            t_aligned = self._l2_norm(torch.matmul(t_base, R_inference.T))
            logits = self.s_opt * torch.matmul(z_b, t_aligned.T) + self.b_c
            probs = torch.sigmoid(logits / self.T_opt)

            top_confs, top_preds = torch.max(probs, dim=1)
            g_info = self._guardian_update_from_window(step, probs)
            if g_info is not None and bool(g_info.get("changed", False)):
                if bool(g_info.get("frozen", False)):
                    self._log(
                        f"[GO] Alarm -> freeze at step={step}, PSI={float(g_info['psi']):.4f}, rollback to R_last_good.",
                        always=True,
                    )
                else:
                    self._log(
                        f"[GO] Resume at step={step}, PSI={float(g_info['psi']):.4f}.",
                        always=True,
                    )
            elif g_info is not None and bool(g_info.get("dry_run", False)) and np.isfinite(float(g_info.get("psi", np.nan))):
                if float(g_info["psi"]) > float(getattr(self.config, "GO_PSI_THR", 2.0)):
                    self._log(
                        f"[GO][DryRun] Alarm candidate at step={step}, PSI={float(g_info['psi']):.4f} (no freeze).",
                        always=True,
                    )

            scs, budget = self._run_adaptive_scs(z_b, top_preds, top_confs, R_inference)
            if self._guardian_is_frozen():
                n_updates, n_rejected, n_hpq = 0, 0, 0
                if isinstance(self.R_last_good, torch.Tensor):
                    self.current_R = self.R_last_good.clone()
                    self.R_frozen = self.R_last_good.clone()
            else:
                n_updates, n_rejected, n_hpq = self._update_centroids_ema_multilabel(
                    z_b, probs, scs, t_aligned, hpq_threshold=curr_thresh
                )
            
            # === 实时计算 Mask 和 计数 ===
            mask = self.class_counts_hpq >= self.config.N_MIN_HPQ_FOR_ACTIVE
            n_act = mask.sum().item()
            n_total = len(self.config.ORDERED_CLASS_NAMES)
            
            pbar_metrics = {
                "Ph": phase_name, 
                "Thr": f"{curr_thresh:.2f}", 
                "N_act": f"{n_act}/{n_total}",
                "Upd": f"{n_updates}",
                "HPQ": f"{n_hpq}",
                "LQ": f"{n_rejected}",
                "G": str(self.guardian_status),
                "PSI": f"{float(self.guardian_last_psi):.4f}" if np.isfinite(self.guardian_last_psi) else "NA",
            }
            
            check_gate = False
            if self._guardian_is_frozen():
                if self.config.VERBOSE:
                    pbar.set_description(" GuardianFrozen")
            else:
                if is_warmup:
                    if self.config.VERBOSE:
                        pbar.set_description(f" Warmup {step}/{self.config.WARMUP_BATCHES}")
                elif step == self.config.WARMUP_BATCHES:
                    check_gate = True
                    if self.config.PRINT_SUMMARY:
                        self._log("\n[Warmup Complete] Checking constraints...")
                else:
                    if self.config.VERBOSE:
                        pbar.set_description(" Adapting")
                    if step % 5 == 0:
                        check_gate = True

            if check_gate:
                # Start adaptation only when enough classes are active.
                MIN_CLASSES_FOR_ADAPTATION = self.config.MIN_CLASSES_FOR_ADAPTATION
                
                if n_act < MIN_CLASSES_FOR_ADAPTATION:
                    if self.config.VERBOSE and (step % 10 == 0 or step == self.config.WARMUP_BATCHES):
                        tqdm.write(f" [Gate Held] Too few active classes ({n_act}/{n_total}). Need at least {MIN_CLASSES_FOR_ADAPTATION}.")
                else:
                    self.t_align_base = self._compute_ot_mixed_text()
                    # 有足够类别时，计算 R（_solve_procrustes 内部会自动忽略样本不足的类别）
                    R_cand = self._solve_procrustes()
                    stats = self._diagnose_and_log(R_cand, step, phase=phase_name)
                    pbar_metrics.update(
                        {
                            "dS": f"{stats['dS']:.4f}",
                            "dOff": f"{float(stats.get('dOffDiag', 0.0)):+.4f}",
                            "rho*": f"{float(stats.get('rho_eff', self.config.RHO)):.3f}",
                        }
                    )
                    
                    if stats["passed"]:
                        self.current_R = R_cand
                        self.R_frozen = R_cand
                        if str(self.guardian_status) == "normal":
                            self.R_last_good = R_cand.detach().clone()
                        t_base_eval = self._get_alignment_text_base()
                        T_rot = torch.matmul(t_base_eval, R_cand.T)
                        Sim = torch.matmul(self.image_centroids, T_rot.T)
                        Sim_pre = torch.matmul(self.image_centroids, t_base_eval.T)
                        off_diag_mask = ~torch.eye(Sim.shape[0], dtype=torch.bool, device=self.device)
                        self.final_alignment_stats = {
                            "dS_gain": stats["dS"],
                            "n_act": stats["n_act"],
                            "off_diag_mean": Sim[off_diag_mask].mean().item(),
                            "off_diag_max": Sim[off_diag_mask].max().item(),
                            "off_diag_pre_mean": Sim_pre[off_diag_mask].mean().item(),
                            "hpq_count": float(self.class_counts_hpq.sum().item()),
                            "gate_pass": True,
                        }
                        
                        # 分层冻结策略：
                        # 1. 如果覆盖率高（>80%）且指标合格 → 彻底冻结退出
                        # 2. 否则只更新参数，继续收集更多类别的数据
                        coverage_ratio = n_act / n_total
                        if coverage_ratio >= 0.8:
                            self.is_frozen = True
                            pbar.set_postfix(pbar_metrics)
                            self._log(f"[Gate Passed] Freeze R at step={step} (coverage={coverage_ratio:.1%}, Δs={stats['dS']:.4f})")
                            self._save_and_report_per_class()
                            saved_state = True
                            pbar.close()
                            break
                        else:
                            # 参数更新但不冻结，继续适应
                            self._log(f" [Update] R updated with {n_act}/{n_total} classes ({coverage_ratio:.1%}).")
                    elif step == self.config.WARMUP_BATCHES:
                        self._log(f" [Gate Failed] Warmup insufficient (Δs={stats['dS']:.4f}).")

            if self.config.VERBOSE:
                pbar.set_postfix(pbar_metrics)
            
        if not self.is_frozen:
            self._log("[WARN] Loop finished without freezing; using latest R for evaluation.")
            if self.current_R is None: self.current_R = torch.eye(d, device=self.device)
            self.R_frozen = self.current_R 
            if isinstance(self.current_R, torch.Tensor) and str(self.guardian_status) == "normal":
                self.R_last_good = self.current_R.detach().clone()
        self._set_prior_bias_from_counts(self.class_counts_hpq)
        if not saved_state:
            self._save_and_report_per_class()

        self._run_evaluation()
        self._run_scale_sweep([8, 12, 16, 24, 32, 40])
