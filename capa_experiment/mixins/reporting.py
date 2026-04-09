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

class ReportingMixin:
    def _fmt_num(self, v: float, digits: int = 4, signed: bool = False) -> str:
        if v is None or (not np.isfinite(v)):
            return "NA"
        av = abs(float(v))
        sign_flag = "+" if signed else ""
        if av >= 1e4 or (av > 0 and av < 1e-4):
            return f"{float(v):{sign_flag}.{digits}e}"
        return f"{float(v):{sign_flag}.{digits}f}"

    def _fmt_p(self, p: float) -> str:
        if p is None or (not np.isfinite(p)):
            return "NA"
        if p < 1e-4:
            return "<1e-4"
        if p < 1e-3:
            return f"{p:.1e}"
        return f"{p:.4f}"

    def print_final_gate_summary(self, final_rows: List[Dict[str, object]], sim_source: Optional[str] = None):
        sim_src = self._resolve_sim_source(sim_source)
        name_map = {"MIMIC": "MIMIC-CXR"}
        header = (
            "Dataset    Sim_before  Sim_after  dSim     dOffDiag   dMacroAUC   95% CI           "
            "p-value     dECE(%)"
        )
        sep = "-" * len(header)

        lines = []
        title = "Gate-centroid metrics" if sim_src == "gate" else "Dataset-centroid metrics"
        lines.append(f"=== CAPA Final Summary ({title}) ===")
        lines.append("")
        if sim_src == "gate":
            lines.append(
                "Sim: class-wise mean paired cosine between Procrustes image centroids mu_c"
            )
            lines.append(
                "     and text prototypes t_c in the aligned (center+whiten+norm) space over C_tau."
            )
            lines.append(
                "Note: in gate mode, Sim columns are expected to be identical across datasets."
            )
        else:
            lines.append(
                "Sim: same paired-cosine formula, but mu_c is recomputed per evaluation dataset"
            )
            lines.append(
                "     from dataset labels; R* and text prototypes t_c remain the frozen gate solution."
            )
        lines.append(
            "dSim = Sim_after - Sim_before; negative dOffDiag indicates better separation."
        )
        lines.append("")
        lines.append(header)
        lines.append(sep)

        sig_pos = []
        neutral = []
        for r in final_rows:
            ds_name = str(r.get("Dataset", ""))
            ds = name_map.get(ds_name, ds_name)
            sim_b = float(r.get("Sim_Before", np.nan))
            sim_a = float(r.get("Sim_After", np.nan))
            dsim = float(r.get("Sim_Gain", np.nan))
            doff = float(r.get("Delta_OffDiag", np.nan))
            dauc = float(r.get("Delta_AUC_CAPA_minus_Baseline_Macro", np.nan))
            ci_l = float(r.get("Delta_AUC_CI_Low", np.nan))
            ci_h = float(r.get("Delta_AUC_CI_High", np.nan))
            pval = float(r.get("Delta_AUC_DeLong_p", np.nan))
            dece = float(r.get("Delta_ECE", np.nan))

            ci_txt = f"[{self._fmt_num(ci_l)}, {self._fmt_num(ci_h)}]"
            lines.append(
                f"{ds:<10} {self._fmt_num(sim_b):>10} {self._fmt_num(sim_a):>10} {self._fmt_num(dsim, signed=True):>8} "
                f"{self._fmt_num(doff, signed=True):>10} {self._fmt_num(dauc, signed=True):>11}   "
                f"{ci_txt:<16} {self._fmt_p(pval):>9} {self._fmt_num(dece, signed=True):>10}"
            )

            if np.isfinite(dauc) and np.isfinite(pval) and pval < 0.05 and dauc > 0:
                sig_pos.append(ds)
            else:
                neutral.append(ds)

        if len(final_rows) == 0:
            lines.append("(no datasets)")
        else:
            c0 = final_rows[0]
            if "Calib_Sim_Before" in c0 and "Calib_Sim_After" in c0:
                lines.append("")
                lines.append(
                    "Calibration-centroid (shared gate state): "
                    f"Sim_before={self._fmt_num(float(c0.get('Calib_Sim_Before', np.nan)))}, "
                    f"Sim_after={self._fmt_num(float(c0.get('Calib_Sim_After', np.nan)))}, "
                    f"dSim={self._fmt_num(float(c0.get('Calib_Sim_Gain', np.nan)), signed=True)}, "
                    f"dOffDiag={self._fmt_num(float(c0.get('Calib_Delta_OffDiag', np.nan)), signed=True)}"
                )
        lines.append("")
        if len(sig_pos) > 0:
            lines.append(
                f"Summary: significant dMacroAUC gains on {', '.join(sig_pos)}; "
                f"neutral/mixed on {', '.join(neutral) if neutral else 'none'}."
            )
        else:
            lines.append(
                f"Summary: no statistically significant positive dMacroAUC; "
                f"neutral/mixed on {', '.join(neutral) if neutral else 'all datasets'}."
            )
        lines.append("Note: dMacroAUC is relative to the frozen zero-shot baseline.")
        lines.append("      CAPA optimizes geometry (dSim); discrimination gains are indirect.")

        print("\n".join(lines))

    def print_three_way_auc_summary(self, final_rows: List[Dict[str, object]]):
        header = (
            "Dataset    BaseAUC    CAPA-AUC   CAPA+Cache   dAUC(CAPA)   dAUC(Cache)"
        )
        sep = "-" * len(header)
        lines = ["=== Three-way AUC (Macro) ===", "", header, sep]
        for r in final_rows:
            ds = str(r.get("Dataset", ""))
            b = float(r.get("AUC_Baseline_ZeroShot_Macro", np.nan))
            c = float(r.get("AUC_CAPA_Aligned_Macro", np.nan))
            cc = float(r.get("AUC_CAPA_Cache_Macro", np.nan))
            dc = float(r.get("Delta_AUC_CAPA_minus_Baseline_Macro", np.nan))
            dcc = float(r.get("Delta_AUC_CAPA_Cache_minus_Baseline_Macro", np.nan))
            lines.append(
                f"{ds:<10} {self._fmt_num(b):>9} {self._fmt_num(c):>10} {self._fmt_num(cc):>12} "
                f"{self._fmt_num(dc, signed=True):>12} {self._fmt_num(dcc, signed=True):>12}"
            )
        if len(final_rows) == 0:
            lines.append("(no datasets)")
        print("\n".join(lines))

    def run_shared_vs_per_dataset_capa(
        self,
        shared_rows: List[Dict[str, object]],
        scoring_mode: Optional[str] = None,
    ) -> pd.DataFrame:
        mode = self._resolve_scoring_mode(scoring_mode)
        shared_map: Dict[str, Dict[str, object]] = {}
        for r in shared_rows:
            key = self._canonical_dataset_name(str(r.get("Dataset", "")))
            if key:
                shared_map[key] = r

        rows: List[Dict[str, object]] = []
        self._log("[PerDatasetCAPA] running per-dataset CAPA branch ...", always=True)
        for ds_name, path in self.config.TEST_DATA_PATHS.items():
            key = self._canonical_dataset_name(ds_name)
            if not os.path.exists(path):
                continue

            per_row: Dict[str, object] = {}
            try:
                cfg_ds = copy.deepcopy(self.config)
                cfg_ds.DEBUG = False
                cfg_ds.VERBOSE = False
                cfg_ds.PRINT_SUMMARY = False
                cfg_ds.TEST_DATA_PATHS = {key: path}
                # Experimental branch: use same dataset file with split_override logic.
                cfg_ds.TRAIN_DATA_PATH = path
                cfg_ds.CALIB_DATA_PATH = path
                cfg_ds.TAU_CALIB_DATA_PATH = path
                cfg_ds.SAVE_DIR = os.path.join(self.config.SAVE_DIR, f"per_dataset_capa_{key.lower()}")
                os.makedirs(cfg_ds.SAVE_DIR, exist_ok=True)

                runner_ds = CAPA5NotebookRunner(cfg_ds)
                runner_ds._build_prototypes()
                if runner_ds.class_counts_hpq is None:
                    runner_ds.class_counts_hpq = torch.ones(
                        len(runner_ds.config.ORDERED_CLASS_NAMES),
                        device=runner_ds.device,
                    ) * runner_ds.config.N_MIN_HPQ_FOR_ACTIVE
                if runner_ds.image_centroids is None:
                    runner_ds.image_centroids = runner_ds.t_raw_pooled.clone()
                if runner_ds.current_R is None:
                    runner_ds.current_R = torch.eye(runner_ds.t_raw_pooled.shape[1], device=runner_ds.device)

                runner_ds.run_pipeline()
                per_rows = runner_ds.run_manuscript_validation(scoring_mode=mode, sim_source="dataset")
                if len(per_rows) > 0:
                    per_row = per_rows[0]
                del runner_ds
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                self._log(f"[PerDatasetCAPA:{key}] failed: {e}", always=True)
                per_row = {}

            shared_row = shared_map.get(key, {})
            shared_dauc = float(shared_row.get("Delta_AUC_CAPA_minus_Baseline_Macro", np.nan))
            per_dauc = float(per_row.get("Delta_AUC_CAPA_minus_Baseline_Macro", np.nan))
            rows.append(
                {
                    "Dataset": key,
                    "Shared_dMacroAUC": shared_dauc,
                    "PerDataset_dMacroAUC": per_dauc,
                    "PerMinusShared_dMacroAUC": (per_dauc - shared_dauc)
                    if np.isfinite(per_dauc) and np.isfinite(shared_dauc)
                    else np.nan,
                    "Shared_dECE": float(shared_row.get("Delta_ECE", np.nan)),
                    "PerDataset_dECE": float(per_row.get("Delta_ECE", np.nan)),
                    "Shared_dSim_dataset": float(shared_row.get("Sim_Gain", np.nan)),
                    "PerDataset_dSim_dataset": float(per_row.get("Sim_Gain", np.nan)),
                    "Shared_dSim_calib": float(shared_row.get("Calib_Sim_Gain", np.nan)),
                    "PerDataset_dSim_calib": float(per_row.get("Calib_Sim_Gain", np.nan)),
                }
            )

        df = pd.DataFrame(rows)
        out_path = os.path.join(self.config.SAVE_DIR, f"shared_vs_per_dataset_capa_{mode}.csv")
        df.to_csv(out_path, index=False)

        header = "Dataset    dAUC(shared)  dAUC(per-ds)  per-shared   dSim_calib(shared)  dSim_calib(per-ds)"
        sep = "-" * len(header)
        lines = ["=== Shared vs Per-dataset CAPA ===", "", header, sep]
        for r in rows:
            lines.append(
                f"{r['Dataset']:<10} "
                f"{self._fmt_num(r['Shared_dMacroAUC'], signed=True):>12} "
                f"{self._fmt_num(r['PerDataset_dMacroAUC'], signed=True):>12} "
                f"{self._fmt_num(r['PerMinusShared_dMacroAUC'], signed=True):>11} "
                f"{self._fmt_num(r['Shared_dSim_calib'], signed=True):>19} "
                f"{self._fmt_num(r['PerDataset_dSim_calib'], signed=True):>20}"
            )
        if len(rows) == 0:
            lines.append("(no datasets)")
        lines.append("")
        lines.append(f"[Saved] {out_path}")
        print("\n".join(lines))
        return df

    def print_scoring_mode_comparison(
        self,
        rows_mixed: List[Dict[str, object]],
        rows_softmax: List[Dict[str, object]],
    ) -> pd.DataFrame:
        by_ds_mixed = {str(r.get("Dataset", "")): r for r in rows_mixed}
        by_ds_soft = {str(r.get("Dataset", "")): r for r in rows_softmax}
        datasets = sorted(set(by_ds_mixed.keys()) | set(by_ds_soft.keys()))

        comp_rows = []
        win_mixed = 0
        win_softmax = 0
        ties = 0
        for ds in datasets:
            rm = by_ds_mixed.get(ds, {})
            rs = by_ds_soft.get(ds, {})
            dm = float(rm.get("Delta_AUC_CAPA_minus_Baseline_Macro", np.nan))
            dsf = float(rs.get("Delta_AUC_CAPA_minus_Baseline_Macro", np.nan))
            em = float(rm.get("ECE_Post", np.nan))
            esf = float(rs.get("ECE_Post", np.nan))
            gap_auc = dsf - dm
            gap_ece = esf - em

            winner = "Tie"
            if np.isfinite(dm) and np.isfinite(dsf):
                if abs(dsf - dm) > 1e-6:
                    winner = "Softmax" if dsf > dm else "Mixed"
                elif np.isfinite(em) and np.isfinite(esf) and abs(esf - em) > 1e-6:
                    winner = "Softmax" if esf < em else "Mixed"
            if winner == "Mixed":
                win_mixed += 1
            elif winner == "Softmax":
                win_softmax += 1
            else:
                ties += 1

            comp_rows.append(
                {
                    "Dataset": ds,
                    "DeltaAUC_Mixed": dm,
                    "DeltaAUC_Softmax": dsf,
                    "Softmax_minus_Mixed_DeltaAUC": gap_auc,
                    "ECE_Post_Mixed": em,
                    "ECE_Post_Softmax": esf,
                    "Softmax_minus_Mixed_ECE": gap_ece,
                    "Winner": winner,
                }
            )

        df = pd.DataFrame(comp_rows)
        out_path = os.path.join(self.config.SAVE_DIR, "scoring_mode_comparison.csv")
        df.to_csv(out_path, index=False)

        header = (
            "Dataset    ΔAUC(mixed)  ΔAUC(softmax)  softmax-mixed  "
            "ECE%(mixed)  ECE%(softmax)  Winner"
        )
        sep = "-" * len(header)
        lines = ["=== Scoring Mode Comparison (same frozen state) ===", "", header, sep]
        for r in comp_rows:
            lines.append(
                f"{r['Dataset']:<10} {self._fmt_num(r['DeltaAUC_Mixed'], signed=True):>11} "
                f"{self._fmt_num(r['DeltaAUC_Softmax'], signed=True):>14} "
                f"{self._fmt_num(r['Softmax_minus_Mixed_DeltaAUC'], signed=True):>14} "
                f"{self._fmt_num(r['ECE_Post_Mixed']):>12} {self._fmt_num(r['ECE_Post_Softmax']):>14} "
                f"{r['Winner']:>7}"
            )
        lines.append("")
        lines.append(f"Summary: Mixed wins={win_mixed}, Softmax wins={win_softmax}, Tie={ties}.")
        lines.append(f"[Saved] {out_path}")
        print("\n".join(lines))
        return df
