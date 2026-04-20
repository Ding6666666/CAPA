from __future__ import annotations

import argparse
import os
from typing import List, Optional

import numpy as np
import torch

from .config import CAPA5Config
from .runner import CAPAExperimentRunner


def _parse_float_csv(value: Optional[str]) -> Optional[List[float]]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        return None
    return [float(part) for part in parts]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CAPAv1-GT clean engineering runner")
    parser.add_argument("--debug", action="store_true", help="Enable detailed debug prints.")
    parser.add_argument(
        "--param-profile",
        type=str,
        default="default",
        choices=["default", "professor"],
        help="Parameter strategy profile: original defaults or professor-style auto rules.",
    )
    parser.add_argument(
        "--label-space",
        type=str,
        default="chexpert5",
        choices=["14", "chexpert5", "unified5"],
        help="Internal semantic label space used end-to-end.",
    )
    parser.add_argument(
        "--source-label-order-profile",
        type=str,
        default="chexpert5_reordered_200x5",
        choices=["default", "chexpert5_reordered_200x5"],
        help="How to interpret source label order when a file does not store class_names.",
    )
    parser.add_argument(
        "--init-temperature",
        type=float,
        default=None,
        help="Override INIT_TEMPERATURE used for default evaluation/calibration.",
    )
    parser.add_argument(
        "--init-scale",
        type=float,
        default=None,
        help="Override INIT_SCALE_FACTOR used for logits.",
    )
    parser.add_argument(
        "--min-classes-adapt",
        type=int,
        default=None,
        help="Override MIN_CLASSES_FOR_ADAPTATION gate.",
    )
    parser.add_argument(
        "--scoring-mode",
        type=str,
        default="mixed",
        choices=["mixed", "softmax"],
        help="Scoring mode for evaluation outputs.",
    )
    parser.add_argument(
        "--sim-source",
        type=str,
        default="gate",
        choices=["gate", "dataset"],
        help="Centroid source for Sim metrics: gate centroids or per-dataset centroids.",
    )
    parser.add_argument(
        "--compare-softmax",
        action="store_true",
        help="Run manuscript validation in both mixed and softmax modes and print a comparison table.",
    )
    parser.add_argument(
        "--cache-mode",
        type=str,
        default="off",
        choices=["off", "gated"],
        help="Cache policy for evaluation: off or gated expert.",
    )
    parser.add_argument(
        "--cache-alpha-max",
        type=float,
        default=0.10,
        help="Maximum sample-wise cache blend alpha.",
    )
    parser.add_argument(
        "--cache-topk",
        type=int,
        default=16,
        help="Top-K nearest neighbors for cache expert.",
    )
    parser.add_argument(
        "--cache-temp",
        type=float,
        default=0.08,
        help="Softmax temperature for cache neighbor weighting.",
    )
    parser.add_argument(
        "--cache-dataset-psi-thr",
        type=float,
        default=0.25,
        help="Dataset-level PSI threshold; cache is disabled for the whole dataset above this drift level.",
    )
    parser.add_argument(
        "--cache-min-sim-q",
        type=float,
        default=0.25,
        help="Reference quantile used to derive sample-level minimum top1 similarity.",
    )
    parser.add_argument(
        "--cache-min-purity-q",
        type=float,
        default=0.50,
        help="Reference quantile used to derive sample-level minimum neighbor purity.",
    )
    parser.add_argument(
        "--cache-max-entropy-q",
        type=float,
        default=0.75,
        help="Reference quantile used to derive sample-level maximum cache entropy.",
    )
    parser.add_argument(
        "--cache-require-agree",
        type=str,
        default="on",
        choices=["on", "off"],
        help="Require cache top1 to agree with the base model before cache can contribute.",
    )
    parser.add_argument(
        "--kappa0",
        type=float,
        default=0.0,
        help="vMF shrink strength for centroid EMA update.",
    )
    parser.add_argument(
        "--n-min-support",
        type=int,
        default=8,
        help="Minimum true support per class to join C_tau (Procrustes/gate).",
    )
    parser.add_argument(
        "--n-cap",
        type=int,
        default=500,
        help="Support cap N_cap in class-weight formula.",
    )
    parser.add_argument(
        "--gamma-weight",
        type=float,
        default=0.5,
        help="Gamma in class-weight formula.",
    )
    parser.add_argument(
        "--beta-weight",
        type=float,
        default=0.2,
        help="Beta in class-weight formula (under-aligned emphasis).",
    )
    parser.add_argument(
        "--hard-neg",
        type=str,
        default="off",
        choices=["on", "off"],
        help="Enable/disable hard-negative Procrustes term: M = M_pos - beta*M_neg.",
    )
    parser.add_argument(
        "--hard-neg-beta",
        type=float,
        default=0.15,
        help="Hard-negative coefficient beta for Procrustes.",
    )
    parser.add_argument(
        "--hard-neg-topk",
        type=int,
        default=3,
        help="Top-K hardest non-matching prototypes per class for M_neg.",
    )
    parser.add_argument(
        "--hard-neg-temp",
        type=float,
        default=0.07,
        help="Softmax temperature for hard-negative weighting.",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.85,
        help="Base rho gate upper bound on post-rotation paired cosine.",
    )
    parser.add_argument(
        "--rho-quantile-gate",
        type=str,
        default="off",
        choices=["on", "off"],
        help="Use quantile-based conservative rho for gate.",
    )
    parser.add_argument(
        "--rho-quantile",
        type=float,
        default=0.70,
        help="Quantile used to derive conservative rho when enabled.",
    )
    parser.add_argument(
        "--gate-offdiag",
        type=str,
        default="off",
        choices=["on", "off"],
        help="Require off-diagonal change to stay below threshold in gate.",
    )
    parser.add_argument(
        "--gate-max-offdiag-delta",
        type=float,
        default=0.0,
        help="Max allowed dOffDiag in gate (<=0 means stricter separation).",
    )
    parser.add_argument(
        "--soft-fusion",
        type=str,
        default="on",
        choices=["on", "off"],
        help="Enable/disable CAPA-baseline soft fusion: lambda*capa + (1-lambda)*baseline.",
    )
    parser.add_argument(
        "--soft-fusion-lambda",
        type=float,
        default=1.0,
        help="Lambda for CAPA-baseline soft fusion.",
    )
    parser.add_argument(
        "--guarded-alphas",
        type=str,
        default=None,
        help="Comma-separated guarded alpha candidates for GT soft fallback, e.g. 1.0,0.85,0.7,0.55,0.4",
    )
    parser.add_argument(
        "--go-guardian",
        type=str,
        default="off",
        choices=["on", "off"],
        help="Enable/disable GO Guardian Stage-1 (PSI monitoring + freeze/rollback).",
    )
    parser.add_argument(
        "--go-psi-window",
        type=int,
        default=512,
        help="Sliding window size for Guardian PSI.",
    )
    parser.add_argument(
        "--go-psi-bins",
        type=int,
        default=10,
        help="Histogram bins for Guardian PSI.",
    )
    parser.add_argument(
        "--go-psi-thr",
        type=float,
        default=2.0,
        help="PSI alarm threshold; above this triggers freeze.",
    )
    parser.add_argument(
        "--go-tau-resume",
        type=float,
        default=1.0,
        help="Resume threshold; need consecutive windows below this to unfreeze.",
    )
    parser.add_argument(
        "--go-resume-windows",
        type=int,
        default=3,
        help="Consecutive low-PSI windows required to resume from freeze.",
    )
    parser.add_argument(
        "--go-warmup-steps",
        type=int,
        default=50,
        help="GO disabled before this step index.",
    )
    parser.add_argument(
        "--go-baseline-collect-steps",
        type=int,
        default=50,
        help="Steps used to collect GO baseline histogram after GO warmup.",
    )
    parser.add_argument(
        "--go-dry-run",
        type=str,
        default="off",
        choices=["on", "off"],
        help="Dry-run GO alarms without freezing/rollback.",
    )
    parser.add_argument(
        "--go-ml-proj",
        type=str,
        default="on",
        choices=["on", "off"],
        help="Enable/disable GO multi-label order-free residual projection updates.",
    )
    parser.add_argument(
        "--go-ml-tau",
        type=float,
        default=1e-2,
        help="Base ridge tau for GO multi-label projection.",
    )
    parser.add_argument(
        "--go-ml-cond-target",
        type=float,
        default=1e3,
        help="Target condition-number upper bound for (T^T T + tau I).",
    )
    parser.add_argument(
        "--go-ml-residual-norm",
        type=str,
        default="on",
        choices=["on", "off"],
        help="Use residual norm as extra sample weight in GO multi-label update.",
    )
    parser.add_argument(
        "--go-ml-signal",
        type=str,
        default="original",
        choices=["original", "residual"],
        help="Signal vector for GO multi-label update: original embedding or residual direction.",
    )
    parser.add_argument(
        "--compare-per-dataset-capa",
        action="store_true",
        help="Run experimental shared-CAPA vs per-dataset-CAPA comparison.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Optional output directory. If omitted, use config default SAVE_DIR.",
    )
    return parser


def build_config(args: argparse.Namespace) -> CAPA5Config:
    base_cfg = CAPA5Config(
        LABEL_SPACE=str(args.label_space),
        PARAM_PROFILE=str(args.param_profile),
        SOURCE_LABEL_ORDER_PROFILE=str(args.source_label_order_profile),
    )
    guarded_alphas_override = _parse_float_csv(args.guarded_alphas)
    return CAPA5Config(
        PARAM_PROFILE=str(args.param_profile),
        LABEL_SPACE=str(args.label_space),
        SOURCE_LABEL_ORDER_PROFILE=str(args.source_label_order_profile),
        DEBUG=bool(args.debug),
        VERBOSE=bool(args.debug),
        PRINT_SUMMARY=False,
        INIT_TEMPERATURE=(
            float(args.init_temperature)
            if args.init_temperature is not None
            else float(base_cfg.INIT_TEMPERATURE)
        ),
        INIT_SCALE_FACTOR=(
            float(args.init_scale) if args.init_scale is not None else float(base_cfg.INIT_SCALE_FACTOR)
        ),
        MIN_CLASSES_FOR_ADAPTATION=(
            max(1, int(args.min_classes_adapt))
            if args.min_classes_adapt is not None
            else int(base_cfg.MIN_CLASSES_FOR_ADAPTATION)
        ),
        SCORING_MODE=str(args.scoring_mode),
        SIM_SOURCE=str(args.sim_source),
        CACHE_MODE=str(args.cache_mode),
        CACHE_ALPHA_MAX=min(1.0, max(0.0, float(args.cache_alpha_max))),
        CACHE_TOPK=max(1, int(args.cache_topk)),
        CACHE_TEMP=max(1e-4, float(args.cache_temp)),
        CACHE_DATASET_PSI_THR=max(0.0, float(args.cache_dataset_psi_thr)),
        CACHE_MIN_SIM_Q=min(1.0, max(0.0, float(args.cache_min_sim_q))),
        CACHE_MIN_PURITY_Q=min(1.0, max(0.0, float(args.cache_min_purity_q))),
        CACHE_MAX_ENTROPY_Q=min(1.0, max(0.0, float(args.cache_max_entropy_q))),
        CACHE_REQUIRE_AGREE=(str(args.cache_require_agree).lower() == "on"),
        KAPPA0=max(0.0, float(args.kappa0)),
        N_MIN_SUPPORT_FOR_ACTIVE=max(1, int(args.n_min_support)),
        N_CAP=max(1, int(args.n_cap)),
        GAMMA_WEIGHT=max(0.0, float(args.gamma_weight)),
        BETA_WEIGHT=max(0.0, float(args.beta_weight)),
        ENABLE_HARD_NEG_PROCRUSTES=(str(args.hard_neg).lower() == "on"),
        HARD_NEG_BETA=max(0.0, float(args.hard_neg_beta)),
        HARD_NEG_TOPK=max(1, int(args.hard_neg_topk)),
        HARD_NEG_TEMP=max(1e-4, float(args.hard_neg_temp)),
        RHO=float(args.rho),
        GATE_USE_RHO_QUANTILE=(str(args.rho_quantile_gate).lower() == "on"),
        RHO_QUANTILE=min(1.0, max(0.0, float(args.rho_quantile))),
        GATE_REQUIRE_OFFDIAG_IMPROVEMENT=(str(args.gate_offdiag).lower() == "on"),
        GATE_MAX_OFFDIAG_DELTA=float(args.gate_max_offdiag_delta),
        ENABLE_CAPA_BASELINE_SOFT_FUSION=(str(args.soft_fusion).lower() == "on"),
        CAPA_BASELINE_FUSION_LAMBDA=min(1.0, max(0.0, float(args.soft_fusion_lambda))),
        CAPAV1_GUARDED_ALPHAS=(
            guarded_alphas_override
            if guarded_alphas_override is not None
            else list(base_cfg.CAPAV1_GUARDED_ALPHAS)
        ),
        ENABLE_GO_GUARDIAN=(str(args.go_guardian).lower() == "on"),
        GO_PSI_WINDOW=max(16, int(args.go_psi_window)),
        GO_PSI_BINS=max(5, int(args.go_psi_bins)),
        GO_PSI_THR=max(0.0, float(args.go_psi_thr)),
        GO_TAU_RESUME=max(0.0, float(args.go_tau_resume)),
        GO_RESUME_WINDOWS=max(1, int(args.go_resume_windows)),
        GO_WARMUP_STEPS=max(0, int(args.go_warmup_steps)),
        GO_BASELINE_COLLECT_STEPS=max(0, int(args.go_baseline_collect_steps)),
        GO_DRY_RUN=(str(args.go_dry_run).lower() == "on"),
        ENABLE_GO_MULTILABEL_PROJECTION=(str(args.go_ml_proj).lower() == "on"),
        GO_ML_TAU_BASE=max(1e-8, float(args.go_ml_tau)),
        GO_ML_COND_TARGET=max(1.01, float(args.go_ml_cond_target)),
        GO_ML_USE_RESIDUAL_NORM_WEIGHT=(str(args.go_ml_residual_norm).lower() == "on"),
        GO_ML_SIGNAL_USE_ORIGINAL=(str(args.go_ml_signal).lower() == "original"),
        SAVE_DIR=(str(args.save_dir) if args.save_dir else base_cfg.SAVE_DIR),
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = build_config(args)
    runner = CAPAExperimentRunner(config)

    if config.VERBOSE:
        print("[Checks] Quick sanity checks")
        for path in [config.CALIB_DATA_PATH, config.TRAIN_DATA_PATH] + list(config.TEST_DATA_PATHS.values()):
            if not os.path.exists(path):
                print(f"  [WARN] Missing path: {path}")
                continue
            z, y, is_multi = runner._load_data(path, is_calibration=("calib" in path.lower()))
            dim = z.shape[1]
            print(f"  {os.path.basename(path)} -> N={len(z)}, D={dim}, labels_present={y is not None}, multi={is_multi}")

    runner._build_prototypes()
    if config.VERBOSE:
        print("  prototypes raw/processed shapes:", runner.t_raw_pooled_raw.shape, runner.t_raw_pooled.shape)
        assert runner.t_raw_pooled.shape == runner.t_raw_pooled_raw.shape

    if runner.support_counts is None:
        runner.support_counts = (
            torch.ones(len(runner.config.ORDERED_CLASS_NAMES), device=runner.device)
            * runner.config.N_MIN_SUPPORT_FOR_ACTIVE
        )
    if runner.image_centroids is None:
        runner.image_centroids = runner.t_raw_pooled.clone()
    if runner.current_R is None:
        runner.current_R = torch.eye(runner.t_raw_pooled.shape[1], device=runner.device)

    if config.VERBOSE and runner.W_zca is not None:
        wt = runner.W_zca.cpu().numpy()
        i_approx = wt @ np.linalg.pinv(wt)
        print("  ZCA approx identity max_err:", np.max(np.abs(i_approx - np.eye(i_approx.shape[0]))))

    dim = runner.t_raw_pooled.shape[1]
    eye = torch.eye(dim, device=runner.device)
    if config.VERBOSE:
        rtest = runner._solve_procrustes()
        ortho_err = torch.norm(rtest @ rtest.T - eye).item()
        print("  Procrustes R ortho_err:", ortho_err)
        assert ortho_err < 1e-3, "Procrustes R not sufficiently orthogonal"

    runner.run_pipeline()

    if args.compare_softmax:
        rows_mixed = runner.run_manuscript_validation(scoring_mode="mixed", sim_source=config.SIM_SOURCE)
        rows_softmax = runner.run_manuscript_validation(scoring_mode="softmax", sim_source=config.SIM_SOURCE)
        runner.print_scoring_mode_comparison(rows_mixed, rows_softmax)
        rows_selected = rows_mixed if config.SCORING_MODE == "mixed" else rows_softmax
        runner.print_final_gate_summary(rows_selected, sim_source=config.SIM_SOURCE)
        runner.print_three_way_auc_summary(rows_selected)
        if args.compare_per_dataset_capa:
            runner.run_shared_vs_per_dataset_capa(rows_selected, scoring_mode=config.SCORING_MODE)
    else:
        final_rows = runner.run_manuscript_validation(
            scoring_mode=config.SCORING_MODE,
            sim_source=config.SIM_SOURCE,
        )
        runner.print_final_gate_summary(final_rows, sim_source=config.SIM_SOURCE)
        runner.print_three_way_auc_summary(final_rows)
        if args.compare_per_dataset_capa:
            runner.run_shared_vs_per_dataset_capa(final_rows, scoring_mode=config.SCORING_MODE)
    return 0


__all__ = ["main", "CAPA5Config", "CAPA5NotebookRunner", "CAPAExperimentRunner"]
