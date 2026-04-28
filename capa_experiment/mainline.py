import os
import json
import warnings
import copy
from pathlib import Path
# ---- Performance / stability guards for sklearn / MKL on Windows ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
# ---------------------------------------------------------------------
import pickle
import random
import sys
from datetime import datetime
import numpy as np
import pandas as pd 
import torch
import torch.nn.functional as F
import open_clip
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple, Optional
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import beta as beta_dist, norm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve, auc
from sklearn.calibration import calibration_curve
from sklearn.exceptions import UndefinedMetricWarning
from dataclasses import dataclass, field
from tqdm import tqdm

# Silence noisy warnings for clean experiment logs (set VERBOSE=True to inspect details)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def _resolve_project_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [here.parents[1], here.parents[2]]
    for candidate in candidates:
        if any((candidate / name).exists() for name in ("data", "model", "results")):
            return candidate
    return here.parents[1]


PROJECT_ROOT = _resolve_project_root()
DEFAULT_MODEL_DIR = PROJECT_ROOT / "model"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
ENTRY_SCRIPT_PATH = str(Path(__file__).resolve())
DEFAULT_TRAIN_SOURCE_FILENAME = "data_train.pkl"
DEFAULT_CALIB_SUBSET_FILENAME = "data_train_image_calibration_5000.pkl"
DEFAULT_CALIB_SUBSET_SIZE = 5000
DEFAULT_TARGET_AWARE_TRAIN_FILENAME = "data_train_chexpert5_target_positive.pkl"
DEFAULT_TARGET_AWARE_CALIB_FILENAME = "data_train_chexpert5_target_positive_image_calibration.pkl"
DEFAULT_TARGET_AWARE_SUMMARY_FILENAME = "data_train_chexpert5_target_positive_summary.json"


@dataclass(frozen=True)
class EvalModeSpec:
    prototype_key: str
    final_logits_source: str
    image_preprocessing: bool
    alignment: bool
    test_time_adaptation: bool
    guarded_alignment: bool
    dual_track: bool
    cache: bool
    guardian: bool
    prior_correction: bool
    calibration: bool
    runtime_scale: bool
    soft_fusion: bool
    offdiag_gate: bool
    deploy_overrides: bool
    notes: str


EVAL_MODE_SPECS: Dict[str, EvalModeSpec] = {
    "raw_baseline": EvalModeSpec(
        prototype_key="t_raw_text",
        final_logits_source="raw_direct",
        image_preprocessing=False,
        alignment=False,
        test_time_adaptation=False,
        guarded_alignment=False,
        dual_track=False,
        cache=False,
        guardian=False,
        prior_correction=False,
        calibration=False,
        runtime_scale=False,
        soft_fusion=False,
        offdiag_gate=False,
        deploy_overrides=False,
        notes="Frozen backbone with raw image/text embeddings and direct scoring only.",
    ),
    "preprocessed_baseline": EvalModeSpec(
        prototype_key="t_processed_text",
        final_logits_source="processed_direct",
        image_preprocessing=True,
        alignment=False,
        test_time_adaptation=False,
        guarded_alignment=False,
        dual_track=False,
        cache=False,
        guardian=False,
        prior_correction=False,
        calibration=False,
        runtime_scale=False,
        soft_fusion=False,
        offdiag_gate=False,
        deploy_overrides=False,
        notes="Optional diagnostic baseline in the shared processed feature space, without CAPA alignment or deployment routing.",
    ),
    "full_capa": EvalModeSpec(
        prototype_key="t_aligned_text",
        final_logits_source="aligned_full",
        image_preprocessing=True,
        alignment=True,
        test_time_adaptation=True,
        guarded_alignment=True,
        dual_track=True,
        cache=True,
        guardian=True,
        prior_correction=True,
        calibration=True,
        runtime_scale=True,
        soft_fusion=True,
        offdiag_gate=True,
        deploy_overrides=True,
        notes="Complete CAPA path in the shared processed feature space, including official deployment-time routing controls.",
    ),
}
EVAL_MODE_ORDER: Tuple[str, ...] = tuple(EVAL_MODE_SPECS.keys())
MAIN_EVAL_MODE_ORDER: Tuple[str, ...] = ("raw_baseline", "full_capa")
EVAL_MODE_ALIASES: Dict[str, str] = {
    "baseline": "preprocessed_baseline",
    "processed_baseline": "preprocessed_baseline",
}


def _json_default(value: Any):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)

FULL_14_CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion",
    "Pneumonia", "Pneumothorax", "Fracture", "Lung Lesion", "Lung Opacity",
    "Pleural Other", "Enlargement of the Cardiac Silhouette", "Pneumoperitoneum", "Support Devices",
]
CHEXPERT_5_CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion",
]
CHEXPERT5_REORDERED_ACTIVE_INDEX_TO_LABEL = {
    2: "Atelectasis",
    5: "Cardiomegaly",
    6: "Edema",
    8: "Consolidation",
    10: "Pleural Effusion",
}


def _build_chexpert5_reordered_source_order_14() -> List[str]:
    order: List[Optional[str]] = [None] * len(FULL_14_CLASS_NAMES)
    used = set(CHEXPERT5_REORDERED_ACTIVE_INDEX_TO_LABEL.values())
    for idx, name in CHEXPERT5_REORDERED_ACTIVE_INDEX_TO_LABEL.items():
        order[idx] = name
    remaining = [name for name in FULL_14_CLASS_NAMES if name not in used]
    rem_iter = iter(remaining)
    for i in range(len(order)):
        if order[i] is None:
            order[i] = next(rem_iter)
    return [str(x) for x in order]


CHEXPERT5_REORDERED_SOURCE_ORDER_14 = _build_chexpert5_reordered_source_order_14()
UNIFIED_5_CLASS_NAMES = [
    "Consolidation", "Pneumonia", "Pneumothorax", "Lung Lesion", "Pleural Other",
]
MEDICAL_SYNONYM_MAP_CHEXPERT5 = {
    "Atelectasis": ["atelectasis", "collapsed lung tissue"],
    "Cardiomegaly": ["cardiomegaly", "enlarged heart"],
    "Consolidation": ["consolidation", "lung consolidation"],
    "Edema": ["edema", "pulmonary edema", "fluid overload"],
    "Pleural Effusion": ["pleural effusion", "fluid in pleural space"],
}
MEDICAL_SYNONYM_MAP_14 = {
    "Atelectasis": ["atelectasis", "collapsed lung tissue"],
    "Cardiomegaly": ["cardiomegaly", "enlarged heart"],
    "Consolidation": ["consolidation", "lung consolidation"],
    "Edema": ["edema", "pulmonary edema", "fluid overload"],
    "Pleural Effusion": ["pleural effusion", "fluid in pleural space"],
    "Pneumonia": ["pneumonia", "lung infection"],
    "Pneumothorax": ["pneumothorax", "collapsed lung"],
    "Fracture": ["fracture", "bone fracture", "rib fracture"],
    "Lung Lesion": ["lung lesion", "pulmonary nodule", "mass"],
    "Lung Opacity": ["lung opacity", "white lung"],
    "Pleural Other": ["pleural abnormality", "pleural thickening"],
    "Enlargement of the Cardiac Silhouette": ["enlarged cardiac silhouette", "large heart shadow"],
    "Pneumoperitoneum": ["pneumoperitoneum", "free air under diaphragm"],
    "Support Devices": ["support devices", "medical tubes", "lines", "pacemaker"],
}
MEDICAL_SYNONYM_MAP_5 = {
    "Consolidation": ["consolidation", "lung consolidation"],
    "Pneumonia": ["pneumonia", "lung infection"],
    "Pneumothorax": ["pneumothorax", "collapsed lung"],
    "Lung Lesion": ["lung lesion", "pulmonary nodule", "mass"],
    "Pleural Other": ["pleural abnormality", "pleural thickening"],
}
PROMPT_BUCKET_ORDER = [
    "impression",
    "finding",
    "anatomy",
    "morphology",
    "severity",
    "context",
]
PROMPT_BUCKET_TEMPLATES_V1 = {
    "impression": [
        "Impression: {phrase}.",
        "Overall impression: {phrase}.",
    ],
    "finding": [
        "Chest radiograph shows {phrase}.",
        "There is {phrase}.",
    ],
    "anatomy": [
        "{phrase} is seen on this chest radiograph.",
        "This frontal chest X-ray demonstrates {phrase}.",
    ],
    "morphology": [
        "Imaging shows {phrase}.",
        "A chest X-ray pattern of {phrase} is present.",
    ],
    "severity": [
        "Findings are consistent with {phrase}.",
        "This study shows {phrase}.",
    ],
    "context": [
        "On frontal chest radiograph, {phrase}.",
        "Portable chest radiograph demonstrates {phrase}.",
    ],
}
STRUCTURED_PROMPT_PHRASE_BANK_CHEXPERT5 = {
    "Atelectasis": {
        "impression": [
            "mild bibasilar atelectatic change",
            "subsegmental atelectatic opacity",
        ],
        "finding": [
            "atelectatic opacity",
            "collapsed lung tissue",
        ],
        "anatomy": [
            "basilar atelectatic change",
            "linear bibasal atelectatic opacity",
        ],
        "morphology": [
            "volume-loss related linear opacity",
            "subsegmental plate-like atelectatic change",
        ],
        "severity": [
            "mild atelectasis",
            "moderate bibasal atelectatic change",
        ],
        "context": [
            "low-volume bibasal atelectatic opacity is present",
        ],
    },
    "Cardiomegaly": {
        "impression": [
            "cardiomegaly",
            "enlargement of the cardiac silhouette",
        ],
        "finding": [
            "an enlarged cardiac silhouette",
            "cardiomediastinal enlargement",
        ],
        "anatomy": [
            "enlargement of the cardiomediastinal silhouette",
            "mild enlargement of the heart shadow",
        ],
        "morphology": [
            "globular cardiac enlargement",
            "prominent cardiomediastinal contour",
        ],
        "severity": [
            "mild cardiomegaly",
            "marked enlargement of the cardiac silhouette",
        ],
        "context": [
            "frontal chest radiograph shows cardiomediastinal enlargement",
        ],
    },
    "Consolidation": {
        "impression": [
            "focal consolidation",
            "patchy airspace consolidation",
        ],
        "finding": [
            "airspace consolidation",
            "focal lung consolidation",
        ],
        "anatomy": [
            "lobar airspace opacity",
            "unilateral patchy airspace opacity",
        ],
        "morphology": [
            "dense focal airspace opacity",
            "multifocal patchy consolidative opacity",
        ],
        "severity": [
            "mild focal consolidation",
            "extensive airspace consolidation",
        ],
        "context": [
            "frontal chest radiograph demonstrates focal airspace consolidation",
        ],
    },
    "Edema": {
        "impression": [
            "pulmonary edema",
            "diffuse edema pattern",
        ],
        "finding": [
            "bilateral pulmonary edema",
            "interstitial edema",
        ],
        "anatomy": [
            "perihilar edema opacity",
            "diffuse bilateral pulmonary opacification related to edema",
        ],
        "morphology": [
            "interstitial to alveolar edema pattern",
            "bilateral hazy perihilar opacity",
        ],
        "severity": [
            "mild pulmonary edema",
            "marked diffuse pulmonary edema",
        ],
        "context": [
            "portable chest radiograph shows bilateral pulmonary edema pattern",
        ],
    },
    "Pleural Effusion": {
        "impression": [
            "pleural effusion",
            "bilateral pleural effusions",
        ],
        "finding": [
            "pleural fluid",
            "fluid in the pleural space",
        ],
        "anatomy": [
            "blunting of the costophrenic angle from pleural fluid",
            "dependent pleural fluid collection",
        ],
        "morphology": [
            "small pleural effusion with meniscus configuration",
            "layering pleural fluid opacity",
        ],
        "severity": [
            "small pleural effusion",
            "moderate to large pleural effusion",
        ],
        "context": [
            "frontal chest radiograph demonstrates pleural fluid with blunted costophrenic angle",
        ],
    },
}
STRUCTURED_PROMPT_PHRASE_BANK_CHEXPERT5_V2 = {
    "Atelectasis": {
        "impression": [
            "mild bibasal atelectatic change",
            "subsegmental atelectatic change",
        ],
        "finding": [
            "linear atelectatic opacity",
            "basilar atelectatic opacity",
        ],
        "anatomy": [
            "bibasal linear opacity with volume loss",
            "left basilar atelectatic change",
        ],
        "morphology": [
            "plate-like atelectatic opacity",
            "low-volume linear opacity",
        ],
        "severity": [
            "mild atelectatic change",
        ],
    },
    "Cardiomegaly": {
        "impression": [
            "cardiomegaly",
            "mild cardiomegaly",
        ],
        "finding": [
            "enlarged cardiac silhouette",
            "cardiomediastinal enlargement",
        ],
        "anatomy": [
            "enlargement of the cardiac silhouette",
            "prominent cardiomediastinal silhouette",
        ],
        "morphology": [
            "globular enlargement of the cardiac silhouette",
            "cardiac silhouette enlargement",
        ],
        "severity": [
            "marked cardiomegaly",
        ],
    },
    "Consolidation": {
        "impression": [
            "focal airspace consolidation",
            "patchy airspace opacity",
        ],
        "finding": [
            "focal airspace opacity",
            "patchy consolidative opacity",
        ],
        "anatomy": [
            "right basilar airspace opacity",
            "lobar airspace opacity",
        ],
        "morphology": [
            "dense consolidative opacity",
            "multifocal patchy airspace opacity",
        ],
        "severity": [
            "mild focal consolidation",
        ],
    },
    "Edema": {
        "impression": [
            "pulmonary edema",
            "mild pulmonary edema",
        ],
        "finding": [
            "bilateral interstitial pulmonary opacity",
            "bilateral perihilar hazy opacity",
        ],
        "anatomy": [
            "bilateral perihilar opacity",
            "diffuse bilateral pulmonary opacity",
        ],
        "morphology": [
            "interstitial edema pattern",
            "alveolar edema pattern",
        ],
        "severity": [
            "marked bilateral pulmonary edema",
        ],
    },
    "Pleural Effusion": {
        "impression": [
            "pleural effusion",
            "small pleural effusion",
        ],
        "finding": [
            "pleural fluid",
            "blunting of the costophrenic angle",
        ],
        "anatomy": [
            "left pleural effusion",
            "bilateral pleural effusions",
        ],
        "morphology": [
            "layering pleural fluid",
            "meniscus-like pleural opacity",
        ],
        "severity": [
            "moderate pleural effusion",
        ],
    },
}
STRUCTURED_PROMPT_PHRASE_BANK_CHEXPERT5_REPORT = {
    "Atelectasis": {
        "impression": [
            "mild bibasal atelectatic change",
            "low-volume bibasal atelectatic opacity",
        ],
        "finding": [
            "bibasal linear atelectatic opacity",
            "subsegmental atelectatic change",
        ],
        "anatomy": [
            "left basilar atelectatic change",
            "right basilar plate-like atelectatic opacity",
        ],
        "morphology": [
            "linear bibasal opacity with volume loss",
        ],
    },
    "Cardiomegaly": {
        "impression": [
            "mild cardiomegaly",
            "enlargement of the cardiac silhouette",
        ],
        "finding": [
            "cardiomediastinal silhouette is enlarged",
            "mild enlargement of the heart shadow",
        ],
        "anatomy": [
            "prominent cardiomediastinal silhouette",
            "mild cardiac silhouette enlargement",
        ],
        "morphology": [
            "stable appearing enlargement of the cardiac silhouette",
        ],
    },
    "Consolidation": {
        "impression": [
            "focal airspace opacity",
            "patchy basilar airspace opacity",
        ],
        "finding": [
            "patchy lower lung airspace opacity",
            "focal consolidative opacity",
        ],
        "anatomy": [
            "right basilar airspace opacity",
            "left lower lung consolidative opacity",
        ],
        "morphology": [
            "patchy focal consolidative change",
        ],
    },
    "Edema": {
        "impression": [
            "mild pulmonary edema",
            "mild diffuse interstitial edema pattern",
        ],
        "finding": [
            "bilateral perihilar interstitial opacity",
            "diffuse bilateral pulmonary vascular-interstitial opacity",
        ],
        "anatomy": [
            "mild bilateral perihilar hazy opacity",
            "central bilateral interstitial pulmonary opacity",
        ],
        "morphology": [
            "mild interstitial pulmonary edema pattern",
        ],
    },
    "Pleural Effusion": {
        "impression": [
            "small pleural effusion",
            "small bilateral pleural effusions",
        ],
        "finding": [
            "trace pleural fluid",
            "blunting of the costophrenic angles from pleural fluid",
        ],
        "anatomy": [
            "small left pleural effusion",
            "small bilateral pleural effusions with blunted costophrenic angles",
        ],
        "morphology": [
            "small layering pleural fluid opacity",
        ],
    },
}
EARLY_TEXT_SUPPORT_SYNONYMS_CHEXPERT5 = {
    "Atelectasis": [
        "atelectasis",
        "subsegmental atelectasis",
        "linear atelectasis",
        "basilar atelectasis",
        "bibasal atelectatic change",
        "linear atelectatic opacity",
        "plate-like atelectatic opacity",
        "low-volume atelectatic change",
        "volume-loss related atelectatic opacity",
        "dependent atelectasis",
        "segmental atelectatic change",
        "discoid atelectatic opacity",
    ],
    "Cardiomegaly": [
        "cardiomegaly",
        "mild cardiomegaly",
        "moderate cardiomegaly",
        "marked cardiomegaly",
        "enlarged heart",
        "enlarged cardiac silhouette",
        "cardiac silhouette enlargement",
        "prominent cardiac silhouette",
        "cardiomediastinal enlargement",
        "enlargement of the cardiac silhouette",
        "increased cardiothoracic ratio",
        "large cardiac silhouette",
    ],
    "Consolidation": [
        "consolidation",
        "airspace consolidation",
        "focal airspace consolidation",
        "patchy airspace consolidation",
        "lobar consolidation",
        "segmental consolidation",
        "consolidative opacity",
        "patchy consolidative opacity",
        "dense airspace opacity",
        "confluent airspace opacity",
        "focal airspace opacity",
        "basilar consolidative opacity",
    ],
    "Edema": [
        "pulmonary edema",
        "mild pulmonary edema",
        "interstitial pulmonary edema",
        "alveolar pulmonary edema",
        "interstitial edema pattern",
        "alveolar edema pattern",
        "bilateral perihilar edema",
        "pulmonary vascular congestion with edema",
        "diffuse bilateral edema pattern",
        "fluid overload pulmonary edema",
        "bilateral hazy perihilar opacity from edema",
        "diffuse interstitial pulmonary opacity from edema",
    ],
    "Pleural Effusion": [
        "pleural effusion",
        "small pleural effusion",
        "moderate pleural effusion",
        "bilateral pleural effusions",
        "pleural fluid",
        "layering pleural fluid",
        "dependent pleural fluid",
        "pleural fluid collection",
        "blunting of the costophrenic angle",
        "costophrenic angle blunting from pleural fluid",
        "meniscus-like pleural opacity",
        "basal pleural fluid opacity",
    ],
}
EARLY_TEXT_SUPPORT_REPORT_TEMPLATES = [
    "Impression: {finding}.",
    "Findings: {finding}.",
    "Chest radiograph demonstrates {finding}.",
    "Frontal chest radiograph demonstrates {finding}.",
    "Radiographic findings are consistent with {finding}.",
]
PROMPT_BUCKET_TEMPLATES_V2 = {
    "impression": [
        "Impression: {phrase}.",
    ],
    "finding": [
        "Chest radiograph shows {phrase}.",
    ],
    "anatomy": [
        "{phrase}.",
    ],
    "morphology": [
        "Imaging shows {phrase}.",
    ],
    "severity": [
        "Findings are consistent with {phrase}.",
    ],
}
PROMPT_BUCKET_TEMPLATES_REPORT = {
    "impression": [
        "Impression: {phrase}.",
    ],
    "finding": [
        "Findings: {phrase}.",
    ],
    "anatomy": [
        "{phrase}.",
    ],
    "morphology": [
        "Chest radiograph demonstrates {phrase}.",
    ],
}
DEFAULT_PROMPT_BUCKET_PRIORS = {
    "impression": 1.00,
    "finding": 1.15,
    "anatomy": 1.10,
    "morphology": 1.10,
    "severity": 0.95,
    "context": 0.80,
}
PROMPT_BANK_PROFILE_BUCKET_ORDER = {
    "v1": list(PROMPT_BUCKET_ORDER),
    "v2": ["impression", "finding", "anatomy", "morphology", "severity"],
    "v3": ["legacy", "impression", "finding", "anatomy", "morphology", "severity"],
    "visual": ["impression", "finding", "anatomy", "morphology", "severity"],
    "report": ["impression", "finding", "anatomy", "morphology"],
}
PROMPT_BANK_PROFILE_BUCKET_TEMPLATES = {
    "v1": PROMPT_BUCKET_TEMPLATES_V1,
    "v2": PROMPT_BUCKET_TEMPLATES_V2,
    "v3": PROMPT_BUCKET_TEMPLATES_V2,
    "visual": PROMPT_BUCKET_TEMPLATES_V2,
    "report": PROMPT_BUCKET_TEMPLATES_REPORT,
}
PROMPT_BANK_PROFILE_PHRASE_BANK = {
    "v1": STRUCTURED_PROMPT_PHRASE_BANK_CHEXPERT5,
    "v2": STRUCTURED_PROMPT_PHRASE_BANK_CHEXPERT5_V2,
    "v3": STRUCTURED_PROMPT_PHRASE_BANK_CHEXPERT5_V2,
    "visual": STRUCTURED_PROMPT_PHRASE_BANK_CHEXPERT5_V2,
    "report": STRUCTURED_PROMPT_PHRASE_BANK_CHEXPERT5_REPORT,
}
PROMPT_BANK_PROFILE_PRIORS = {
    "v1": dict(DEFAULT_PROMPT_BUCKET_PRIORS),
    "v2": {
        "impression": 1.20,
        "finding": 1.20,
        "anatomy": 1.05,
        "morphology": 1.00,
        "severity": 0.70,
    },
    "v3": {
        "legacy": 1.35,
        "impression": 1.10,
        "finding": 1.10,
        "anatomy": 0.95,
        "morphology": 0.90,
        "severity": 0.65,
    },
    "visual": {
        "impression": 1.15,
        "finding": 1.20,
        "anatomy": 1.05,
        "morphology": 1.00,
        "severity": 0.65,
    },
    "report": {
        "impression": 1.30,
        "finding": 1.15,
        "anatomy": 1.00,
        "morphology": 0.85,
    },
}
PROMPT_CLASS_MIX_PROFILE_MAP = {
    "none": {},
    "cxr_conservative": {
        "Atelectasis": 0.18,
        "Cardiomegaly": 0.05,
        "Consolidation": 0.30,
        "Edema": 0.06,
        "Pleural Effusion": 0.22,
    },
    "mimic_hybrid": {
        "Atelectasis": 0.30,
        "Cardiomegaly": 0.05,
        "Consolidation": 0.16,
        "Edema": 0.05,
        "Pleural Effusion": 0.28,
    },
}
BINARY_POSITIVE_CLASS_MAP_14 = {
    "default": ["Pneumonia"],
    "COVID": ["Pneumonia"],
    "RSNA": ["Pneumonia"],
}
BINARY_POSITIVE_CLASS_MAP_5 = {
    "default": ["Pneumonia"],
    "COVID": ["Consolidation", "Pneumonia"],
    "RSNA": ["Pneumonia"],
}
BINARY_POSITIVE_CLASS_MAP_CHEXPERT5 = {
    "default": ["Consolidation"],
    "COVID": ["Consolidation"],
    "RSNA": ["Consolidation"],
}

@dataclass
class CAPA5Config:
    RANDOM_SEED: int = 42
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME: str = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    LOCAL_MODEL_PATH: str = str(DEFAULT_MODEL_DIR)
    DATA_ROOT: str = str(DEFAULT_DATA_DIR)
    # Source 14-label pool. The mainline materializes a CheXpert-5 target-aware view from it.
    TRAIN_SOURCE_DATA_PATH: str = rf"{DATA_ROOT}\{DEFAULT_TRAIN_SOURCE_FILENAME}"
    # Main CheXpert-5 target-aware training pool: rows containing at least one target finding.
    TRAIN_DATA_PATH: str = rf"{DATA_ROOT}\{DEFAULT_TARGET_AWARE_TRAIN_FILENAME}"
    # Cross-modal pool used for retrieval-style analysis only.
    CROSS_MODAL_DATA_PATH: str = rf"{DATA_ROOT}\CHEXPERT_MIMIC.pkl"
    # Dedicated target-aware image-only calibration subset, materialized from TRAIN_SOURCE_DATA_PATH if missing.
    CALIB_DATA_PATH: str = rf"{DATA_ROOT}\{DEFAULT_TARGET_AWARE_CALIB_FILENAME}"
    # Post-hoc temperature scaling should be fit on a held-out calibration set (NOT test sets).
    TAU_CALIB_DATA_PATH: str = rf"{DATA_ROOT}\{DEFAULT_TARGET_AWARE_CALIB_FILENAME}"
    TAU_CALIB_FRAC: float = 0.2
    TAU_CALIB_MAX: int = 5000
    TAU_CALIB_MIN: int = 2000

    # === Logging ===
    VERBOSE: bool = False
    PRINT_SUMMARY: bool = True
    DEBUG: bool = False
    SAVE_DIR: str = str(DEFAULT_RESULTS_DIR)
    ENTRY_SCRIPT: str = ENTRY_SCRIPT_PATH
    AUC_BOOTSTRAP_ROUNDS: int = 1000
    
    TEST_DATA_PATHS: Dict[str, str] = field(default_factory=lambda: {
        "CheXpert": str(DEFAULT_DATA_DIR / "cheXpert_200x5.pkl"),
        "MIMIC":    str(DEFAULT_DATA_DIR / "MIMIC_200x5.pkl"),
        "COVID":    str(DEFAULT_DATA_DIR / "COVID_3616x2.pkl"),
        "RSNA":     str(DEFAULT_DATA_DIR / "RSNA_4243x2.pkl"),
    })
    PARAM_PROFILE: str = "default"
    LABEL_SPACE: str = "chexpert5"
    SOURCE_LABEL_ORDER_PROFILE: str = "chexpert5_reordered_200x5"
    SOURCE_CLASS_NAMES_14: List[str] = field(default_factory=lambda: list(FULL_14_CLASS_NAMES))
    ORDERED_CLASS_NAMES: List[str] = field(default_factory=list)

    # === GT support + update ===
    MULTI_LABEL_RIDGE: float = 0.1  
    KAPPA_EMA: float = 25         
    
    # === Warm-up ===
    WARMUP_BATCHES: int = 50        
    M: int = 5
    ENABLE_STRUCTURED_PROMPT_BANK: bool = False
    PROMPT_BANK_PROFILE: str = "v3"
    PROMPT_POOLING_MODE: str = "mean"  # mean | bucketed
    PROMPT_LEGACY_MIX: float = 0.0
    PROMPT_CLASS_MIX_PROFILE: str = "none"
    ENABLE_PROMPT_CORESET: bool = False
    PROMPT_CORESET_SIZE: int = 5
    PROMPT_BUCKET_KEEP: int = 2
    PROMPT_RESOURCE_MAX_CANDIDATES: int = 24
    PROMPT_SCORE_TEMP: float = 0.12
    PROMPT_BUCKET_SCORE_TEMP: float = 0.18
    PROMPT_BUCKET_PRIORS: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PROMPT_BUCKET_PRIORS))
    ENABLE_EARLY_TEXT_PROMPT_SUPPORT: bool = True
    EARLY_TEXT_PROMPT_TOP_K: int = 12
    EARLY_TEXT_PROMPT_SELECTION_MODE: str = "margin"  # margin | own | mean_margin | soft_margin
    EARLY_TEXT_PROMPT_ENTRY_MODE: str = "full"  # full | proto_only | mean_only
    EARLY_TEXT_PROMPT_SOURCE: str = "report"
    EARLY_TEXT_PROMPT_SELECTOR_SUBDIR: str = "early_text_prompt_selector"
    SITE_EXPERT_UNKNOWN_LOW_CONF: float = 0.60
    SITE_EXPERT_UNKNOWN_CONF_MARGIN: float = 0.0
    USE_ZCA_WHITEN: bool = False


    
    # === GO (Guardian + Multi-label residual projection) ===
    ENABLE_GO_GUARDIAN: bool = False
    GO_GUARDIAN_SCALAR: str = "top1_conf"
    GO_PSI_WINDOW: int = 512
    GO_PSI_BINS: int = 10
    GO_PSI_THR: float = 2.0
    GO_TAU_RESUME: float = 1.0
    GO_RESUME_WINDOWS: int = 3
    GO_WARMUP_STEPS: int = 50
    GO_BASELINE_COLLECT_STEPS: int = 50
    GO_DRY_RUN: bool = False
    GO_PSI_BASELINE_MAX: int = 5000
    GO_PSI_EVAL_EVERY: int = 1
    # Stage-2 (CUSUM/Page-Hinkley) kept optional; not enabled by default.
    ENABLE_GO_GUARDIAN_STAGE2: bool = False
    GO_STAGE2_DELTA: float = 0.01
    GO_STAGE2_LAMBDA: float = 0.08
    GO_STAGE2_MIN_STEPS: int = 3

    ENABLE_GO_MULTILABEL_PROJECTION: bool = True
    GO_ML_TAU_BASE: float = 1e-2
    GO_ML_COND_TARGET: float = 1e3
    GO_ML_USE_RESIDUAL_NORM_WEIGHT: bool = True
    GO_ML_SIGNAL_MODE: str = "residual"  # original | residual | adaptive
    GO_ML_SIGNAL_USE_ORIGINAL: bool = False
    GO_ML_ADAPTIVE_MIN_RESID_RATIO: float = 0.15
    GO_ML_CONFOUNDER_MODE: str = "full"  # full | topm | sim_weighted
    GO_ML_TOPM: int = 1
    GO_ML_SIM_WEIGHT_TEMP: float = 0.20
    GO_ML_ROBUST_MODE: str = "none"  # none | huber
    GO_ML_HUBER_DELTA: float = 0.20
    GO_ML_HUBER_SCOPE: str = "always"  # always | conditional | warmup
    GO_ML_HUBER_COND_MIN_ACTIVE: int = 3
    GO_ML_HUBER_COND_MIN_COND: float = 25.0
    GO_ML_HUBER_COND_MIN_RESID_RATIO: float = 0.90
    GO_ML_HUBER_COND_MIN_OTHER_SIM: float = 0.35
    GO_ML_HUBER_WARMUP_STEPS: int = 50

    # === Procrustes & Gating ===
    KAPPA0: float = 0.0
    # [鍏抽敭鍙傛暟] 姣忎釜绫诲埆蹇呴』杈惧埌鐨勬渶灏忔牱鏈暟
    N_MIN_SUPPORT_FOR_ACTIVE: int = 8
    
    N_CAP: int = 500            
    GAMMA_WEIGHT: float = 0.5
    ALPHA_WEIGHT: float = 1.0
    BETA_WEIGHT: float = 0.2    
    EPSILON: float = 0.0005      
    RHO: float = 0.85           
    GATE_USE_RHO_QUANTILE: bool = False
    RHO_QUANTILE: float = 0.70
    GATE_REQUIRE_OFFDIAG_IMPROVEMENT: bool = False
    GATE_MAX_OFFDIAG_DELTA: float = 0.0
    CAPAV1_DYNAMIC_OFFDIAG_FRAC: float = 0.30
    CAPAV1_RELAXED_GAIN_OFFDIAG_RATIO: float = 2.5
    CAPAV1_RHO_BYPASS_ACTIVE_MARGIN: int = 2
    PROCRUSTES_WEIGHT_CAP_MULT: float = 2.0
    ENABLE_HARD_NEG_PROCRUSTES: bool = False
    HARD_NEG_BETA: float = 0.15
    HARD_NEG_TOPK: int = 3
    HARD_NEG_TEMP: float = 0.07
    ENABLE_DISC_AXIS_PROCRUSTES: bool = True
    DISC_AXIS_NEG_LAMBDA: float = 0.25
    DISABLE_SHARED_CENTERING: bool = False
    ENABLE_RESIDUAL_LOCAL_SLERP: bool = True
    RESIDUAL_LOCAL_TAU_DELTA_DEG: float = 25.0
    RESIDUAL_LOCAL_N_MIN: int = 100
    RESIDUAL_LOCAL_ETA: float = 0.20
    RESIDUAL_LOCAL_LAMBDA_MAX: float = 0.08
    RESIDUAL_LOCAL_ANGLE_FLOOR_DEG: float = 23.0
    RESIDUAL_LOCAL_MIN_GAIN: float = 0.0
    MIN_CLASSES_FOR_ADAPTATION: int = -1
    
    INIT_TEMPERATURE: float = 0.590625
    INIT_SCALE_FACTOR: float = 5.0
    TAU_PRIOR: float = 0.05
    SCORING_MODE: str = "mixed"  # "mixed" (default) or "softmax"
    SIM_SOURCE: str = "gate"  # "gate" or "dataset"
    EVAL_MODE: str = "full_capa"
    TRAIN_BATCH_SIZE: int = 128
    AUDIT_DISABLE_EARLY_FREEZE: bool = False
    # Cache is an evaluation-only gated expert, disabled by default.
    CACHE_MODE: str = "off"  # off | gated
    CACHE_ALPHA_MAX: float = 0.10
    CACHE_TOPK: int = 16
    CACHE_TEMP: float = 0.08
    CACHE_DATASET_PSI_THR: float = 0.25
    CACHE_MIN_SIM_Q: float = 0.25
    CACHE_MIN_PURITY_Q: float = 0.50
    CACHE_MAX_ENTROPY_Q: float = 0.75
    CACHE_REQUIRE_AGREE: bool = True
    CACHE_CHUNK: int = 512
    CACHE_SELF_MATCH_COS: float = 0.999999
    ENABLE_CAPA_BASELINE_SOFT_FUSION: bool = True
    CAPA_BASELINE_FUSION_LAMBDA: float = 1.0
    CAPAV1_DUALTRACK_CONF_MARGIN: float = 0.02
    CAPAV1_DUALTRACK_BLEND: float = 0.65
    CAPAV1_DUALTRACK_ENABLE_ABSTAIN: bool = False
    CAPAV1_DUALTRACK_ABSTAIN_CONF: float = 0.60
    CAPAV1_GUARDED_ALPHAS: List[float] = field(default_factory=lambda: [1.0, 0.85, 0.70, 0.55, 0.40, 0.25])
    ENABLE_CAPAV1_GUARDED_SLERP: bool = False
    CAPAV1_GUARDED_SLERP_LAMBDA_MAX: float = 0.10
    CAPAV1_SOFT_FALLBACK_MIN_GAIN: float = 0.03
    CAPAV1_SOFT_FALLBACK_OFFDIAG_MULT: float = 1.50
    CAPAV1_GUARDED_DUMP: bool = False
    CAPAV1_SMALLER_ALPHA_SCORE_TOL: float = 0.015
    CAPAV1_SMALLER_ALPHA_MIN_GAIN_FRAC: float = 0.40
    ENABLE_PROMPT_BANK_READOUT: bool = True
    PROMPT_BANK_READOUT_TEMP: float = 0.07
    PROMPT_BANK_READOUT_CALIB_T: float = 2.0
    PROMPT_BANK_READOUT_TEMPLATES: List[str] = field(default_factory=lambda: [
        "This is an image of a chest X-ray depicting {finding}.",
        "A chest X-ray showing {finding}.",
        "There is evidence of {finding} on the chest X-ray.",
        "Findings are consistent with {finding}.",
        "The radiograph demonstrates {finding}.",
    ])
    ENABLE_CAPA_SHIFT_GATE: bool = True
    CAPA_SHIFT_GATE_THRESHOLD: float = 0.22
    # Professor profile knobs / targets.
    PROF_TARGET_TOP1: float = 0.75
    PROF_EPSILON_GAP_PCT: float = 0.015
    PROF_LAMBDA_MAX: float = 0.1
    PROF_KAPPA0: float = 40.0
    PROF_KAPPA_EMA: float = 50.0
    PROF_PSI_THR: float = 0.25
    PROF_AUTO_SCALE: bool = False
    PROF_AUTO_TEMPERATURE: bool = False
    PROF_AUTO_THRESHOLDS: bool = False
    PROF_DYNAMIC_EPSILON: bool = False
    PROF_DYNAMIC_TAU: bool = False

    MEDICAL_SYNONYM_MAP: Dict[str, List[str]] = field(default_factory=dict)

    TEMPLATES_PI: List[str] = field(default_factory=lambda: [
        "This chest X-ray shows {finding}.",
        "There is evidence of {finding}.",
        "Findings consistent with {finding}.",
        "The image demonstrates {finding}.",
        "Signs of {finding} are present."
    ])
    BINARY_POSITIVE_CLASS_MAP: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self):
        profile = str(self.PARAM_PROFILE).strip().lower()
        if profile in ("default", "base", "paper"):
            self.PARAM_PROFILE = "default"
        elif profile in ("professor", "prof"):
            self.PARAM_PROFILE = "professor"
        else:
            raise ValueError(f"Unsupported PARAM_PROFILE={self.PARAM_PROFILE}. Use 'default' or 'professor'.")

        label_space = str(self.LABEL_SPACE).strip().lower()
        if label_space in ("14", "full14", "14label", "14-label"):
            self.LABEL_SPACE = "14"
            self.ORDERED_CLASS_NAMES = list(FULL_14_CLASS_NAMES)
            self.MEDICAL_SYNONYM_MAP = copy.deepcopy(MEDICAL_SYNONYM_MAP_14)
            self.BINARY_POSITIVE_CLASS_MAP = copy.deepcopy(BINARY_POSITIVE_CLASS_MAP_14)
            if int(self.MIN_CLASSES_FOR_ADAPTATION) <= 0:
                self.MIN_CLASSES_FOR_ADAPTATION = 6
        elif label_space in ("chexpert5", "chexpert-5", "chex5", "c5"):
            self.LABEL_SPACE = "chexpert5"
            self.ORDERED_CLASS_NAMES = list(CHEXPERT_5_CLASS_NAMES)
            self.MEDICAL_SYNONYM_MAP = copy.deepcopy(MEDICAL_SYNONYM_MAP_CHEXPERT5)
            self.BINARY_POSITIVE_CLASS_MAP = copy.deepcopy(BINARY_POSITIVE_CLASS_MAP_CHEXPERT5)
            if int(self.MIN_CLASSES_FOR_ADAPTATION) <= 0:
                self.MIN_CLASSES_FOR_ADAPTATION = 3
            default_source_path = os.path.join(self.DATA_ROOT, DEFAULT_TRAIN_SOURCE_FILENAME)
            default_train_path = os.path.join(self.DATA_ROOT, DEFAULT_TARGET_AWARE_TRAIN_FILENAME)
            default_calib_path = os.path.join(self.DATA_ROOT, DEFAULT_TARGET_AWARE_CALIB_FILENAME)
            legacy_default_calib_path = os.path.join(self.DATA_ROOT, DEFAULT_CALIB_SUBSET_FILENAME)

            def _same_path(a: str, b: str) -> bool:
                if not a or not b:
                    return False
                return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))

            source_path = str(getattr(self, "TRAIN_SOURCE_DATA_PATH", "") or "").strip()
            if (not source_path) or os.path.isdir(source_path):
                self.TRAIN_SOURCE_DATA_PATH = str(default_source_path)
            train_path = str(getattr(self, "TRAIN_DATA_PATH", "") or "").strip()
            if (
                (not train_path)
                or os.path.isdir(train_path)
                or _same_path(train_path, self.TRAIN_SOURCE_DATA_PATH)
                or _same_path(train_path, default_source_path)
            ):
                self.TRAIN_DATA_PATH = str(default_train_path)
            cross_modal_path = str(getattr(self, "CROSS_MODAL_DATA_PATH", "") or "").strip()
            if (not cross_modal_path) or os.path.isdir(cross_modal_path):
                self.CROSS_MODAL_DATA_PATH = rf"{self.DATA_ROOT}\CHEXPERT_MIMIC.pkl"
            calib_path = str(getattr(self, "CALIB_DATA_PATH", "") or "").strip()
            if (
                (not calib_path)
                or os.path.isdir(calib_path)
                or _same_path(calib_path, self.CROSS_MODAL_DATA_PATH)
                or _same_path(calib_path, self.TRAIN_DATA_PATH)
                or _same_path(calib_path, self.TRAIN_SOURCE_DATA_PATH)
                or _same_path(calib_path, legacy_default_calib_path)
            ):
                self.CALIB_DATA_PATH = str(default_calib_path)
            tau_calib_path = str(getattr(self, "TAU_CALIB_DATA_PATH", "") or "").strip()
            if (
                (not tau_calib_path)
                or os.path.isdir(tau_calib_path)
                or _same_path(tau_calib_path, self.CROSS_MODAL_DATA_PATH)
                or _same_path(tau_calib_path, self.TRAIN_DATA_PATH)
                or _same_path(tau_calib_path, self.TRAIN_SOURCE_DATA_PATH)
                or _same_path(tau_calib_path, legacy_default_calib_path)
            ):
                self.TAU_CALIB_DATA_PATH = str(default_calib_path)
            chexpert_test_path = str(self.TEST_DATA_PATHS.get("CheXpert", "") or "").strip()
            mimic_test_path = str(self.TEST_DATA_PATHS.get("MIMIC", "") or "").strip()
            covid_test_path = str(self.TEST_DATA_PATHS.get("COVID", "") or "").strip()
            rsna_test_path = str(self.TEST_DATA_PATHS.get("RSNA", "") or "").strip()
            if (not chexpert_test_path) or os.path.isdir(chexpert_test_path):
                self.TEST_DATA_PATHS["CheXpert"] = rf"{self.DATA_ROOT}\cheXpert_200x5.pkl"
            if (not mimic_test_path) or os.path.isdir(mimic_test_path):
                self.TEST_DATA_PATHS["MIMIC"] = rf"{self.DATA_ROOT}\MIMIC_200x5.pkl"
            if (not covid_test_path) or os.path.isdir(covid_test_path):
                self.TEST_DATA_PATHS["COVID"] = rf"{self.DATA_ROOT}\COVID_3616x2.pkl"
            if (not rsna_test_path) or os.path.isdir(rsna_test_path):
                self.TEST_DATA_PATHS["RSNA"] = rf"{self.DATA_ROOT}\RSNA_4243x2.pkl"
        elif label_space in ("5", "u5", "unified5", "unified-5", "5label", "5-label"):
            self.LABEL_SPACE = "unified5"
            self.ORDERED_CLASS_NAMES = list(UNIFIED_5_CLASS_NAMES)
            self.MEDICAL_SYNONYM_MAP = copy.deepcopy(MEDICAL_SYNONYM_MAP_5)
            self.BINARY_POSITIVE_CLASS_MAP = copy.deepcopy(BINARY_POSITIVE_CLASS_MAP_5)
            if int(self.MIN_CLASSES_FOR_ADAPTATION) <= 0:
                self.MIN_CLASSES_FOR_ADAPTATION = 3
        else:
            raise ValueError(f"Unsupported LABEL_SPACE={self.LABEL_SPACE}. Use '14', 'chexpert5', or 'unified5'.")

        source_profile = str(self.SOURCE_LABEL_ORDER_PROFILE).strip().lower()
        if source_profile in ("default", "none", ""):
            self.SOURCE_LABEL_ORDER_PROFILE = "default"
        elif source_profile in ("chexpert5_reordered_200x5", "chex5_reordered_200x5", "chex5-reordered-200x5"):
            self.SOURCE_LABEL_ORDER_PROFILE = "chexpert5_reordered_200x5"
        else:
            raise ValueError(
                f"Unsupported SOURCE_LABEL_ORDER_PROFILE={self.SOURCE_LABEL_ORDER_PROFILE}. "
                "Use 'default' or 'chexpert5_reordered_200x5'."
            )

        eval_mode = str(getattr(self, "EVAL_MODE", "full_capa")).strip().lower()
        eval_mode = EVAL_MODE_ALIASES.get(eval_mode, eval_mode)
        if eval_mode not in EVAL_MODE_SPECS:
            raise ValueError(f"Unsupported EVAL_MODE={self.EVAL_MODE}. Use one of {list(EVAL_MODE_SPECS)}.")
        self.EVAL_MODE = eval_mode

        cache_mode = str(getattr(self, "CACHE_MODE", "off")).strip().lower()
        if cache_mode not in ("off", "gated"):
            raise ValueError(f"Unsupported CACHE_MODE={self.CACHE_MODE}. Use 'off' or 'gated'.")
        self.CACHE_MODE = cache_mode
        prompt_pooling = str(getattr(self, "PROMPT_POOLING_MODE", "mean")).strip().lower()
        if prompt_pooling not in ("mean", "bucketed"):
            raise ValueError(f"Unsupported PROMPT_POOLING_MODE={self.PROMPT_POOLING_MODE}. Use 'mean' or 'bucketed'.")
        self.PROMPT_POOLING_MODE = prompt_pooling
        prompt_bank_profile = str(getattr(self, "PROMPT_BANK_PROFILE", "v2")).strip().lower()
        if prompt_bank_profile not in PROMPT_BANK_PROFILE_BUCKET_ORDER:
            raise ValueError(f"Unsupported PROMPT_BANK_PROFILE={self.PROMPT_BANK_PROFILE}. Use one of {list(PROMPT_BANK_PROFILE_BUCKET_ORDER.keys())}.")
        self.PROMPT_BANK_PROFILE = prompt_bank_profile
        self.PROMPT_LEGACY_MIX = min(1.0, max(0.0, float(getattr(self, "PROMPT_LEGACY_MIX", 0.0))))
        prompt_class_mix_profile = str(getattr(self, "PROMPT_CLASS_MIX_PROFILE", "none")).strip().lower()
        if prompt_class_mix_profile not in PROMPT_CLASS_MIX_PROFILE_MAP:
            raise ValueError(f"Unsupported PROMPT_CLASS_MIX_PROFILE={self.PROMPT_CLASS_MIX_PROFILE}. Use one of {list(PROMPT_CLASS_MIX_PROFILE_MAP.keys())}.")
        self.PROMPT_CLASS_MIX_PROFILE = prompt_class_mix_profile
        self.PROMPT_BUCKET_KEEP = max(1, int(getattr(self, "PROMPT_BUCKET_KEEP", 2)))
        self.PROMPT_CORESET_SIZE = max(1, int(getattr(self, "PROMPT_CORESET_SIZE", self.M)))
        self.PROMPT_RESOURCE_MAX_CANDIDATES = max(
            1,
            int(getattr(self, "PROMPT_RESOURCE_MAX_CANDIDATES", 24)),
        )
        self.EARLY_TEXT_PROMPT_TOP_K = max(1, int(getattr(self, "EARLY_TEXT_PROMPT_TOP_K", 12)))
        early_prompt_mode = str(getattr(self, "EARLY_TEXT_PROMPT_ENTRY_MODE", "full")).strip().lower()
        if early_prompt_mode not in ("full", "proto_only", "mean_only"):
            raise ValueError("Unsupported EARLY_TEXT_PROMPT_ENTRY_MODE. Use 'full', 'proto_only', or 'mean_only'.")
        self.EARLY_TEXT_PROMPT_ENTRY_MODE = early_prompt_mode
        early_select_mode = str(getattr(self, "EARLY_TEXT_PROMPT_SELECTION_MODE", "margin")).strip().lower()
        if early_select_mode not in ("margin", "own", "mean_margin", "soft_margin"):
            raise ValueError("Unsupported EARLY_TEXT_PROMPT_SELECTION_MODE. Use 'margin', 'own', 'mean_margin', or 'soft_margin'.")
        self.EARLY_TEXT_PROMPT_SELECTION_MODE = early_select_mode
        early_source = str(getattr(self, "EARLY_TEXT_PROMPT_SOURCE", "report")).strip().lower()
        if early_source not in ("report",):
            raise ValueError("Unsupported EARLY_TEXT_PROMPT_SOURCE. Use 'report'.")
        self.EARLY_TEXT_PROMPT_SOURCE = early_source
        self.PROMPT_SCORE_TEMP = max(1e-3, float(getattr(self, "PROMPT_SCORE_TEMP", 0.12)))
        self.PROMPT_BUCKET_SCORE_TEMP = max(1e-3, float(getattr(self, "PROMPT_BUCKET_SCORE_TEMP", 0.18)))
        priors = dict(PROMPT_BANK_PROFILE_PRIORS.get(self.PROMPT_BANK_PROFILE, DEFAULT_PROMPT_BUCKET_PRIORS))
        priors.update(dict(getattr(self, "PROMPT_BUCKET_PRIORS", {})))
        self.PROMPT_BUCKET_PRIORS = {str(k): float(v) for k, v in priors.items()}
        self.DISC_AXIS_NEG_LAMBDA = max(0.0, float(getattr(self, "DISC_AXIS_NEG_LAMBDA", 0.25)))
        self.RESIDUAL_LOCAL_TAU_DELTA_DEG = max(0.0, float(getattr(self, "RESIDUAL_LOCAL_TAU_DELTA_DEG", 25.0)))
        self.RESIDUAL_LOCAL_N_MIN = max(1, int(getattr(self, "RESIDUAL_LOCAL_N_MIN", 100)))
        self.RESIDUAL_LOCAL_ETA = max(0.0, float(getattr(self, "RESIDUAL_LOCAL_ETA", 0.20)))
        self.RESIDUAL_LOCAL_LAMBDA_MAX = min(1.0, max(0.0, float(getattr(self, "RESIDUAL_LOCAL_LAMBDA_MAX", 0.08))))
        self.RESIDUAL_LOCAL_ANGLE_FLOOR_DEG = min(179.0, max(0.0, float(getattr(self, "RESIDUAL_LOCAL_ANGLE_FLOOR_DEG", 23.0))))
        self.RESIDUAL_LOCAL_MIN_GAIN = float(getattr(self, "RESIDUAL_LOCAL_MIN_GAIN", 0.0))
        self.PROMPT_BANK_READOUT_TEMP = max(1e-4, float(getattr(self, "PROMPT_BANK_READOUT_TEMP", 0.07)))
        self.PROMPT_BANK_READOUT_CALIB_T = max(1e-4, float(getattr(self, "PROMPT_BANK_READOUT_CALIB_T", 2.0)))
        self.PROMPT_BANK_READOUT_TEMPLATES = [str(x) for x in list(getattr(self, "PROMPT_BANK_READOUT_TEMPLATES", []))]
        if not self.PROMPT_BANK_READOUT_TEMPLATES:
            self.PROMPT_BANK_READOUT_TEMPLATES = ["This is an image of a chest X-ray depicting {finding}."]
        self.CAPA_SHIFT_GATE_THRESHOLD = max(0.0, float(getattr(self, "CAPA_SHIFT_GATE_THRESHOLD", 0.22)))

        go_ml_signal_mode = str(getattr(self, "GO_ML_SIGNAL_MODE", "")).strip().lower()
        if not go_ml_signal_mode:
            go_ml_signal_mode = "original" if bool(getattr(self, "GO_ML_SIGNAL_USE_ORIGINAL", False)) else "residual"
        if go_ml_signal_mode not in ("original", "residual", "adaptive"):
            raise ValueError("Unsupported GO_ML_SIGNAL_MODE. Use 'original', 'residual', or 'adaptive'.")
        self.GO_ML_SIGNAL_MODE = go_ml_signal_mode
        self.GO_ML_SIGNAL_USE_ORIGINAL = bool(go_ml_signal_mode == "original")
        conf_mode = str(getattr(self, "GO_ML_CONFOUNDER_MODE", "full")).strip().lower()
        if conf_mode not in ("full", "topm", "sim_weighted"):
            raise ValueError("Unsupported GO_ML_CONFOUNDER_MODE. Use 'full', 'topm', or 'sim_weighted'.")
        self.GO_ML_CONFOUNDER_MODE = conf_mode
        self.GO_ML_TOPM = max(1, int(getattr(self, "GO_ML_TOPM", 1)))
        self.GO_ML_SIM_WEIGHT_TEMP = max(1e-3, float(getattr(self, "GO_ML_SIM_WEIGHT_TEMP", 0.20)))
        robust_mode = str(getattr(self, "GO_ML_ROBUST_MODE", "none")).strip().lower()
        if robust_mode not in ("none", "huber"):
            raise ValueError("Unsupported GO_ML_ROBUST_MODE. Use 'none' or 'huber'.")
        self.GO_ML_ROBUST_MODE = robust_mode
        self.GO_ML_HUBER_DELTA = max(1e-4, float(getattr(self, "GO_ML_HUBER_DELTA", 0.20)))
        huber_scope = str(getattr(self, "GO_ML_HUBER_SCOPE", "always")).strip().lower()
        if huber_scope not in ("always", "conditional", "warmup"):
            raise ValueError("Unsupported GO_ML_HUBER_SCOPE. Use 'always', 'conditional', or 'warmup'.")
        self.GO_ML_HUBER_SCOPE = huber_scope
        self.GO_ML_HUBER_COND_MIN_ACTIVE = max(2, int(getattr(self, "GO_ML_HUBER_COND_MIN_ACTIVE", 3)))
        self.GO_ML_HUBER_COND_MIN_COND = max(1.0, float(getattr(self, "GO_ML_HUBER_COND_MIN_COND", 25.0)))
        self.GO_ML_HUBER_COND_MIN_RESID_RATIO = min(1.0, max(0.0, float(getattr(self, "GO_ML_HUBER_COND_MIN_RESID_RATIO", 0.90))))
        self.GO_ML_HUBER_COND_MIN_OTHER_SIM = min(1.0, max(0.0, float(getattr(self, "GO_ML_HUBER_COND_MIN_OTHER_SIM", 0.35))))
        self.GO_ML_HUBER_WARMUP_STEPS = max(0, int(getattr(self, "GO_ML_HUBER_WARMUP_STEPS", int(self.WARMUP_BATCHES))))
        self.GO_ML_ADAPTIVE_MIN_RESID_RATIO = min(1.0, max(0.0, float(getattr(self, "GO_ML_ADAPTIVE_MIN_RESID_RATIO", 0.15))))

        if bool(self.DEBUG):
            self.CAPAV1_GUARDED_DUMP = True

        save_dir = str(self.SAVE_DIR)
        if os.path.basename(os.path.normpath(save_dir)).lower() != "capav1_gt":
            self.SAVE_DIR = os.path.join(save_dir, "capav1_gt")

        if self.PARAM_PROFILE == "professor":
            self.KAPPA0 = float(np.clip(self.PROF_KAPPA0, 20.0, 60.0))
            self.KAPPA_EMA = float(np.clip(self.PROF_KAPPA_EMA, 30.0, 80.0))
            self.GATE_USE_RHO_QUANTILE = True
            self.RHO_QUANTILE = 0.80

class CAPA5NotebookRunner:
    def __init__(self, config: CAPA5Config):
        self.config = config
        self.device = torch.device(self.config.DEVICE)
        self.config.VERBOSE = bool(self.config.VERBOSE) or bool(getattr(self.config, "DEBUG", False))
        self._ensure_calibration_files_ready()
        self.eval_runtime = self._build_eval_runtime()
        self._log(f"[CAPA] Init on {self.device} (verbose={self.config.VERBOSE})")
        self._init_seed()
        self._init_clip_model()
        self._init_state()
        self._log_eval_mode_summary(always=True)
        if not os.path.exists(self.config.SAVE_DIR):
            os.makedirs(self.config.SAVE_DIR)

    def _log(self, msg: str, *, always: bool = False):
        if always or self.config.VERBOSE:
            print(msg)

    @staticmethod
    def _norm_path(path: str) -> str:
        return os.path.normcase(os.path.abspath(str(path)))

    def _default_data_path(self, filename: str) -> str:
        return os.path.join(str(self.config.DATA_ROOT), filename)

    def _is_default_data_path(self, path: str, filename: str) -> bool:
        if not path:
            return False
        return self._norm_path(self._resolve_legacy_data_path(path)) == self._norm_path(
            self._default_data_path(filename)
        )

    def _ensure_calibration_files_ready(self) -> None:
        self._ensure_target_aware_data_files_ready()
        targets = []
        for path in [getattr(self.config, "CALIB_DATA_PATH", ""), getattr(self.config, "TAU_CALIB_DATA_PATH", "")]:
            path_str = str(path or "").strip()
            if not path_str:
                continue
            resolved = self._resolve_legacy_data_path(path_str)
            if resolved not in targets:
                targets.append(resolved)
        for target in targets:
            if os.path.exists(target):
                continue
            self._materialize_calibration_subset_from_train(target)

    def _ensure_target_aware_data_files_ready(self) -> None:
        if str(getattr(self.config, "LABEL_SPACE", "")).strip().lower() != "chexpert5":
            return

        train_path = str(getattr(self.config, "TRAIN_DATA_PATH", "") or "").strip()
        calib_path = str(getattr(self.config, "CALIB_DATA_PATH", "") or "").strip()
        tau_calib_path = str(getattr(self.config, "TAU_CALIB_DATA_PATH", "") or "").strip()

        wants_default_train = self._is_default_data_path(train_path, DEFAULT_TARGET_AWARE_TRAIN_FILENAME)
        wants_default_calib = self._is_default_data_path(calib_path, DEFAULT_TARGET_AWARE_CALIB_FILENAME)
        wants_default_tau_calib = self._is_default_data_path(
            tau_calib_path,
            DEFAULT_TARGET_AWARE_CALIB_FILENAME,
        )
        if not (wants_default_train or wants_default_calib or wants_default_tau_calib):
            return

        target_train_path = (
            self._resolve_legacy_data_path(train_path)
            if wants_default_train
            else self._default_data_path(DEFAULT_TARGET_AWARE_TRAIN_FILENAME)
        )
        target_calib_path = (
            self._resolve_legacy_data_path(calib_path)
            if wants_default_calib
            else self._default_data_path(DEFAULT_TARGET_AWARE_CALIB_FILENAME)
        )
        missing_train = wants_default_train and (not os.path.exists(target_train_path))
        missing_calib = (wants_default_calib or wants_default_tau_calib) and (not os.path.exists(target_calib_path))
        summary_path = self._default_data_path(DEFAULT_TARGET_AWARE_SUMMARY_FILENAME)
        if not (missing_train or missing_calib or (not os.path.exists(summary_path))):
            return

        self._materialize_chexpert5_target_aware_data(
            train_target_path=target_train_path,
            calib_target_path=target_calib_path,
        )

    @staticmethod
    def _stack_label_values(values) -> np.ndarray:
        rows = []
        for value in values:
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            rows.append(np.asarray(value))
        return np.stack(rows).astype(np.float32)

    @staticmethod
    def _value_counts_as_json_dict(series: pd.Series) -> Dict[str, int]:
        return {str(k): int(v) for k, v in series.value_counts(dropna=False).to_dict().items()}

    @staticmethod
    def _int_distribution(values: np.ndarray) -> Dict[str, int]:
        values = np.asarray(values).astype(int)
        uniq, counts = np.unique(values, return_counts=True)
        return {str(int(k)): int(v) for k, v in zip(uniq, counts)}

    def _load_dataframe_pickle(self, path: str, *, role: str) -> pd.DataFrame:
        try:
            with open(path, "rb") as f:
                blob = pickle.load(f)
        except Exception:
            blob = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(blob, pd.DataFrame):
            raise RuntimeError(
                f"Automatic {role} materialization expects a DataFrame pickle; "
                f"got {type(blob).__name__} from {path}."
            )
        return blob

    def _materialize_chexpert5_target_aware_data(
        self,
        *,
        train_target_path: str,
        calib_target_path: str,
    ) -> None:
        source_path = str(getattr(self.config, "TRAIN_SOURCE_DATA_PATH", "") or "").strip()
        if not source_path:
            source_path = self._default_data_path(DEFAULT_TRAIN_SOURCE_FILENAME)
        source_path = self._resolve_legacy_data_path(source_path)
        if not os.path.exists(source_path):
            raise FileNotFoundError(
                f"Cannot build CheXpert-5 target-aware data because TRAIN_SOURCE_DATA_PATH is missing: {source_path}"
            )

        df_source = self._load_dataframe_pickle(source_path, role="CheXpert-5 target-aware data")
        if "Embedding" not in df_source.columns:
            raise RuntimeError(f"Cannot build target-aware train from {source_path}: missing 'Embedding' column.")
        if "labels" not in df_source.columns:
            raise RuntimeError(f"Cannot build target-aware train from {source_path}: missing 'labels' column.")

        y_source = self._stack_label_values(df_source["labels"].values)
        if y_source.ndim != 2 or y_source.shape[1] < len(FULL_14_CLASS_NAMES):
            raise RuntimeError(
                f"Expected full 14-label vectors in {source_path}; got shape={tuple(y_source.shape)}."
            )

        target_cols = [FULL_14_CLASS_NAMES.index(name) for name in CHEXPERT_5_CLASS_NAMES]
        y_target = y_source[:, target_cols]
        target_positive_counts = y_target.sum(axis=1)
        train_mask = target_positive_counts > 0
        df_train = df_source.loc[train_mask].copy().reset_index(drop=True)
        if len(df_train) <= 0:
            raise RuntimeError(f"No rows in {source_path} contain any CheXpert-5 target label.")

        if "modality" in df_train.columns:
            image_mask = df_train["modality"].astype(str).str.lower().eq("image")
            df_calib_pool = df_train.loc[image_mask].copy().reset_index(drop=True)
        else:
            df_calib_pool = df_train.copy()
        if len(df_calib_pool) <= 0:
            raise RuntimeError(f"No image rows in {source_path} contain any CheXpert-5 target label.")

        n_keep = min(int(DEFAULT_CALIB_SUBSET_SIZE), int(len(df_calib_pool)))
        df_calib = df_calib_pool.sample(n=n_keep, random_state=int(self.config.RANDOM_SEED)).reset_index(drop=True)

        train_target_path = self._resolve_legacy_data_path(str(train_target_path))
        calib_target_path = self._resolve_legacy_data_path(str(calib_target_path))
        os.makedirs(os.path.dirname(os.path.abspath(train_target_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(calib_target_path)), exist_ok=True)

        wrote_parts = []
        if not os.path.exists(train_target_path):
            with open(train_target_path, "wb") as f:
                pickle.dump(df_train, f, protocol=pickle.HIGHEST_PROTOCOL)
            wrote_parts.append("train")
        if not os.path.exists(calib_target_path):
            with open(calib_target_path, "wb") as f:
                pickle.dump(df_calib, f, protocol=pickle.HIGHEST_PROTOCOL)
            wrote_parts.append("calib")

        y_train = self._stack_label_values(df_train["labels"].values)
        y_calib = self._stack_label_values(df_calib["labels"].values)
        summary = {
            "source_train_path": str(source_path),
            "target_train_path": str(train_target_path),
            "target_calib_path": str(calib_target_path),
            "label_space": "chexpert5",
            "target_class_names": list(CHEXPERT_5_CLASS_NAMES),
            "target_cols_in_full14": [int(x) for x in target_cols],
            "random_seed": int(self.config.RANDOM_SEED),
            "calib_subset_size_requested": int(DEFAULT_CALIB_SUBSET_SIZE),
            "source_rows": int(len(df_source)),
            "target_positive_train_rows": int(len(df_train)),
            "target_positive_calib_rows": int(len(df_calib)),
            "source_modality_counts": (
                self._value_counts_as_json_dict(df_source["modality"])
                if "modality" in df_source.columns
                else {}
            ),
            "train_modality_counts": (
                self._value_counts_as_json_dict(df_train["modality"])
                if "modality" in df_train.columns
                else {}
            ),
            "calib_modality_counts": (
                self._value_counts_as_json_dict(df_calib["modality"])
                if "modality" in df_calib.columns
                else {}
            ),
            "target_positive_count_distribution_train": self._int_distribution(
                y_train[:, target_cols].sum(axis=1)
            ),
            "target_positive_count_distribution_calib": self._int_distribution(
                y_calib[:, target_cols].sum(axis=1)
            ),
            "target_class_counts_train": {
                name: int(y_train[:, col].sum()) for name, col in zip(CHEXPERT_5_CLASS_NAMES, target_cols)
            },
            "target_class_counts_calib": {
                name: int(y_calib[:, col].sum()) for name, col in zip(CHEXPERT_5_CLASS_NAMES, target_cols)
            },
        }
        summary_path = self._default_data_path(DEFAULT_TARGET_AWARE_SUMMARY_FILENAME)
        os.makedirs(os.path.dirname(os.path.abspath(summary_path)), exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=True)

        action = "Created" if wrote_parts else "Verified"
        self._log(
            f"[DataLayout] {action} CheXpert-5 target-aware train/calib "
            f"from {source_path} (train_rows={len(df_train)}, calib_rows={len(df_calib)}, "
            f"wrote={','.join(wrote_parts) if wrote_parts else 'none'}).",
            always=True,
        )

    def _materialize_calibration_subset_from_train(self, target_path: str) -> None:
        train_src = self._resolve_legacy_data_path(self.config.TRAIN_DATA_PATH)
        if not os.path.exists(train_src):
            raise FileNotFoundError(f"Cannot build calibration file because TRAIN_DATA_PATH is missing: {train_src}")
        try:
            with open(train_src, "rb") as f:
                train_blob = pickle.load(f)
        except Exception:
            train_blob = torch.load(train_src, map_location="cpu", weights_only=False)

        if not isinstance(train_blob, pd.DataFrame):
            raise RuntimeError(
                f"Automatic calibration subset generation expects TRAIN_DATA_PATH to be a DataFrame pickle; "
                f"got {type(train_blob).__name__} from {train_src}."
            )
        df_train = train_blob.copy()
        if "Embedding" not in df_train.columns:
            raise RuntimeError(f"Cannot build calibration subset from {train_src}: missing 'Embedding' column.")
        if "labels" not in df_train.columns:
            raise RuntimeError(f"Cannot build calibration subset from {train_src}: missing 'labels' column.")
        if "modality" in df_train.columns:
            image_mask = df_train["modality"].astype(str).str.lower().eq("image")
            df_train = df_train.loc[image_mask].copy()
        if len(df_train) <= 0:
            raise RuntimeError(f"Cannot build calibration subset from {train_src}: no image rows found.")

        n_keep = min(int(DEFAULT_CALIB_SUBSET_SIZE), int(len(df_train)))
        df_calib = df_train.sample(n=n_keep, random_state=int(self.config.RANDOM_SEED)).reset_index(drop=True)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "wb") as f:
            pickle.dump(df_calib, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._log(
            f"[DataLayout] Created calibration subset {target_path} from {train_src} "
            f"(rows={len(df_calib)}, modality=image, seed={self.config.RANDOM_SEED}).",
            always=True,
        )
    
    def _init_seed(self):
        random.seed(self.config.RANDOM_SEED)
        np.random.seed(self.config.RANDOM_SEED)
        torch.manual_seed(self.config.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _current_eval_mode(self) -> str:
        return str(self.eval_runtime.get("eval_mode", getattr(self.config, "EVAL_MODE", "full_capa")))

    def _build_eval_runtime(self) -> Dict[str, Any]:
        mode = str(getattr(self.config, "EVAL_MODE", "full_capa")).strip().lower()
        mode = EVAL_MODE_ALIASES.get(mode, mode)
        spec = EVAL_MODE_SPECS[mode]
        overrides: List[Dict[str, Any]] = []

        def record_override(name: str, requested: Any, effective: Any, reason: str) -> None:
            if requested == effective:
                return
            overrides.append(
                {
                    "name": name,
                    "requested": requested,
                    "effective": effective,
                    "reason": reason,
                }
            )

        guardian_enabled = bool(spec.guardian)
        guardian_requested = bool(getattr(self.config, "ENABLE_GO_GUARDIAN", False))
        if spec.deploy_overrides and spec.guardian and (not guardian_requested):
            record_override("ENABLE_GO_GUARDIAN", False, True, f"{mode} restores guardian")
        elif (not spec.guardian) and guardian_requested:
            record_override("ENABLE_GO_GUARDIAN", True, False, f"{mode} disables guardian")

        prompt_bank_readout_enabled = mode == "full_capa" and bool(getattr(self.config, "ENABLE_PROMPT_BANK_READOUT", True))
        dual_track_enabled = bool(spec.dual_track)
        if prompt_bank_readout_enabled:
            dual_track_enabled = False
        if (not spec.dual_track) and bool(getattr(self.config, "CAPAV1_DUALTRACK_ENABLE_ABSTAIN", False)):
            record_override("CAPAV1_DUALTRACK_ENABLE_ABSTAIN", True, False, f"{mode} disables dual-track routing")

        cache_requested = str(getattr(self.config, "CACHE_MODE", "off")).strip().lower() == "gated"
        cache_enabled = bool(spec.cache) and bool(cache_requested) and (not prompt_bank_readout_enabled)
        if (not spec.cache) and cache_requested:
            record_override("CACHE_MODE", "gated", "off", f"{mode} disables cache")

        calibration_enabled = bool(spec.calibration)
        runtime_scale_enabled = bool(spec.runtime_scale)

        prior_requested = float(getattr(self.config, "TAU_PRIOR", 0.0))
        if spec.prior_correction:
            prior_strength = prior_requested
            if spec.deploy_overrides and prior_strength <= 0.0:
                prior_strength = 0.05
                record_override("TAU_PRIOR", prior_requested, prior_strength, f"{mode} restores prior correction")
        else:
            prior_strength = 0.0
            if abs(prior_requested) > 1e-12:
                record_override("TAU_PRIOR", prior_requested, 0.0, f"{mode} disables prior correction")

        soft_fusion_requested = bool(getattr(self.config, "ENABLE_CAPA_BASELINE_SOFT_FUSION", False))
        soft_fusion_enabled = bool(spec.soft_fusion) and soft_fusion_requested and (not prompt_bank_readout_enabled)
        if (not spec.soft_fusion) and soft_fusion_requested:
            record_override(
                "ENABLE_CAPA_BASELINE_SOFT_FUSION",
                soft_fusion_requested,
                False,
                f"{mode} disables baseline fusion",
            )
        fusion_lambda = float(getattr(self.config, "CAPA_BASELINE_FUSION_LAMBDA", 1.0))
        if soft_fusion_enabled and spec.deploy_overrides and fusion_lambda >= 1.0:
            eff_lambda = float(getattr(self.config, "CAPAV1_DUALTRACK_BLEND", 1.0))
            record_override(
                "CAPA_BASELINE_FUSION_LAMBDA",
                fusion_lambda,
                eff_lambda,
                f"{mode} restores baseline fusion mixing",
            )
            fusion_lambda = eff_lambda
        if not soft_fusion_enabled:
            fusion_lambda = 1.0

        offdiag_gate_enabled = bool(spec.offdiag_gate)
        gate_requested = bool(getattr(self.config, "GATE_REQUIRE_OFFDIAG_IMPROVEMENT", False))
        if spec.deploy_overrides and spec.offdiag_gate and (not gate_requested):
            record_override(
                "GATE_REQUIRE_OFFDIAG_IMPROVEMENT",
                gate_requested,
                True,
                f"{mode} restores guarded off-diagonal gate",
            )
        elif (not spec.offdiag_gate) and gate_requested:
            record_override(
                "GATE_REQUIRE_OFFDIAG_IMPROVEMENT",
                gate_requested,
                False,
                f"{mode} disables guarded off-diagonal gate",
            )
        gate_max_offdiag_delta = float(getattr(self.config, "GATE_MAX_OFFDIAG_DELTA", 0.0))
        if offdiag_gate_enabled and spec.deploy_overrides and gate_max_offdiag_delta <= 0.0:
            record_override(
                "GATE_MAX_OFFDIAG_DELTA",
                gate_max_offdiag_delta,
                0.06,
                f"{mode} restores off-diagonal gate threshold",
            )
            gate_max_offdiag_delta = 0.06

        rho_effective = float(getattr(self.config, "RHO", 0.85))
        if spec.deploy_overrides and rho_effective < 0.88:
            record_override("RHO", rho_effective, 0.88, f"{mode} restores rho floor")
            rho_effective = 0.88

        final_logits_source = str(spec.final_logits_source)
        if mode == "full_capa" and bool(getattr(self.config, "ENABLE_DISC_AXIS_PROCRUSTES", True)):
            final_logits_source = "disc_axis"
        if mode == "full_capa" and bool(getattr(self.config, "ENABLE_RESIDUAL_LOCAL_SLERP", True)):
            final_logits_source = f"{final_logits_source}+residual_local_slerp"
        if mode == "full_capa" and bool(getattr(self.config, "ENABLE_PROMPT_BANK_READOUT", True)):
            final_logits_source = f"{final_logits_source}+logit_prompt_bank"
        if dual_track_enabled:
            final_logits_source = f"{final_logits_source}+dualtrack"
        if cache_enabled:
            final_logits_source = f"{final_logits_source}+cache"
        if mode == "full_capa" and bool(getattr(self.config, "ENABLE_CAPA_SHIFT_GATE", True)):
            final_logits_source = f"{final_logits_source}+shift_gate"

        return {
            "eval_mode": mode,
            "spec": spec,
            "prototype_source": spec.prototype_key,
            "final_logits_source": final_logits_source,
            "image_preprocessing": bool(spec.image_preprocessing),
            "alignment": bool(spec.alignment),
            "test_time_adaptation": bool(spec.test_time_adaptation),
            "guarded_alignment": bool(spec.guarded_alignment),
            "dual_track": dual_track_enabled,
            "cache": cache_enabled,
            "cache_mode": "gated" if cache_enabled else "off",
            "guardian": guardian_enabled,
            "prior_correction": bool(spec.prior_correction),
            "prior_strength": float(prior_strength),
            "calibration": calibration_enabled,
            "runtime_scale": runtime_scale_enabled,
            "soft_fusion": soft_fusion_enabled,
            "fusion_lambda": float(fusion_lambda),
            "offdiag_gate": offdiag_gate_enabled,
            "gate_max_offdiag_delta": float(gate_max_offdiag_delta),
            "rho": float(rho_effective),
            "metric_source": f"{Path(__file__).name}::_compute_metrics",
            "overrides": overrides,
            "notes": str(spec.notes),
        }

    def _log_eval_mode_summary(self, *, always: bool = False) -> None:
        runtime = dict(self.eval_runtime)
        summary = (
            f"[EvalMode] mode={runtime['eval_mode']} "
            f"| protos={runtime['prototype_source']} "
            f"| prep={int(runtime['image_preprocessing'])} "
            f"| align={int(runtime['alignment'])} "
            f"| dualtrack={int(runtime['dual_track'])} "
            f"| cache={int(runtime['cache'])} "
            f"| guardian={int(runtime['guardian'])} "
            f"| prior={int(runtime['prior_correction'])} "
            f"| calib={int(runtime['calibration'])} "
            f"| scale={int(runtime['runtime_scale'])}"
        )
        self._log(summary, always=always)
        for item in runtime.get("overrides", []):
            self._log(
                f"[EvalMode][Override] {item['name']}: requested={item['requested']} -> "
                f"effective={item['effective']} ({item['reason']})",
                always=always,
            )

    def _mode_uses_image_preprocessing(self) -> bool:
        return bool(self.eval_runtime.get("image_preprocessing", False))

    def _mode_uses_alignment(self) -> bool:
        return bool(self.eval_runtime.get("alignment", False))

    def _mode_uses_test_time_adaptation(self) -> bool:
        return bool(self.eval_runtime.get("test_time_adaptation", False))

    def _mode_uses_guarded_alignment(self) -> bool:
        return bool(self.eval_runtime.get("guarded_alignment", False))

    def _mode_uses_prior_correction(self) -> bool:
        return bool(self.eval_runtime.get("prior_correction", False))

    def _mode_uses_calibration(self) -> bool:
        return bool(self.eval_runtime.get("calibration", False))

    def _mode_uses_runtime_scale(self) -> bool:
        return bool(self.eval_runtime.get("runtime_scale", False))

    def _effective_prior_strength(self) -> float:
        return float(self.eval_runtime.get("prior_strength", 0.0))

    def _effective_fusion_lambda(self) -> float:
        return float(self.eval_runtime.get("fusion_lambda", 1.0))

    def _effective_rho(self) -> float:
        return float(self.eval_runtime.get("rho", getattr(self.config, "RHO", 0.85)))

    def _effective_gate_max_offdiag_delta(self) -> float:
        return float(self.eval_runtime.get("gate_max_offdiag_delta", getattr(self.config, "GATE_MAX_OFFDIAG_DELTA", 0.0)))

    def _effective_eval_scale(self) -> float:
        return float(self.s_opt) if self._mode_uses_runtime_scale() else 1.0

    def _uses_dual_track_inference(self) -> bool:
        return bool(self.eval_runtime.get("dual_track", False))
    
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
        self.t_raw_text = None
        self.t_processed_text = None
        self.t_aligned_text = None
        self.t_raw_pooled = None
        self.t_raw_pooled_raw = None  # store unwhitened text prototypes for raw-space geometry reporting
        self.t_zero_shot_base = None
        self.t_zero_shot_base_raw = None
        self.t_align_base = None
        self.t_paraphrases = []
        self.current_R = None; self.image_centroids = None
        self.disc_neg_centroids = None
        self.disc_neg_counts = None
        self.support_counts = None; self.rejected_counts = None; self.prior_counts = None; self.b_c = None
        self.is_frozen = False; self.R_frozen = None
        self.cache_keys = None
        self.cache_labels = None
        self.cache_is_multi = False
        self.cache_ready = False
        self.cache_reference = None
        self.cache_reference_ready = False
        self.last_cache_eval_info = {}
        self.last_dualtrack_eval_info = {}
        self.final_alignment_stats = {}
        self.residual_local_slerp_info: List[Dict[str, object]] = []
        self.last_shift_gate_info: Dict[str, object] = {}
        self.early_text_prompt_support_info: Dict[str, object] = {}
        self.early_text_prompt_groups: Dict[str, List[str]] = {}
        self.max_leverage_info = "N/A"
        self.R_last_good = None
        self.centroids_last_good = None

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
        self.guardian_stage2_ref_mean = np.nan
        self.guardian_stage2_stat = 0.0
        self.guardian_stage2_last = np.nan
        self.guardian_stage2_steps = 0

        # GO multi-label tau cache by active-label cardinality.
        self.go_ml_tau_cache: Dict[int, float] = {}

    def _l2_norm(self, x, dim=-1):
        return F.normalize(x, p=2, dim=dim, eps=1e-8)

    def _encode_text(self, texts: List[str]):
        with torch.no_grad():
            text_inputs = self.tokenizer(texts).to(self.device)
            return self._l2_norm(self.clip_model.encode_text(text_inputs))

    def _early_text_prompt_support_enabled(self) -> bool:
        return (
            self._current_eval_mode() == "full_capa"
            and str(getattr(self.config, "LABEL_SPACE", "")).strip().lower() == "chexpert5"
            and bool(getattr(self.config, "ENABLE_EARLY_TEXT_PROMPT_SUPPORT", True))
        )

    def _render_early_text_prompt_candidates(self) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = {}
        for cls_name in list(self.config.ORDERED_CLASS_NAMES):
            phrases = list(EARLY_TEXT_SUPPORT_SYNONYMS_CHEXPERT5.get(cls_name, [cls_name.lower()]))
            prompts: List[str] = []
            seen = set()
            for phrase in phrases:
                for template in EARLY_TEXT_SUPPORT_REPORT_TEMPLATES:
                    text = " ".join(str(template).replace("{finding}", str(phrase)).strip().split())
                    if text and text not in seen:
                        seen.add(text)
                        prompts.append(text)
            groups[cls_name] = prompts
        return groups

    def _select_early_text_prompt_groups(self, selector: "CAPA5NotebookRunner") -> Tuple[Dict[str, List[str]], Dict[str, object]]:
        if not isinstance(selector.image_centroids, torch.Tensor):
            raise RuntimeError("Early text prompt selector is missing image centroids.")
        if not isinstance(selector.R_frozen, torch.Tensor):
            raise RuntimeError("Early text prompt selector is missing R_frozen.")
        candidates = self._render_early_text_prompt_candidates()
        refs = selector._l2_norm(selector.image_centroids)
        top_k = max(1, int(getattr(self.config, "EARLY_TEXT_PROMPT_TOP_K", 12)))
        select_mode = str(getattr(self.config, "EARLY_TEXT_PROMPT_SELECTION_MODE", "margin")).strip().lower()
        selected: Dict[str, List[str]] = {}
        selected_rows: List[Dict[str, object]] = []
        for class_idx, cls_name in enumerate(list(self.config.ORDERED_CLASS_NAMES)):
            prompts = list(candidates.get(cls_name, []))
            if not prompts:
                prompts = [f"Impression: {cls_name.lower()}."]
            raw = selector._encode_text(prompts)
            proc = selector._apply_preprocessing(raw, selector.zT_mean)
            proc = selector._l2_norm(proc)
            proc = selector._l2_norm(torch.matmul(proc, selector.R_frozen.T))
            sims = torch.matmul(proc, refs.T)
            own = sims[:, int(class_idx)]
            other_mask = torch.ones(int(refs.shape[0]), dtype=torch.bool, device=sims.device)
            other_mask[int(class_idx)] = False
            other = sims[:, other_mask]
            if select_mode == "own":
                scores = own
            elif select_mode == "mean_margin":
                scores = own - other.mean(dim=1)
            elif select_mode == "soft_margin":
                tau = 0.07
                scores = own - tau * torch.logsumexp(other / tau, dim=1)
            else:
                scores = own - other.max(dim=1).values
            scores_np = scores.detach().cpu().numpy()
            order = np.argsort(-scores_np)[: min(top_k, len(prompts))]
            selected[cls_name] = [prompts[int(i)] for i in order]
            for rank, idx in enumerate(order, start=1):
                selected_rows.append(
                    {
                        "class": cls_name,
                        "rank": int(rank),
                        "score": float(scores_np[int(idx)]),
                        "prompt": prompts[int(idx)],
                    }
                )
        info = {
            "enabled": True,
            "source": str(getattr(self.config, "EARLY_TEXT_PROMPT_SOURCE", "report")),
            "selection_mode": select_mode,
            "top_k": int(top_k),
            "entry_mode": str(getattr(self.config, "EARLY_TEXT_PROMPT_ENTRY_MODE", "full")),
            "candidate_count_per_class": {k: int(len(v)) for k, v in candidates.items()},
            "selected_count_per_class": {k: int(len(v)) for k, v in selected.items()},
            "selected_prompts": selected,
            "selected_rows": selected_rows,
        }
        return selected, info

    def _ensure_early_text_prompt_support(self) -> None:
        if not self._early_text_prompt_support_enabled():
            self.early_text_prompt_support_info = {"enabled": False}
            return
        existing = getattr(self.config, "PROMPT_TEXT_EMBEDDING_GROUPS", None)
        if isinstance(existing, dict) and existing:
            self.early_text_prompt_groups = {str(k): list(v) for k, v in existing.items()}
            self.early_text_prompt_support_info = {
                "enabled": True,
                "source": "preconfigured",
                "entry_mode": str(getattr(self.config, "EARLY_TEXT_PROMPT_ENTRY_MODE", "full")),
                "selected_count_per_class": {str(k): int(len(v)) for k, v in self.early_text_prompt_groups.items()},
            }
            setattr(self.config, "PROMPT_TEXT_ENTRY_MODE", str(getattr(self.config, "EARLY_TEXT_PROMPT_ENTRY_MODE", "full")))
            return

        selector_cfg = copy.deepcopy(self.config)
        selector_cfg.ENABLE_EARLY_TEXT_PROMPT_SUPPORT = False
        selector_cfg.PRINT_SUMMARY = False
        selector_cfg.VERBOSE = False
        selector_cfg.DEBUG = False
        selector_cfg.SAVE_DIR = os.path.join(
            str(self.config.SAVE_DIR),
            str(getattr(self.config, "EARLY_TEXT_PROMPT_SELECTOR_SUBDIR", "early_text_prompt_selector")),
        )
        selector = CAPA5NotebookRunner(selector_cfg)
        selector._init_state()
        selector.eval_runtime = selector._build_eval_runtime()
        selector.run_pipeline(run_stage4=False)

        selected, info = self._select_early_text_prompt_groups(selector)
        self.early_text_prompt_groups = selected
        self.early_text_prompt_support_info = info
        setattr(self.config, "PROMPT_TEXT_EMBEDDING_GROUPS", selected)
        setattr(self.config, "PROMPT_TEXT_ENTRY_MODE", str(getattr(self.config, "EARLY_TEXT_PROMPT_ENTRY_MODE", "full")))
        self.config.M = max(int(getattr(self.config, "M", 5)), int(getattr(self.config, "EARLY_TEXT_PROMPT_TOP_K", 12)))
        out_dir = os.path.join(str(self.config.SAVE_DIR), "audit")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "early_text_prompt_support.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=True, default=_json_default)
        self._log(
            f"[PromptSupport] selected top-{int(getattr(self.config, 'EARLY_TEXT_PROMPT_TOP_K', 12))} "
            "report prompts per class for early text prototype construction.",
            always=True,
        )

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

    def _project_multilabels_to_runtime_label_space(
        self,
        y,
        source_class_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, bool]:
        y_arr = np.asarray(y)
        if y_arr.ndim != 2:
            return y_arr.astype(np.int32), False
        if y_arr.shape[1] == 1:
            return y_arr[:, 0].astype(np.int32), False
        target_names = list(self.config.ORDERED_CLASS_NAMES)
        if y_arr.shape[1] == len(target_names):
            return (y_arr > 0).astype(np.int32), True

        names = None
        if source_class_names is not None:
            names = [str(x) for x in source_class_names]
        elif y_arr.shape[1] == len(self.config.SOURCE_CLASS_NAMES_14):
            names = list(self.config.SOURCE_CLASS_NAMES_14)

        if names is None:
            return (y_arr > 0).astype(np.int32), True

        index_map = {name: idx for idx, name in enumerate(names)}
        out = np.zeros((y_arr.shape[0], len(target_names)), dtype=np.int32)
        missing = []
        for j, cls_name in enumerate(target_names):
            src_idx = index_map.get(cls_name, None)
            if src_idx is None or src_idx >= y_arr.shape[1]:
                missing.append(cls_name)
                continue
            out[:, j] = (y_arr[:, src_idx] > 0).astype(np.int32)

        if missing:
            self._log(
                f"[LabelSpace] Missing labels for active space={self.config.LABEL_SPACE}: {missing}. They will stay zero.",
                always=True,
            )
        return out, True

    def _get_source_class_names_override(
        self,
        src: str,
        y,
    ) -> Optional[List[str]]:
        y_arr = np.asarray(y)
        if y_arr.ndim != 2 or int(y_arr.shape[1]) != len(FULL_14_CLASS_NAMES):
            return None
        profile = str(getattr(self.config, "SOURCE_LABEL_ORDER_PROFILE", "default"))
        if profile != "chexpert5_reordered_200x5":
            return None
        base = os.path.basename(str(src)).lower()
        if base in {"mimic_forchexpert-5_200x5.pkl", "mimic_200x5.pkl", "chexpert_200x5.pkl"}:
            self._log(
                f"[LabelOrder] Applying assumed CheXpert-5 reordered source order to {base}.",
                always=True,
            )
            return list(CHEXPERT5_REORDERED_SOURCE_ORDER_14)
        return None

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
            source_class_names = data.get("class_names", None)
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
                if source_class_names is None:
                    source_class_names = self._get_source_class_names_override(src, y)
                y, is_multi = self._project_multilabels_to_runtime_label_space(y, source_class_names=source_class_names)
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
                    y = np.stack(raw)  # 鐩存帴 stack 鎴?(N, C) 鐨?2D 鏁扮粍
                    source_class_names = self._get_source_class_names_override(src, y)
                    y, is_multi = self._project_multilabels_to_runtime_label_space(
                        y,
                        source_class_names=source_class_names,
                    )
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
        if bool(getattr(self.config, "DISABLE_SHARED_CENTERING", False)):
            return self._l2_norm(z)
        if mean_vec is None: return self._l2_norm(z)
        z_centered = z - mean_vec
        if self.config.USE_ZCA_WHITEN and self.W_zca is not None:
            z_centered = torch.matmul(z_centered, self.W_zca)
        return self._l2_norm(z_centered)

    def _prepare_shared_feature_space(self) -> Optional[torch.Tensor]:
        z_cal = None
        self.zI_mean = None
        self.W_zca = None
        if self._mode_uses_image_preprocessing():
            self._log("[Stage I] Shared preprocessing calibration")
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
        else:
            self._log("[Stage I] Shared preprocessing disabled by eval_mode.", always=True)

        self._build_prototypes()
        proto_ref = self.t_raw_text
        if self._mode_uses_image_preprocessing() and isinstance(self.t_processed_text, torch.Tensor):
            proto_ref = self.t_processed_text
        if not isinstance(proto_ref, torch.Tensor):
            raise RuntimeError("Prototype build failed; missing raw/processed text prototypes.")

        d = int(proto_ref.shape[1])
        n_cls = int(proto_ref.shape[0])
        self.current_R = torch.eye(d, device=self.device)
        self.R_frozen = torch.eye(d, device=self.device)
        self.image_centroids = proto_ref.clone()
        self.disc_neg_centroids = proto_ref.clone()
        self.disc_neg_counts = torch.zeros(n_cls, device=self.device)
        self.support_counts = torch.zeros(n_cls, device=self.device)
        self.rejected_counts = torch.zeros_like(self.support_counts)
        self.prior_counts = torch.zeros_like(self.support_counts)
        self.t_align_base = proto_ref.clone()
        self._set_prior_bias_from_counts(self.prior_counts)
        self._snapshot_last_good_state()
        self._refresh_aligned_text()

        if isinstance(z_cal, torch.Tensor):
            z_cal_proc = self._apply_preprocessing(z_cal, self.zI_mean)
            self._apply_param_profile_from_calibration(z_cal_proc)
        return z_cal

    def _prepare_eval_embeddings(self, z_embed: torch.Tensor) -> torch.Tensor:
        if self._mode_uses_image_preprocessing():
            return self._apply_preprocessing(z_embed, self.zI_mean)
        return z_embed.clone()

    def _get_eval_prototype_bundle(self) -> Tuple[torch.Tensor, Optional[torch.Tensor], str]:
        mode = self._current_eval_mode()
        if mode == "raw_baseline":
            if not isinstance(self.t_raw_text, torch.Tensor):
                raise RuntimeError("raw_baseline requires t_raw_text.")
            return self.t_raw_text.clone(), None, "t_raw_text"
        if mode == "preprocessed_baseline":
            if not isinstance(self.t_processed_text, torch.Tensor):
                raise RuntimeError("preprocessed_baseline requires t_processed_text.")
            return self.t_processed_text.clone(), None, "t_processed_text"

        aligned = self._refresh_aligned_text()
        if not isinstance(aligned, torch.Tensor):
            raise RuntimeError(f"{mode} requires aligned text prototypes.")
        source = "t_aligned_text"
        if bool(getattr(self.config, "ENABLE_RESIDUAL_LOCAL_SLERP", True)):
            aligned = self._build_residual_local_text(aligned)
            source = "t_aligned_text+residual_local_slerp"
        baseline_t = self.t_processed_text.clone() if isinstance(self.t_processed_text, torch.Tensor) else None
        return aligned.clone(), baseline_t, source

    def _set_eval_bias_for_prototypes(self, t_protos: torch.Tensor) -> None:
        if self._mode_uses_prior_correction() and isinstance(self.prior_counts, torch.Tensor):
            self._set_prior_bias_from_counts(self.prior_counts)
            return
        self.b_c = torch.zeros((1, int(t_protos.shape[0])), device=self.device)

    def _select_prompt_coreset_indices(self, embeds: torch.Tensor, k: int) -> List[int]:
        n = int(embeds.shape[0])
        k = max(1, min(int(k), n))
        if k >= n:
            return list(range(n))
        sims_to_mean = torch.matmul(embeds, embeds.mean(dim=0, keepdim=True).T).squeeze(1)
        first = int(torch.argmax(sims_to_mean).item())
        selected = [first]
        min_dist = 1.0 - torch.matmul(embeds, embeds[first].view(-1, 1)).squeeze(1)
        for _ in range(1, k):
            next_idx = int(torch.argmax(min_dist).item())
            if next_idx in selected:
                break
            selected.append(next_idx)
            cand_dist = 1.0 - torch.matmul(embeds, embeds[next_idx].view(-1, 1)).squeeze(1)
            min_dist = torch.minimum(min_dist, cand_dist)
        while len(selected) < k:
            for idx in range(n):
                if idx not in selected:
                    selected.append(idx)
                if len(selected) >= k:
                    break
        return selected[:k]

    def _get_prompt_bank_profile(self) -> str:
        return str(getattr(self.config, "PROMPT_BANK_PROFILE", "v2")).strip().lower()

    def _get_prompt_bucket_order(self) -> List[str]:
        profile = self._get_prompt_bank_profile()
        return list(PROMPT_BANK_PROFILE_BUCKET_ORDER.get(profile, PROMPT_BUCKET_ORDER))

    def _get_prompt_bucket_templates(self) -> Dict[str, List[str]]:
        profile = self._get_prompt_bank_profile()
        return dict(PROMPT_BANK_PROFILE_BUCKET_TEMPLATES.get(profile, PROMPT_BUCKET_TEMPLATES_V1))

    def _get_prompt_bucket_priors(self) -> Dict[str, float]:
        priors = dict(PROMPT_BANK_PROFILE_PRIORS.get(self._get_prompt_bank_profile(), DEFAULT_PROMPT_BUCKET_PRIORS))
        priors.update(dict(getattr(self.config, "PROMPT_BUCKET_PRIORS", {})))
        return priors

    def _get_prompt_class_mix(self, cls_name: str, default_mix: float) -> float:
        profile = str(getattr(self.config, "PROMPT_CLASS_MIX_PROFILE", "none")).strip().lower()
        overrides = dict(PROMPT_CLASS_MIX_PROFILE_MAP.get(profile, {}))
        if cls_name in overrides:
            return float(overrides[cls_name])
        return float(default_mix)

    def _build_flat_prompt_entries_for_class(self, cls_name: str) -> List[Dict[str, str]]:
        syns = list(self.config.MEDICAL_SYNONYM_MAP.get(cls_name, [cls_name]))
        templates = list(self.config.TEMPLATES_PI)
        target_m = max(1, int(self.config.M))
        if not bool(getattr(self.config, "ENABLE_PROMPT_CORESET", False)):
            cls_texts = [
                templates[i % len(templates)].replace("{finding}", syns[i % len(syns)])
                for i in range(target_m)
            ]
            return [{"bucket": "flat", "text": text} for text in cls_texts]

        candidate_texts: List[str] = []
        for template in templates:
            for syn in syns:
                candidate_texts.append(template.replace("{finding}", syn))
        seen = set()
        deduped = []
        for text in candidate_texts:
            if text not in seen:
                seen.add(text)
                deduped.append(text)
        max_candidates = max(target_m, int(getattr(self.config, "PROMPT_RESOURCE_MAX_CANDIDATES", 24)))
        deduped = deduped[:max_candidates]
        return [{"bucket": "flat", "text": text} for text in deduped]

    def _build_legacy_baseline_entries_for_class(self, cls_name: str) -> List[Dict[str, str]]:
        syns = list(self.config.MEDICAL_SYNONYM_MAP.get(cls_name, [cls_name]))
        templates = list(self.config.TEMPLATES_PI)
        target_m = max(1, int(self.config.M))
        cls_texts = [
            templates[i % len(templates)].replace("{finding}", syns[i % len(syns)])
            for i in range(target_m)
        ]
        return [{"bucket": "flat", "text": text} for text in cls_texts]

    def _get_structured_prompt_phrases(self, cls_name: str) -> Dict[str, List[str]]:
        label_space = str(getattr(self.config, "LABEL_SPACE", ""))
        bank_lookup = PROMPT_BANK_PROFILE_PHRASE_BANK.get(self._get_prompt_bank_profile(), STRUCTURED_PROMPT_PHRASE_BANK_CHEXPERT5)
        bucket_order = self._get_prompt_bucket_order()
        if label_space == "chexpert5" and cls_name in bank_lookup:
            bank = {}
            for bucket in bucket_order:
                if bucket == "legacy":
                    continue
                vals = list(bank_lookup[cls_name].get(bucket, []))
                if vals:
                    bank[bucket] = vals
            return bank

        syns = list(self.config.MEDICAL_SYNONYM_MAP.get(cls_name, [cls_name]))
        base = syns[0]
        bank: Dict[str, List[str]] = {
            "impression": [base],
            "finding": syns[: max(1, min(3, len(syns)))],
            "anatomy": [f"{base} on chest radiograph"],
            "morphology": [base],
            "severity": [f"mild {base}", f"moderate {base}"],
            "context": [f"frontal chest radiograph shows {base}"],
        }
        return bank

    def _build_prompt_bank_entries_for_class(self, cls_name: str) -> List[Dict[str, str]]:
        target_m = max(1, int(self.config.M))
        entries: List[Dict[str, str]] = []

        if not bool(getattr(self.config, "ENABLE_STRUCTURED_PROMPT_BANK", False)):
            return self._build_flat_prompt_entries_for_class(cls_name)

        phrase_bank = self._get_structured_prompt_phrases(cls_name)
        bucket_order = self._get_prompt_bucket_order()
        bucket_templates = self._get_prompt_bucket_templates()
        seen = set()
        if self._get_prompt_bank_profile() == "v3":
            syns = list(self.config.MEDICAL_SYNONYM_MAP.get(cls_name, [cls_name]))
            for template in list(self.config.TEMPLATES_PI):
                for syn in syns:
                    text = template.replace("{finding}", syn)
                    if text in seen:
                        continue
                    seen.add(text)
                    entries.append({"bucket": "legacy", "text": text})
        for bucket in bucket_order:
            if bucket == "legacy":
                continue
            phrases = list(phrase_bank.get(bucket, []))
            templates = list(bucket_templates.get(bucket, ["{phrase}"]))
            for phrase in phrases:
                for template in templates:
                    text = template.replace("{phrase}", phrase)
                    if text in seen:
                        continue
                    seen.add(text)
                    entries.append({"bucket": bucket, "text": text})

        max_candidates = max(target_m, int(getattr(self.config, "PROMPT_RESOURCE_MAX_CANDIDATES", 24)))
        if len(entries) > max_candidates:
            clipped: List[Dict[str, str]] = []
            per_bucket_cap = max(1, int(np.ceil(max_candidates / max(1, len(bucket_order)))))
            bucket_counts = {bucket: 0 for bucket in bucket_order}
            for item in entries:
                bucket = str(item["bucket"])
                if bucket_counts.get(bucket, 0) >= per_bucket_cap:
                    continue
                clipped.append(item)
                bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
                if len(clipped) >= max_candidates:
                    break
            entries = clipped
        return entries

    def _select_prompt_bank_entries(
        self,
        entries: List[Dict[str, str]],
        structured_bank: Optional[bool] = None,
    ) -> Tuple[List[Dict[str, str]], torch.Tensor]:
        texts = [str(item["text"]) for item in entries]
        embeds = self._encode_text(texts)
        if (not bool(getattr(self.config, "ENABLE_PROMPT_CORESET", False))) or int(embeds.shape[0]) <= 1:
            return entries, embeds

        is_structured = bool(structured_bank) if structured_bank is not None else bool(
            any(str(item.get("bucket", "")) not in {"flat"} for item in entries)
        )

        if is_structured:
            selected: List[int] = []
            target_total = min(
                int(embeds.shape[0]),
                max(int(self.config.M), int(getattr(self.config, "PROMPT_CORESET_SIZE", self.config.M))),
            )
            for bucket in self._get_prompt_bucket_order():
                idxs = [i for i, item in enumerate(entries) if str(item.get("bucket", "")) == bucket]
                if not idxs:
                    continue
                local_k = min(len(idxs), int(getattr(self.config, "PROMPT_BUCKET_KEEP", 2)))
                local_sel = self._select_prompt_coreset_indices(embeds[idxs], local_k)
                selected.extend([idxs[j] for j in local_sel])
            selected = list(dict.fromkeys(selected))
            if len(selected) < target_total:
                global_sel = self._select_prompt_coreset_indices(embeds, target_total)
                for idx in global_sel:
                    if idx not in selected:
                        selected.append(idx)
                    if len(selected) >= target_total:
                        break
        else:
            coreset_k = max(1, int(getattr(self.config, "PROMPT_CORESET_SIZE", self.config.M)))
            coreset_k = min(int(embeds.shape[0]), max(int(self.config.M), coreset_k))
            selected = self._select_prompt_coreset_indices(embeds, coreset_k)

        selected_entries = [entries[i] for i in selected]
        selected_embeds = embeds[selected]
        return selected_entries, selected_embeds

    def _get_prompt_reference_prototypes(self, provisional_text_prototypes: torch.Tensor) -> torch.Tensor:
        if (
            isinstance(self.image_centroids, torch.Tensor)
            and tuple(self.image_centroids.shape) == tuple(provisional_text_prototypes.shape)
        ):
            return self._l2_norm(self.image_centroids)
        return self._l2_norm(provisional_text_prototypes)

    def _prompt_margin_scores(
        self,
        embeds: torch.Tensor,
        class_idx: int,
        refs: torch.Tensor,
    ) -> torch.Tensor:
        sims = torch.matmul(embeds, refs.T)
        own = sims[:, int(class_idx)]
        if int(refs.shape[0]) <= 1:
            return own
        other_mask = torch.ones(int(refs.shape[0]), dtype=torch.bool, device=embeds.device)
        other_mask[int(class_idx)] = False
        other = sims[:, other_mask].max(dim=1).values
        return own - other

    def _pool_prompt_bank_for_class(
        self,
        class_idx: int,
        entries: List[Dict[str, str]],
        raw_embeds: torch.Tensor,
        proc_embeds: torch.Tensor,
        refs: torch.Tensor,
        structured_bank: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_m = max(1, int(self.config.M))
        is_structured = bool(structured_bank) if structured_bank is not None else bool(
            any(str(item.get("bucket", "")) not in {"flat"} for item in entries)
        )
        if (
            (not is_structured)
            or str(getattr(self.config, "PROMPT_POOLING_MODE", "mean")) != "bucketed"
        ):
            proto_raw = self._l2_norm(raw_embeds.mean(dim=0, keepdim=True)).squeeze(0)
            proto_proc = self._l2_norm(proc_embeds.mean(dim=0, keepdim=True)).squeeze(0)
            chosen = proc_embeds
        else:
            margin_scores = self._prompt_margin_scores(proc_embeds, class_idx, refs)
            bucket_proto_raw: List[torch.Tensor] = []
            bucket_proto_proc: List[torch.Tensor] = []
            bucket_scores: List[torch.Tensor] = []
            bucket_names: List[str] = []
            bucket_order = self._get_prompt_bucket_order()
            bucket_priors = self._get_prompt_bucket_priors()
            prompt_combined_scores = torch.full(
                (int(proc_embeds.shape[0]),),
                fill_value=-1e9,
                device=proc_embeds.device,
            )
            for bucket in bucket_order:
                idxs = [i for i, item in enumerate(entries) if str(item.get("bucket", "")) == bucket]
                if not idxs:
                    continue
                idx_tensor = torch.tensor(idxs, dtype=torch.long, device=proc_embeds.device)
                local_scores = margin_scores[idx_tensor]
                alpha = torch.softmax(local_scores / float(getattr(self.config, "PROMPT_SCORE_TEMP", 0.12)), dim=0)
                proto_b_proc = self._l2_norm((alpha.unsqueeze(1) * proc_embeds[idx_tensor]).sum(dim=0, keepdim=True)).squeeze(0)
                proto_b_raw = self._l2_norm((alpha.unsqueeze(1) * raw_embeds[idx_tensor]).sum(dim=0, keepdim=True)).squeeze(0)
                score_b = self._prompt_margin_scores(proto_b_proc.view(1, -1), class_idx, refs).squeeze(0)
                prior_b = max(1e-6, float(bucket_priors.get(bucket, 1.0)))
                score_b = score_b + float(np.log(prior_b))
                bucket_proto_raw.append(proto_b_raw)
                bucket_proto_proc.append(proto_b_proc)
                bucket_scores.append(score_b)
                bucket_names.append(bucket)
                prompt_combined_scores[idx_tensor] = local_scores + float(np.log(prior_b))

            if not bucket_proto_proc:
                proto_raw = self._l2_norm(raw_embeds.mean(dim=0, keepdim=True)).squeeze(0)
                proto_proc = self._l2_norm(proc_embeds.mean(dim=0, keepdim=True)).squeeze(0)
                chosen = proc_embeds
            else:
                bucket_scores_tensor = torch.stack(bucket_scores)
                beta = torch.softmax(
                    bucket_scores_tensor / float(getattr(self.config, "PROMPT_BUCKET_SCORE_TEMP", 0.18)),
                    dim=0,
                )
                proto_raw = self._l2_norm(
                    sum(beta[i] * bucket_proto_raw[i] for i in range(len(bucket_proto_raw))).view(1, -1)
                ).squeeze(0)
                proto_proc = self._l2_norm(
                    sum(beta[i] * bucket_proto_proc[i] for i in range(len(bucket_proto_proc))).view(1, -1)
                ).squeeze(0)
                for i, bucket in enumerate(bucket_names):
                    idxs = [j for j, item in enumerate(entries) if str(item.get("bucket", "")) == bucket]
                    if idxs:
                        idx_tensor = torch.tensor(idxs, dtype=torch.long, device=proc_embeds.device)
                        prompt_combined_scores[idx_tensor] = prompt_combined_scores[idx_tensor] + torch.log(beta[i].clamp_min(1e-8))
                topk = min(target_m, int(proc_embeds.shape[0]))
                top_idx = torch.topk(prompt_combined_scores, k=topk).indices
                chosen = proc_embeds[top_idx]

        if int(chosen.shape[0]) < target_m:
            rep_idx = [i % int(chosen.shape[0]) for i in range(target_m)]
            chosen = chosen[rep_idx]
        elif int(chosen.shape[0]) > target_m:
            chosen = chosen[:target_m]
        return proto_raw, proto_proc, chosen

    def _get_mode_alignment_reference(self) -> Optional[torch.Tensor]:
        if self._mode_uses_image_preprocessing() and isinstance(self.t_processed_text, torch.Tensor):
            return self.t_processed_text
        if isinstance(self.t_raw_text, torch.Tensor):
            return self.t_raw_text
        if isinstance(self.t_processed_text, torch.Tensor):
            return self.t_processed_text
        return None

    def _get_alignment_text_base(self) -> torch.Tensor:
        expected_base = self._get_mode_alignment_reference()
        if (
            self.t_align_base is not None
            and expected_base is not None
            and isinstance(self.t_align_base, torch.Tensor)
            and tuple(self.t_align_base.shape) == tuple(expected_base.shape)
        ):
            return self.t_align_base
        return expected_base

    def _sync_named_prototypes(self) -> None:
        self.t_raw_text = self.t_raw_pooled_raw.clone() if isinstance(self.t_raw_pooled_raw, torch.Tensor) else None
        self.t_processed_text = self.t_raw_pooled.clone() if isinstance(self.t_raw_pooled, torch.Tensor) else None
        self.t_aligned_text = None

    def _refresh_aligned_text(self) -> Optional[torch.Tensor]:
        base = self._get_alignment_text_base()
        if not isinstance(base, torch.Tensor):
            self.t_aligned_text = None
            return None
        if isinstance(self.R_frozen, torch.Tensor):
            self.t_aligned_text = self._l2_norm(torch.matmul(base, self.R_frozen.T))
        else:
            self.t_aligned_text = base.clone()
        return self.t_aligned_text

    def _get_text_entry_prompt_groups(self) -> Optional[Dict[str, List[str]]]:
        groups = getattr(self.config, "PROMPT_TEXT_EMBEDDING_GROUPS", None)
        if not isinstance(groups, dict) or not groups:
            return None
        return {str(k): [str(x) for x in list(v)] for k, v in groups.items()}

    def _build_default_text_entry_raw_candidates(self, cls_name: str) -> torch.Tensor:
        syns = list(self.config.MEDICAL_SYNONYM_MAP.get(cls_name, [cls_name]))
        templates = list(self.config.TEMPLATES_PI)
        target_m = 5
        texts = [
            templates[i % len(templates)].replace("{finding}", syns[i % len(syns)])
            for i in range(target_m)
        ]
        return self._encode_text(texts)

    def _repeat_or_clip_prompt_matrix(self, proc: torch.Tensor, target_m: int) -> torch.Tensor:
        if int(proc.shape[0]) <= 0:
            raise RuntimeError("Empty prompt candidate tensor.")
        if int(proc.shape[0]) < target_m:
            idx = [i % int(proc.shape[0]) for i in range(target_m)]
            return proc[idx]
        return proc[:target_m]

    def _build_early_text_entry_prototypes(self, groups: Dict[str, List[str]]) -> bool:
        entry_mode = str(getattr(self.config, "PROMPT_TEXT_ENTRY_MODE", getattr(self.config, "EARLY_TEXT_PROMPT_ENTRY_MODE", "full"))).strip().lower()
        if entry_mode not in ("full", "proto_only", "mean_only"):
            return False
        classes = list(self.config.ORDERED_CLASS_NAMES)
        default_raw_all: List[torch.Tensor] = []
        bank_raw_all: List[torch.Tensor] = []
        for cls_name in classes:
            default_raw_all.append(self._build_default_text_entry_raw_candidates(cls_name))
            prompts = list(groups.get(cls_name, []))
            if not prompts:
                prompts = [f"Impression: {cls_name.lower()}."]
            bank_raw_all.append(self._encode_text(prompts))

        default_proto_raw = torch.stack(
            [self._l2_norm(raw.mean(dim=0, keepdim=True)).squeeze(0) for raw in default_raw_all]
        )
        bank_proto_raw = torch.stack(
            [self._l2_norm(raw.mean(dim=0, keepdim=True)).squeeze(0) for raw in bank_raw_all]
        )
        default_zT_mean = default_proto_raw.mean(dim=0, keepdim=True)
        bank_zT_mean = bank_proto_raw.mean(dim=0, keepdim=True)

        self.t_zero_shot_base_raw = default_proto_raw
        self.t_zero_shot_base = torch.stack(
            [
                self._l2_norm(self._apply_preprocessing(raw, default_zT_mean).mean(dim=0, keepdim=True)).squeeze(0)
                for raw in default_raw_all
            ]
        )

        if entry_mode == "full":
            selected_raw_all = bank_raw_all
            self.zT_mean = bank_zT_mean
        elif entry_mode == "proto_only":
            selected_raw_all = bank_raw_all
            self.zT_mean = default_zT_mean
        else:
            selected_raw_all = default_raw_all
            self.zT_mean = bank_zT_mean

        target_m = max(1, int(getattr(self.config, "M", 5)))
        final_raw_list: List[torch.Tensor] = []
        final_proc_list: List[torch.Tensor] = []
        prompt_matrix_list: List[torch.Tensor] = []
        for raw in selected_raw_all:
            proc = self._apply_preprocessing(raw, self.zT_mean)
            final_raw_list.append(self._l2_norm(raw.mean(dim=0, keepdim=True)).squeeze(0))
            final_proc_list.append(self._l2_norm(proc.mean(dim=0, keepdim=True)).squeeze(0))
            prompt_matrix_list.append(self._repeat_or_clip_prompt_matrix(proc, target_m))

        self.t_raw_pooled_raw = torch.stack(final_raw_list)
        self.t_raw_pooled = torch.stack(final_proc_list)
        base_ref = self._get_mode_alignment_reference()
        self.t_align_base = base_ref.clone() if isinstance(base_ref, torch.Tensor) else self.t_raw_pooled.clone()
        self._sync_named_prototypes()
        self.cache_keys = None
        self.cache_labels = None
        self.cache_is_multi = False
        self.cache_ready = False
        self.cache_reference = None
        self.cache_reference_ready = False
        self.last_cache_eval_info = {}
        t_reshaped_proc = torch.stack(prompt_matrix_list)
        self.t_paraphrases = [self._l2_norm(t_reshaped_proc[:, m, :]) for m in range(target_m)]
        return True

    def _build_prototypes(self):
        entry_groups = self._get_text_entry_prompt_groups()
        if entry_groups and self._build_early_text_entry_prototypes(entry_groups):
            return

        classes = self.config.ORDERED_CLASS_NAMES
        legacy_base_raw_candidates: List[torch.Tensor] = []
        for cls_name in classes:
            base_entries = self._build_legacy_baseline_entries_for_class(cls_name)
            legacy_base_raw_candidates.append(self._encode_text([str(item["text"]) for item in base_entries]))
        self.t_zero_shot_base_raw = torch.stack(
            [self._l2_norm(raw.mean(dim=0, keepdim=True)).squeeze(0) for raw in legacy_base_raw_candidates]
        )
        legacy_zT_mean = self.t_zero_shot_base_raw.mean(dim=0, keepdim=True)
        self.t_zero_shot_base = torch.stack(
            [
                self._l2_norm(self._apply_preprocessing(raw, legacy_zT_mean).mean(dim=0, keepdim=True)).squeeze(0)
                for raw in legacy_base_raw_candidates
            ]
        )

        legacy_mix = float(getattr(self.config, "PROMPT_LEGACY_MIX", 0.0))
        if bool(getattr(self.config, "ENABLE_STRUCTURED_PROMPT_BANK", False)) and legacy_mix > 0.0:
            legacy_entries_all: List[List[Dict[str, str]]] = []
            legacy_raw_all: List[torch.Tensor] = []
            structured_entries_all: List[List[Dict[str, str]]] = []
            structured_raw_all: List[torch.Tensor] = []
            for cls_name in classes:
                legacy_entries = self._build_flat_prompt_entries_for_class(cls_name)
                legacy_sel_entries, legacy_sel_embeds = self._select_prompt_bank_entries(legacy_entries, structured_bank=False)
                legacy_entries_all.append(legacy_sel_entries)
                legacy_raw_all.append(legacy_sel_embeds)

                structured_entries = self._build_prompt_bank_entries_for_class(cls_name)
                structured_sel_entries, structured_sel_embeds = self._select_prompt_bank_entries(structured_entries, structured_bank=True)
                structured_entries_all.append(structured_sel_entries)
                structured_raw_all.append(structured_sel_embeds)

            provisional_raw = torch.stack(
                [
                    self._l2_norm(
                        (
                            (1.0 - self._get_prompt_class_mix(classes[i], legacy_mix)) * legacy_raw_all[i].mean(dim=0)
                            + self._get_prompt_class_mix(classes[i], legacy_mix) * structured_raw_all[i].mean(dim=0)
                        ).view(1, -1)
                    ).squeeze(0)
                    for i in range(len(classes))
                ]
            )
            self.zT_mean = provisional_raw.mean(dim=0, keepdim=True)

            legacy_proc_all: List[torch.Tensor] = []
            structured_proc_all: List[torch.Tensor] = []
            provisional_proc_mix: List[torch.Tensor] = []
            for i in range(len(classes)):
                legacy_proc = self._apply_preprocessing(legacy_raw_all[i], self.zT_mean)
                struct_proc = self._apply_preprocessing(structured_raw_all[i], self.zT_mean)
                legacy_proc_all.append(legacy_proc)
                structured_proc_all.append(struct_proc)
                legacy_mean = self._l2_norm(legacy_proc.mean(dim=0, keepdim=True)).squeeze(0)
                struct_mean = self._l2_norm(struct_proc.mean(dim=0, keepdim=True)).squeeze(0)
                mix_w = self._get_prompt_class_mix(classes[i], legacy_mix)
                provisional_proc_mix.append(
                    self._l2_norm(((1.0 - mix_w) * legacy_mean + mix_w * struct_mean).view(1, -1)).squeeze(0)
                )
            refs = self._get_prompt_reference_prototypes(torch.stack(provisional_proc_mix))

            final_raw_list: List[torch.Tensor] = []
            final_proc_list: List[torch.Tensor] = []
            prompt_matrix_list: List[torch.Tensor] = []
            for class_idx in range(len(classes)):
                mix_w = self._get_prompt_class_mix(classes[class_idx], legacy_mix)
                legacy_raw_proto, legacy_proc_proto, legacy_chosen = self._pool_prompt_bank_for_class(
                    class_idx,
                    legacy_entries_all[class_idx],
                    legacy_raw_all[class_idx],
                    legacy_proc_all[class_idx],
                    refs,
                    structured_bank=False,
                )
                struct_raw_proto, struct_proc_proto, struct_chosen = self._pool_prompt_bank_for_class(
                    class_idx,
                    structured_entries_all[class_idx],
                    structured_raw_all[class_idx],
                    structured_proc_all[class_idx],
                    refs,
                    structured_bank=True,
                )
                final_raw = self._l2_norm(
                    ((1.0 - mix_w) * legacy_raw_proto + mix_w * struct_raw_proto).view(1, -1)
                ).squeeze(0)
                final_proc = self._l2_norm(
                    ((1.0 - mix_w) * legacy_proc_proto + mix_w * struct_proc_proto).view(1, -1)
                ).squeeze(0)
                final_prompts = self._l2_norm((1.0 - mix_w) * legacy_chosen + mix_w * struct_chosen)
                final_raw_list.append(final_raw)
                final_proc_list.append(final_proc)
                prompt_matrix_list.append(final_prompts)

            self.t_raw_pooled_raw = torch.stack(final_raw_list)
            self.t_raw_pooled = torch.stack(final_proc_list)
            base_ref = self._get_mode_alignment_reference()
            self.t_align_base = base_ref.clone() if isinstance(base_ref, torch.Tensor) else self.t_raw_pooled.clone()
            self._sync_named_prototypes()
            self.cache_keys = None
            self.cache_labels = None
            self.cache_is_multi = False
            self.cache_ready = False
            self.cache_reference = None
            self.cache_reference_ready = False
            self.last_cache_eval_info = {}
            t_reshaped_proc = torch.stack(prompt_matrix_list)
            self.t_paraphrases = [self._l2_norm(t_reshaped_proc[:, m, :]) for m in range(self.config.M)]
            return

        class_entries: List[List[Dict[str, str]]] = []
        class_raw_candidates: List[torch.Tensor] = []
        for cls_name in classes:
            entries = self._build_prompt_bank_entries_for_class(cls_name)
            sel_entries, sel_embeds = self._select_prompt_bank_entries(
                entries,
                structured_bank=bool(getattr(self.config, "ENABLE_STRUCTURED_PROMPT_BANK", False)),
            )
            class_entries.append(sel_entries)
            class_raw_candidates.append(sel_embeds)

        provisional_raw = torch.stack(
            [
                self._l2_norm(raw.mean(dim=0, keepdim=True)).squeeze(0)
                for raw in class_raw_candidates
            ]
        )
        self.zT_mean = provisional_raw.mean(dim=0, keepdim=True)

        class_proc_candidates: List[torch.Tensor] = []
        provisional_proc_list: List[torch.Tensor] = []
        for raw in class_raw_candidates:
            proc = self._apply_preprocessing(raw, self.zT_mean)
            class_proc_candidates.append(proc)
            provisional_proc_list.append(self._l2_norm(proc.mean(dim=0, keepdim=True)).squeeze(0))
        provisional_proc = torch.stack(provisional_proc_list)
        refs = self._get_prompt_reference_prototypes(provisional_proc)

        final_raw_list: List[torch.Tensor] = []
        final_proc_list: List[torch.Tensor] = []
        prompt_matrix_list: List[torch.Tensor] = []
        for class_idx in range(len(classes)):
            proto_raw, proto_proc, chosen_proc = self._pool_prompt_bank_for_class(
                class_idx,
                class_entries[class_idx],
                class_raw_candidates[class_idx],
                class_proc_candidates[class_idx],
                refs,
                structured_bank=bool(getattr(self.config, "ENABLE_STRUCTURED_PROMPT_BANK", False)),
            )
            final_raw_list.append(proto_raw)
            final_proc_list.append(proto_proc)
            prompt_matrix_list.append(chosen_proc)

        self.t_raw_pooled_raw = torch.stack(final_raw_list)
        self.t_raw_pooled = torch.stack(final_proc_list)
        base_ref = self._get_mode_alignment_reference()
        self.t_align_base = base_ref.clone() if isinstance(base_ref, torch.Tensor) else self.t_raw_pooled.clone()
        self._sync_named_prototypes()
        self.cache_keys = None
        self.cache_labels = None
        self.cache_is_multi = False
        self.cache_ready = False
        self.cache_reference = None
        self.cache_reference_ready = False
        self.last_cache_eval_info = {}
        t_reshaped_proc = torch.stack(prompt_matrix_list)
        self.t_paraphrases = [self._l2_norm(t_reshaped_proc[:, m, :]) for m in range(self.config.M)]

    def _set_prior_bias_from_counts(self, counts: torch.Tensor):
        tau_prior = self._effective_prior_strength()
        if tau_prior <= 0.0:
            self.b_c = torch.zeros((1, self.t_raw_pooled.shape[0]), device=self.device)
            return
        cnt = counts.detach().to(self.device).float()
        priors = (cnt + 1.0) / (cnt.sum() + float(len(cnt)))
        self.b_c = (tau_prior * torch.log(priors.clamp_min(1e-8))).view(1, -1)

    def _is_professor_profile(self) -> bool:
        return str(getattr(self.config, "PARAM_PROFILE", "default")) == "professor"

    def _get_alignment_active_mask(self, n_cls: Optional[int] = None) -> torch.Tensor:
        if self.support_counts is None:
            size = len(self.config.ORDERED_CLASS_NAMES) if n_cls is None else int(n_cls)
            return torch.ones(size, dtype=torch.bool, device=self.device)
        base = (self.support_counts >= self.config.N_MIN_SUPPORT_FOR_ACTIVE).to(self.device)
        if n_cls is not None:
            base = base[: int(n_cls)]
        return base

    def _get_alignment_class_weights(self, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.support_counts is None:
            size = len(self.config.ORDERED_CLASS_NAMES)
            return torch.ones(size, device=self.device) / max(1, size)
        if mask is None:
            mask = self._get_alignment_active_mask()
        n_cap = max(1, int(getattr(self.config, "N_CAP", 500)))
        n_capped = torch.clamp(self.support_counts, max=float(n_cap))
        w = (torch.pow(n_capped, self.config.GAMMA_WEIGHT) + self.config.ALPHA_WEIGHT)
        t_ref = self._get_alignment_text_base()
        sim = (self.image_centroids * t_ref).sum(dim=1)
        w *= (1.0 + self.config.BETA_WEIGHT * torch.clamp(self.config.RHO - sim, min=0))
        if mask.any():
            w_med = torch.median(w[mask])
            w_cap = self.config.PROCRUSTES_WEIGHT_CAP_MULT * (w_med + 1e-6)
            w = torch.minimum(w, torch.full_like(w, w_cap))
        w = w * mask.float()
        return w / w.sum().clamp_min(1e-8)

    def _slerp_unit(self, v0: torch.Tensor, v1: torch.Tensor, lam: float) -> torch.Tensor:
        lam = min(1.0, max(0.0, float(lam)))
        if lam <= 0.0:
            return self._l2_norm(v0)
        if lam >= 1.0:
            return self._l2_norm(v1)
        a = self._l2_norm(v0)
        b = self._l2_norm(v1)
        dot = float(torch.clamp(torch.dot(a, b), -1.0 + 1e-6, 1.0 - 1e-6).item())
        if abs(dot) > 0.9995:
            return self._l2_norm((1.0 - lam) * a + lam * b)
        theta = float(np.arccos(dot))
        sin_theta = max(np.sin(theta), 1e-6)
        w0 = np.sin((1.0 - lam) * theta) / sin_theta
        w1 = np.sin(lam * theta) / sin_theta
        out = float(w0) * a + float(w1) * b
        return self._l2_norm(out)

    def _min_interclass_angle_deg(self, t: torch.Tensor) -> float:
        n_cls = int(t.shape[0])
        if n_cls <= 1:
            return float("nan")
        sims = torch.matmul(self._l2_norm(t), self._l2_norm(t).T)
        sims = sims.masked_fill(torch.eye(n_cls, dtype=torch.bool, device=self.device), -1.0)
        max_cos = float(torch.max(sims).item())
        return float(np.degrees(np.arccos(np.clip(max_cos, -1.0, 1.0))))

    def _build_residual_local_text(self, t0: torch.Tensor) -> torch.Tensor:
        self.residual_local_slerp_info = []
        if not bool(getattr(self.config, "ENABLE_RESIDUAL_LOCAL_SLERP", True)):
            return t0
        if not isinstance(self.image_centroids, torch.Tensor) or not isinstance(self.support_counts, torch.Tensor):
            return t0

        tau_rad = float(np.radians(float(getattr(self.config, "RESIDUAL_LOCAL_TAU_DELTA_DEG", 25.0))))
        n_min = int(getattr(self.config, "RESIDUAL_LOCAL_N_MIN", 100))
        eta = float(getattr(self.config, "RESIDUAL_LOCAL_ETA", 0.20))
        lam_max = float(getattr(self.config, "RESIDUAL_LOCAL_LAMBDA_MAX", 0.08))
        floor_cos = float(np.cos(np.radians(float(getattr(self.config, "RESIDUAL_LOCAL_ANGLE_FLOOR_DEG", 23.0)))))
        min_gain = float(getattr(self.config, "RESIDUAL_LOCAL_MIN_GAIN", 0.0))
        mu = self._l2_norm(self.image_centroids)
        t1 = t0.clone()
        class_names = list(self.config.ORDERED_CLASS_NAMES)
        min_angle_before = self._min_interclass_angle_deg(t0)

        for c_idx in range(int(t0.shape[0])):
            cos_before = float(torch.clamp(torch.dot(self._l2_norm(t0[c_idx]), self._l2_norm(mu[c_idx])), -1.0, 1.0).item())
            delta_rad = float(np.arccos(np.clip(cos_before, -1.0, 1.0)))
            support = float(self.support_counts[c_idx].item())
            residual_ok = bool(delta_rad > tau_rad)
            support_ok = bool(support >= n_min)
            alpha_cand = min(lam_max, eta * delta_rad) if residual_ok and support_ok else 0.0
            cand = self._slerp_unit(t0[c_idx], mu[c_idx], alpha_cand) if alpha_cand > 0.0 else t0[c_idx]
            cos_after = float(torch.clamp(torch.dot(self._l2_norm(cand), self._l2_norm(mu[c_idx])), -1.0, 1.0).item())
            gain = cos_after - cos_before
            tmp = t1.clone()
            tmp[c_idx] = cand
            off = torch.matmul(self._l2_norm(tmp), self._l2_norm(tmp).T)
            off = off.masked_fill(torch.eye(tmp.shape[0], dtype=torch.bool, device=self.device), -1.0)
            max_pair_cos = float(torch.max(off).item())
            gain_ok = bool(gain > min_gain)
            angle_ok = bool(max_pair_cos <= floor_cos)
            moved = bool(alpha_cand > 0.0 and gain_ok and angle_ok)
            if moved:
                t1[c_idx] = cand
            self.residual_local_slerp_info.append(
                {
                    "class": class_names[c_idx] if c_idx < len(class_names) else str(c_idx),
                    "support_count": support,
                    "residual_deg": float(np.degrees(delta_rad)),
                    "alpha": float(alpha_cand if moved else 0.0),
                    "candidate_alpha": float(alpha_cand),
                    "cos_before": cos_before,
                    "cos_after_candidate": cos_after,
                    "paired_cosine_gain": gain,
                    "residual_ok": residual_ok,
                    "support_ok": support_ok,
                    "gain_ok": gain_ok,
                    "angle_ok": angle_ok,
                    "moved": moved,
                    "max_pair_cos_after_candidate": max_pair_cos,
                    "min_angle_before_deg": min_angle_before,
                    "min_angle_after_running_deg": self._min_interclass_angle_deg(t1),
                }
            )
        return self._l2_norm(t1)

    def _build_slerp_rotation_candidate(self, R_seed: torch.Tensor, alpha: float) -> Optional[Tuple[torch.Tensor, float]]:
        if not bool(getattr(self.config, "ENABLE_CAPAV1_GUARDED_SLERP", False)):
            return None
        mask = self._get_alignment_active_mask()
        idx = torch.where(mask)[0]
        if int(idx.numel()) < 2:
            return None
        lam_max = max(0.0, float(getattr(self.config, "CAPAV1_GUARDED_SLERP_LAMBDA_MAX", 0.10)))
        lam = min(lam_max, max(0.0, lam_max * float(alpha)))
        if lam <= 0.0:
            return None
        t_base = self._get_alignment_text_base()
        t_rot = self._l2_norm(torch.matmul(t_base, R_seed.T))
        target = t_rot.clone()
        mu = self._l2_norm(self.image_centroids)
        for c_idx in idx.tolist():
            target[c_idx] = self._slerp_unit(t_rot[c_idx], mu[c_idx], lam)
        w = self._get_alignment_class_weights(mask)
        base_act = t_base.index_select(0, idx)
        target_act = target.index_select(0, idx)
        w_act = w.index_select(0, idx)
        m_cov = torch.matmul((w_act.view(-1, 1) * target_act).T, base_act)
        return self._orthogonalize_rotation(m_cov), float(lam)

    def _snapshot_last_good_state(self) -> None:
        if isinstance(self.current_R, torch.Tensor):
            self.R_last_good = self.current_R.detach().clone()
        if isinstance(self.image_centroids, torch.Tensor):
            self.centroids_last_good = self.image_centroids.detach().clone()

    def _professor_target_entropy(self, n_cls: int) -> float:
        n_cls = max(1, int(n_cls))
        if n_cls <= 1:
            return 0.0
        p1 = float(np.clip(getattr(self.config, "PROF_TARGET_TOP1", 0.75), 1e-4, 1.0 - 1e-4))
        p_rest = (1.0 - p1) / max(1, (n_cls - 1))
        return float(-(p1 * np.log(p1) + (n_cls - 1) * p_rest * np.log(max(p_rest, 1e-8))))

    def _estimate_scale_for_target_top1(self, z_ref: torch.Tensor, t_ref: torch.Tensor) -> float:
        if z_ref.numel() == 0:
            return float(self.s_opt)
        logits_base = torch.matmul(z_ref, t_ref.T) + self.b_c
        target = float(np.clip(getattr(self.config, "PROF_TARGET_TOP1", 0.75), 0.5, 0.99))
        candidates = np.geomspace(0.25, 20.0, num=31)
        best_s = float(self.s_opt)
        best_gap = float("inf")
        with torch.no_grad():
            for s_val in candidates:
                probs = F.softmax(logits_base * float(s_val), dim=1)
                top1_med = float(torch.median(probs.max(dim=1).values).item())
                gap = abs(top1_med - target)
                if gap < best_gap:
                    best_gap = gap
                    best_s = float(s_val)
        return best_s

    def _estimate_temperature_for_target_entropy(self, logits_ref: torch.Tensor) -> float:
        if logits_ref.numel() == 0:
            return float(self.T_opt)
        target_h = self._professor_target_entropy(int(logits_ref.shape[1]))
        candidates = np.geomspace(0.25, 4.0, num=31)
        best_t = float(self.T_opt)
        best_gap = float("inf")
        with torch.no_grad():
            for t_val in candidates:
                probs = F.softmax(logits_ref / float(t_val), dim=1)
                ent = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1).mean().item()
                gap = abs(float(ent) - target_h)
                if gap < best_gap:
                    best_gap = gap
                    best_t = float(t_val)
        return best_t

    def _apply_param_profile_from_calibration(self, z_cal_proc: torch.Tensor):
        if not self._is_professor_profile():
            return
        if not self._mode_uses_runtime_scale():
            return
        if z_cal_proc is None or z_cal_proc.numel() == 0:
            return

        n_ref = min(int(len(z_cal_proc)), 2048)
        z_ref = z_cal_proc[:n_ref]
        t_ref = self._get_alignment_text_base()

        if bool(getattr(self.config, "PROF_AUTO_SCALE", False)):
            self.s_opt = self._estimate_scale_for_target_top1(z_ref, t_ref)
        logits_ref = self.s_opt * torch.matmul(z_ref, t_ref.T) + self.b_c
        if bool(getattr(self.config, "PROF_AUTO_TEMPERATURE", False)):
            self.T_opt = self._estimate_temperature_for_target_entropy(logits_ref)
        self.config.INIT_SCALE_FACTOR = float(self.s_opt)
        self.config.INIT_TEMPERATURE = float(self.T_opt)

        self._log(
            (
                f"[Profile:Professor] s={self.s_opt:.4f}, T={self.T_opt:.4f}, "
                f"rho_q={self.config.RHO_QUANTILE:.2f}, "
                f"eps_pct={float(getattr(self.config, 'PROF_EPSILON_GAP_PCT', 0.015)):.3f}, "
                f"psi_thr={float(self.config.GO_PSI_THR):.3f}, lambda={float(self.config.CAPA_BASELINE_FUSION_LAMBDA):.3f}"
            ),
            always=True,
        )

    def _is_go_guardian_enabled(self) -> bool:
        return bool(self.eval_runtime.get("guardian", False))

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
        self.guardian_stage2_ref_mean = float(vals.mean()) if vals.size > 0 else np.nan
        self.guardian_stage2_stat = 0.0
        self.guardian_stage2_last = np.nan
        self.guardian_stage2_steps = 0
        self._log(
            f"[GO] Baseline ready: n={int(vals.size)}, bins={n_bins}, "
            f"psi_thr={float(self.config.GO_PSI_THR):.3f}, resume={float(self.config.GO_TAU_RESUME):.3f}x{int(self.config.GO_RESUME_WINDOWS)}",
            always=True,
        )

    def _guardian_batch_scalar_mean(self, probs: torch.Tensor) -> Optional[float]:
        if probs is None or probs.numel() <= 0:
            return None
        scalar_name = str(getattr(self.config, "GO_GUARDIAN_SCALAR", "top1_conf")).strip().lower()
        if scalar_name == "top1_conf":
            vals = probs.max(dim=1).values.detach()
        else:
            vals = probs.max(dim=1).values.detach()
        if vals.numel() <= 0:
            return None
        return float(vals.mean().item())

    def _guardian_stage2_update(self, probs: torch.Tensor) -> Optional[Dict[str, float]]:
        if not bool(getattr(self.config, "ENABLE_GO_GUARDIAN_STAGE2", False)):
            return None
        x = self._guardian_batch_scalar_mean(probs)
        ref_mean = float(self.guardian_stage2_ref_mean)
        if x is None or (not np.isfinite(ref_mean)):
            return None
        delta = max(0.0, float(getattr(self.config, "GO_STAGE2_DELTA", 0.01)))
        alarm_lambda = max(1e-6, float(getattr(self.config, "GO_STAGE2_LAMBDA", 0.08)))
        min_steps = max(1, int(getattr(self.config, "GO_STAGE2_MIN_STEPS", 3)))
        # Detect downward confidence drift relative to baseline mean.
        drift = max(0.0, ref_mean - x - delta)
        self.guardian_stage2_stat = max(0.0, float(self.guardian_stage2_stat) + drift)
        self.guardian_stage2_last = float(self.guardian_stage2_stat)
        self.guardian_stage2_steps += 1
        alarm = (self.guardian_stage2_steps >= min_steps) and (self.guardian_stage2_stat > alarm_lambda)
        if not alarm and x >= (ref_mean - delta):
            self.guardian_stage2_stat *= 0.5
            self.guardian_stage2_last = float(self.guardian_stage2_stat)
        return {
            "stat": float(self.guardian_stage2_last),
            "ref_mean": ref_mean,
            "batch_mean": float(x),
            "alarm": bool(alarm),
        }

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
                    self._snapshot_last_good_state()
            return {"step": int(step), "status": str(self.guardian_status), "phase": "baseline_collect"}

        if self.guardian_psi_baseline_hist is None or self.guardian_psi_bin_edges is None:
            self.guardian_status = "baseline_collect"
            self._guardian_collect_batch_scalars(probs)
            self._guardian_init_baseline()
            return {"step": int(step), "status": str(self.guardian_status), "phase": "baseline_collect_fallback"}

        if str(self.guardian_status) not in ("normal", "frozen"):
            self.guardian_status = "normal"
            self.guardian_resume_streak = 0
            self._snapshot_last_good_state()

        self._guardian_collect_batch_scalars(probs)
        psi = self._guardian_window_psi()
        if psi is None:
            return None

        psi = float(psi)
        stage2_info = self._guardian_stage2_update(probs)
        stage2_alarm = bool(stage2_info.get("alarm", False)) if isinstance(stage2_info, dict) else False
        self.guardian_last_psi = psi
        self.guardian_psi_history.append(psi)
        changed = False
        dry_run = bool(getattr(self.config, "GO_DRY_RUN", False))
        psi_thr = float(getattr(self.config, "GO_PSI_THR", 2.0))
        tau_resume = float(getattr(self.config, "GO_TAU_RESUME", 1.0))
        resume_windows = max(1, int(getattr(self.config, "GO_RESUME_WINDOWS", 3)))

        if self.guardian_status == "normal":
            if (psi > psi_thr) or stage2_alarm:
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
            if isinstance(self.centroids_last_good, torch.Tensor):
                self.image_centroids = self.centroids_last_good.clone()

        return {
            "step": int(step),
            "psi": psi,
            "status": str(self.guardian_status),
            "changed": bool(changed),
            "frozen": bool(self._guardian_is_frozen()),
            "dry_run": bool(dry_run),
            "stage2_stat": float(stage2_info.get("stat", np.nan)) if isinstance(stage2_info, dict) else np.nan,
            "stage2_alarm": bool(stage2_alarm),
        }

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
        return bool(self.eval_runtime.get("cache", False))

    def _default_cache_eval_info(self) -> Dict[str, float]:
        return {
            "mode": str(self.eval_runtime.get("cache_mode", "off")),
            "dataset_gate": False,
            "psi_top1": np.nan,
            "psi_topk_mean": np.nan,
            "psi_entropy": np.nan,
            "usage_rate": 0.0,
            "mean_alpha": 0.0,
            "agree_rate": np.nan,
            "mean_top1_sim": np.nan,
            "mean_purity": np.nan,
            "mean_entropy": np.nan,
            "tau_sim": np.nan,
            "tau_purity": np.nan,
            "tau_entropy": np.nan,
        }

    def _default_dualtrack_eval_info(self) -> Dict[str, float]:
        return {
            "enabled": bool(self._uses_dual_track_inference()),
            "aligned_rate": np.nan,
            "agree_rate": np.nan,
            "abstain_rate": 0.0,
            "conf_aligned_mean": np.nan,
            "conf_raw_mean": np.nan,
        }

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

    def _compute_cache_payload(self, z_embed: torch.Tensor) -> Optional[Dict[str, torch.Tensor]]:
        if not self._build_global_cache():
            return None
        keys = self.cache_keys
        labels = self.cache_labels
        if keys is None or labels is None:
            return None
        if keys.numel() == 0 or labels.numel() == 0:
            return None

        cache_topk = max(1, min(int(getattr(self.config, "CACHE_TOPK", 16)), int(keys.shape[0])))
        cache_temp = max(float(getattr(self.config, "CACHE_TEMP", 0.08)), 1e-4)
        chunk_size = max(16, int(getattr(self.config, "CACHE_CHUNK", 512)))
        self_match_cos = float(getattr(self.config, "CACHE_SELF_MATCH_COS", 0.999999))

        z_cpu = z_embed.detach().float().cpu()
        logits_chunks: List[torch.Tensor] = []
        top1_chunks: List[torch.Tensor] = []
        topk_mean_chunks: List[torch.Tensor] = []
        q_chunks: List[torch.Tensor] = []
        purity_chunks: List[torch.Tensor] = []
        entropy_chunks: List[torch.Tensor] = []
        top1_idx_chunks: List[torch.Tensor] = []

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
            p_ind = (w.unsqueeze(-1) * neigh).sum(dim=1).clamp(1e-4, 1.0 - 1e-4)
            q_mass = p_ind.clamp_min(1e-8)
            q_norm = q_mass / q_mass.sum(dim=1, keepdim=True).clamp_min(1e-8)
            entropy = -(q_norm * torch.log(q_norm.clamp_min(1e-8))).sum(dim=1)
            purity, cache_top1 = q_norm.max(dim=1)
            logits_chunks.append(torch.log(p_ind / (1.0 - p_ind)))
            top1_chunks.append(topv.max(dim=1).values)
            topk_mean_chunks.append(topv.mean(dim=1))
            q_chunks.append(q_norm)
            purity_chunks.append(purity)
            entropy_chunks.append(entropy)
            top1_idx_chunks.append(cache_top1)

        return {
            "cache_logits": torch.cat(logits_chunks, dim=0).to(z_embed.device),
            "top1_sim": torch.cat(top1_chunks, dim=0).to(z_embed.device),
            "topk_mean_sim": torch.cat(topk_mean_chunks, dim=0).to(z_embed.device),
            "q_cache": torch.cat(q_chunks, dim=0).to(z_embed.device),
            "purity": torch.cat(purity_chunks, dim=0).to(z_embed.device),
            "entropy": torch.cat(entropy_chunks, dim=0).to(z_embed.device),
            "cache_top1": torch.cat(top1_idx_chunks, dim=0).to(z_embed.device),
        }

    def _ensure_cache_reference_stats(self) -> bool:
        if self.cache_reference_ready and isinstance(self.cache_reference, dict):
            return True
        if not self._is_cache_enabled():
            return False
        try:
            z_ref_raw, _, _ = self._load_data(
                self.config.CALIB_DATA_PATH,
                is_calibration=True,
                split_override=1,
            )
            z_ref = self._apply_preprocessing(z_ref_raw, self.zI_mean)
            payload = self._compute_cache_payload(z_ref)
        except Exception as e:
            self._log(f"[Cache] reference build skipped: {e}", always=True)
            return False
        if payload is None:
            return False

        top1 = payload["top1_sim"].detach().cpu().numpy()
        topk_mean = payload["topk_mean_sim"].detach().cpu().numpy()
        purity = payload["purity"].detach().cpu().numpy()
        entropy = payload["entropy"].detach().cpu().numpy()
        self.cache_reference = {
            "top1_sim": top1,
            "topk_mean_sim": topk_mean,
            "purity": purity,
            "entropy": entropy,
            "tau_sim": float(np.quantile(top1, float(getattr(self.config, "CACHE_MIN_SIM_Q", 0.25)))),
            "tau_purity": float(np.quantile(purity, float(getattr(self.config, "CACHE_MIN_PURITY_Q", 0.50)))),
            "tau_entropy": float(np.quantile(entropy, float(getattr(self.config, "CACHE_MAX_ENTROPY_Q", 0.75)))),
        }
        self.cache_reference_ready = True
        return True

    def _blend_with_cache_logits(
        self,
        logits: torch.Tensor,
        z_embed: torch.Tensor,
        dataset_name: Optional[str],
        *,
        use_cache: bool,
        calib_T: Optional[float] = None,
        scoring_mode: Optional[str] = None,
    ) -> torch.Tensor:
        self.last_cache_eval_info = self._default_cache_eval_info()
        if not use_cache:
            return logits
        if not self._is_cache_enabled():
            return logits
        if not self._ensure_cache_reference_stats():
            return logits

        payload = self._compute_cache_payload(z_embed)
        if payload is None:
            return logits

        ref = self.cache_reference or {}
        top1_np = payload["top1_sim"].detach().cpu().numpy()
        topk_np = payload["topk_mean_sim"].detach().cpu().numpy()
        entropy_np = payload["entropy"].detach().cpu().numpy()
        psi_top1 = float(self._compute_psi(ref["top1_sim"], top1_np))
        psi_topk = float(self._compute_psi(ref["topk_mean_sim"], topk_np))
        psi_entropy = float(self._compute_psi(ref["entropy"], entropy_np))
        psi_thr = float(getattr(self.config, "CACHE_DATASET_PSI_THR", 0.25))
        dataset_gate = (psi_top1 <= psi_thr) and (psi_topk <= psi_thr) and (psi_entropy <= psi_thr)

        tau_sim = float(ref.get("tau_sim", np.nan))
        tau_purity = float(ref.get("tau_purity", np.nan))
        tau_entropy = float(ref.get("tau_entropy", np.nan))
        actual_agree = payload["cache_top1"] == logits.argmax(dim=1)

        if not dataset_gate:
            self.last_cache_eval_info = {
                **self._default_cache_eval_info(),
                "mode": "gated",
                "dataset_gate": False,
                "psi_top1": psi_top1,
                "psi_topk_mean": psi_topk,
                "psi_entropy": psi_entropy,
                "agree_rate": float(actual_agree.float().mean().item()),
                "mean_top1_sim": float(payload["top1_sim"].mean().item()),
                "mean_purity": float(payload["purity"].mean().item()),
                "mean_entropy": float(payload["entropy"].mean().item()),
                "tau_sim": tau_sim,
                "tau_purity": tau_purity,
                "tau_entropy": tau_entropy,
            }
            return logits

        mode = self._resolve_scoring_mode(scoring_mode)
        t_val = max(float(self.T_opt if calib_T is None else calib_T), 1e-4)
        if mode == "softmax":
            base_probs_full = F.softmax(logits / t_val, dim=1)
        else:
            base_probs_full = torch.sigmoid(logits / t_val)
        base_top1 = base_probs_full.argmax(dim=1)
        max_p_base = base_probs_full.max(dim=1).values

        agree_mask = (payload["cache_top1"] == base_top1).float()
        if bool(getattr(self.config, "CACHE_REQUIRE_AGREE", True)):
            agree_gate = agree_mask
        else:
            agree_gate = torch.ones_like(agree_mask)
        sim_gate = (payload["top1_sim"] >= tau_sim).float()
        purity_gate = (payload["purity"] >= tau_purity).float()
        entropy_ok = (payload["entropy"] <= tau_entropy).float()
        log_c = max(float(np.log(max(2, int(payload["cache_logits"].shape[1])))), 1e-6)
        entropy_term = torch.clamp(1.0 - (payload["entropy"] / log_c), min=0.0, max=1.0)
        uncertainty_term = torch.clamp(1.0 - max_p_base, min=0.0, max=1.0)
        alpha = (
            float(getattr(self.config, "CACHE_ALPHA_MAX", 0.10))
            * agree_gate
            * sim_gate
            * purity_gate
            * entropy_ok
            * entropy_term
            * uncertainty_term
        ).clamp(min=0.0, max=float(getattr(self.config, "CACHE_ALPHA_MAX", 0.10)))

        cache_logits = payload["cache_logits"]
        n_use = min(int(logits.shape[1]), int(cache_logits.shape[1]))
        if n_use <= 0:
            return logits
        out = logits.clone()
        out[:, :n_use] = (
            (1.0 - alpha).unsqueeze(1) * out[:, :n_use]
            + alpha.unsqueeze(1) * cache_logits[:, :n_use]
        )
        self.last_cache_eval_info = {
            "mode": "gated",
            "dataset_gate": True,
            "psi_top1": psi_top1,
            "psi_topk_mean": psi_topk,
            "psi_entropy": psi_entropy,
            "usage_rate": float((alpha > 0).float().mean().item()),
            "mean_alpha": float(alpha.mean().item()),
            "agree_rate": float(agree_mask.mean().item()),
            "mean_top1_sim": float(payload["top1_sim"].mean().item()),
            "mean_purity": float(payload["purity"].mean().item()),
            "mean_entropy": float(payload["entropy"].mean().item()),
            "tau_sim": tau_sim,
            "tau_purity": tau_purity,
            "tau_entropy": tau_entropy,
        }
        return out

    def _is_soft_fusion_enabled(self) -> bool:
        if not bool(self.eval_runtime.get("soft_fusion", False)):
            return False
        lam = self._effective_fusion_lambda()
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
        lam = min(1.0, max(0.0, self._effective_fusion_lambda()))
        logits_base = scale * torch.matmul(z_embed, baseline_t_protos.T) + self.b_c
        n_use = min(int(logits_capa.shape[1]), int(logits_base.shape[1]))
        if n_use <= 0:
            return logits_capa
        out = logits_capa.clone()
        out[:, :n_use] = lam * out[:, :n_use] + (1.0 - lam) * logits_base[:, :n_use]
        return out

    def _dual_track_confidence(self, logits: torch.Tensor) -> torch.Tensor:
        t_val = max(float(getattr(self, "T_opt", 1.0)), 1e-4)
        mode = self._resolve_scoring_mode(None)
        if mode == "softmax":
            return F.softmax(logits / t_val, dim=1).max(dim=1).values
        return torch.sigmoid(logits / t_val).max(dim=1).values

    def _select_dual_track_logits(
        self,
        logits_aligned: torch.Tensor,
        z_embed: torch.Tensor,
        *,
        scale: float,
        baseline_t_protos: Optional[torch.Tensor],
    ) -> torch.Tensor:
        self.last_dualtrack_eval_info = self._default_dualtrack_eval_info()
        if baseline_t_protos is None:
            return logits_aligned
        logits_raw = scale * torch.matmul(z_embed, baseline_t_protos.T) + self.b_c
        n_use = min(int(logits_aligned.shape[1]), int(logits_raw.shape[1]))
        if n_use <= 0:
            return logits_aligned
        conf_aligned = self._dual_track_confidence(logits_aligned[:, :n_use])
        conf_raw = self._dual_track_confidence(logits_raw[:, :n_use])
        conf_margin = max(0.0, float(getattr(self.config, "CAPAV1_DUALTRACK_CONF_MARGIN", 0.0)))
        choose_aligned = conf_aligned >= (conf_raw + conf_margin)
        pred_aligned = logits_aligned[:, :n_use].argmax(dim=1)
        pred_raw = logits_raw[:, :n_use].argmax(dim=1)
        same_pred = pred_aligned == pred_raw
        choose_aligned = choose_aligned | (same_pred & (conf_aligned >= conf_raw))
        abstain_mask = torch.zeros_like(choose_aligned, dtype=torch.bool)
        if bool(getattr(self.config, "CAPAV1_DUALTRACK_ENABLE_ABSTAIN", False)):
            abstain_thr = min(0.99, max(0.0, float(getattr(self.config, "CAPAV1_DUALTRACK_ABSTAIN_CONF", 0.60))))
            abstain_mask = torch.maximum(conf_aligned, conf_raw) < abstain_thr
            choose_aligned = choose_aligned & (~abstain_mask)
        blend = min(1.0, max(0.0, float(getattr(self.config, "CAPAV1_DUALTRACK_BLEND", 1.0))))
        out = logits_raw.clone()
        if blend < 1.0:
            blended_logits = blend * logits_aligned[:, :n_use] + (1.0 - blend) * logits_raw[:, :n_use]
            out[choose_aligned, :n_use] = blended_logits[choose_aligned]
        else:
            out[choose_aligned, :n_use] = logits_aligned[choose_aligned, :n_use]
        if int(logits_aligned.shape[1]) > n_use:
            out_full = logits_aligned.clone()
            if blend < 1.0:
                out_full[:, :n_use] = blend * logits_aligned[:, :n_use] + (1.0 - blend) * logits_raw[:, :n_use]
            out_full[~choose_aligned, :n_use] = logits_raw[~choose_aligned, :n_use]
            out = out_full
        self.last_dualtrack_eval_info = {
            "enabled": True,
            "aligned_rate": float(choose_aligned.float().mean().item()),
            "agree_rate": float(same_pred.float().mean().item()),
            "abstain_rate": float(abstain_mask.float().mean().item()),
            "conf_aligned_mean": float(conf_aligned.mean().item()),
            "conf_raw_mean": float(conf_raw.mean().item()),
        }
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
        if self._uses_dual_track_inference():
            return self._select_dual_track_logits(
                logits,
                z_embed,
                scale=scale,
                baseline_t_protos=baseline_t_protos,
            )
        logits = self._fuse_with_baseline_logits(
            logits,
            z_embed,
            scale=scale,
            baseline_t_protos=baseline_t_protos,
        )
        return logits

    def _mode_uses_prompt_bank_readout(self) -> bool:
        return (
            self._current_eval_mode() == "full_capa"
            and bool(getattr(self.config, "ENABLE_PROMPT_BANK_READOUT", True))
            and (not bool(getattr(self.config, "DISABLE_SHARED_CENTERING", False)))
        )

    def _encode_prompt_bank_readout_groups(self) -> List[torch.Tensor]:
        classes = list(self.config.ORDERED_CLASS_NAMES)
        templates = list(getattr(self.config, "PROMPT_BANK_READOUT_TEMPLATES", []))
        groups: List[torch.Tensor] = []
        for cls_name in classes:
            prompts = [str(t).replace("{finding}", cls_name.lower()) for t in templates]
            raw = self._encode_text(prompts)
            proc = self._apply_preprocessing(raw, self.zT_mean) if self._mode_uses_image_preprocessing() else raw
            proc = self._l2_norm(proc)
            if isinstance(self.R_frozen, torch.Tensor):
                proc = self._l2_norm(torch.matmul(proc, self.R_frozen.T))
            groups.append(proc)
        return groups

    def _residual_local_alpha_by_class(self) -> Dict[str, float]:
        if not self.residual_local_slerp_info:
            aligned = self._refresh_aligned_text()
            if isinstance(aligned, torch.Tensor):
                self._build_residual_local_text(aligned)
        return {str(row.get("class")): float(row.get("alpha", 0.0)) for row in self.residual_local_slerp_info}

    def _compose_prompt_bank_readout_logits(self, z_embed: torch.Tensor) -> torch.Tensor:
        groups = self._encode_prompt_bank_readout_groups()
        alpha_by_class = self._residual_local_alpha_by_class()
        mu = self._l2_norm(self.image_centroids) if isinstance(self.image_centroids, torch.Tensor) else None
        tau = max(1e-4, float(getattr(self.config, "PROMPT_BANK_READOUT_TEMP", 0.07)))
        logits: List[torch.Tensor] = []
        for c_idx, group in enumerate(groups):
            cls_name = self.config.ORDERED_CLASS_NAMES[c_idx] if c_idx < len(self.config.ORDERED_CLASS_NAMES) else str(c_idx)
            alpha = float(alpha_by_class.get(str(cls_name), 0.0))
            if alpha > 0.0 and isinstance(mu, torch.Tensor):
                group = torch.stack([self._slerp_unit(vec, mu[c_idx], alpha) for vec in group], dim=0)
            sims = torch.matmul(z_embed, group.T)
            logits.append(tau * torch.logsumexp(sims / tau, dim=1))
        return torch.stack(logits, dim=1)

    def _compute_metrics_from_logits(
        self,
        logits: torch.Tensor,
        y_true,
        is_multi: bool,
        *,
        calib_T: float,
        dataset_name: Optional[str],
        scoring_mode: Optional[str],
    ) -> Dict[str, float]:
        scores_rank = self._predict_probs(
            logits,
            calib_T=1.0,
            is_multi=bool(is_multi),
            dataset_name=dataset_name,
            scoring_mode=scoring_mode,
            ranking=True,
        ).detach().cpu().numpy()
        probs_cal = self._predict_probs(
            logits,
            calib_T=float(calib_T),
            is_multi=bool(is_multi),
            dataset_name=dataset_name,
            scoring_mode=scoring_mode,
            ranking=False,
        ).detach().cpu().numpy()
        return self._compute_metrics_from_prob_arrays(y_true, scores_rank, probs_cal, is_multi=bool(is_multi))

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
            if ranking:
                return logits / T
            if mode == "mixed":
                return torch.sigmoid(logits / T)
            return F.softmax(logits / T, dim=1)

        pos_indices = self._get_binary_positive_indices(dataset_name, logits.shape[1])
        if mode == "mixed":
            bin_logit = self._binary_logit_from_multiclass(logits / T, pos_indices)
            if ranking:
                return bin_logit
            return torch.sigmoid(bin_logit)

        probs_full = F.softmax(logits / T, dim=1)
        return probs_full[:, pos_indices].sum(dim=1).clamp(0.0, 1.0)

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

    def _select_projection_tau(self, T_sub: torch.Tensor, z_img: torch.Tensor) -> float:
        tau_floor = self._go_ml_tau(T_sub)
        if (not self._is_professor_profile()) or (not bool(getattr(self.config, "PROF_DYNAMIC_TAU", False))):
            return float(tau_floor)

        K = int(T_sub.shape[0])
        if K <= 1:
            return float(tau_floor)

        G = torch.matmul(T_sub, T_sub.T)
        rhs = torch.matmul(T_sub, z_img)
        try:
            eigvals = torch.linalg.eigvalsh(G).real.clamp_min(1e-12)
        except RuntimeError:
            return float(tau_floor)

        tau_hi = max(float(tau_floor) * 100.0, 1.0)
        candidates = np.unique(
            np.concatenate(
                [
                    np.geomspace(max(float(tau_floor), 1e-6), tau_hi, num=10),
                    np.asarray([float(tau_floor)], dtype=np.float64),
                ]
            )
        )
        stats = []
        D = max(1, int(T_sub.shape[1]))
        eye = torch.eye(K, device=self.device, dtype=T_sub.dtype)
        for tau in candidates:
            tau = float(max(tau, 1e-8))
            try:
                alpha = torch.linalg.solve(G + tau * eye, rhs)
            except RuntimeError:
                continue
            recon = torch.matmul(alpha, T_sub)
            resid = float(torch.norm(z_img - recon).item())
            sol = float(torch.norm(alpha).item())
            trace_h = float(torch.sum(eigvals / (eigvals + tau)).item())
            gcv = (resid ** 2) / max((D - trace_h) ** 2, 1e-8)
            stats.append((tau, max(resid, 1e-8), max(sol, 1e-8), max(gcv, 1e-8)))

        if len(stats) < 3:
            return float(tau_floor)

        taus = np.asarray([x[0] for x in stats], dtype=np.float64)
        log_r = np.log(np.asarray([x[1] for x in stats], dtype=np.float64))
        log_s = np.log(np.asarray([x[2] for x in stats], dtype=np.float64))
        log_g = np.log(np.asarray([x[3] for x in stats], dtype=np.float64))

        curvature = np.zeros_like(log_g)
        for i in range(1, len(taus) - 1):
            x1, y1 = log_r[i] - log_r[i - 1], log_s[i] - log_s[i - 1]
            x2, y2 = log_r[i + 1] - log_r[i], log_s[i + 1] - log_s[i]
            denom = max((x1 * x1 + y1 * y1) ** 1.5, 1e-8)
            curvature[i] = abs(x1 * y2 - y1 * x2) / denom

        gcv_norm = (log_g - np.mean(log_g)) / (np.std(log_g) + 1e-8)
        curv_norm = (curvature - np.mean(curvature)) / (np.std(curvature) + 1e-8)
        score = gcv_norm - 0.25 * curv_norm
        best_idx = int(np.argmin(score))
        return float(max(taus[best_idx], tau_floor))

    def _select_residual_confounders(self, active_indices, target_c_idx, t_aligned):
        other_indices = [idx for idx in active_indices if idx != target_c_idx]
        if not other_indices:
            return [], torch.empty(0, device=self.device), torch.empty(0, device=self.device)

        target_vec = self._l2_norm(t_aligned[target_c_idx].view(1, -1)).view(-1)
        T_other = self._l2_norm(t_aligned[other_indices])
        sims = torch.clamp(torch.matmul(T_other, target_vec), min=-1.0, max=1.0)
        mode = str(getattr(self.config, "GO_ML_CONFOUNDER_MODE", "full")).strip().lower()
        if mode == "topm" and int(sims.numel()) > int(getattr(self.config, "GO_ML_TOPM", 1)):
            m = max(1, min(int(getattr(self.config, "GO_ML_TOPM", 1)), int(sims.numel())))
            top_idx = torch.topk(sims, k=m, largest=True).indices
            sel_indices = [other_indices[int(i)] for i in top_idx.detach().cpu().tolist()]
            sel_sims = sims.index_select(0, top_idx)
            weights = torch.ones_like(sel_sims)
            return sel_indices, sel_sims, weights
        if mode == "sim_weighted" and int(sims.numel()) > 0:
            temp = max(float(getattr(self.config, "GO_ML_SIM_WEIGHT_TEMP", 0.20)), 1e-3)
            logits = torch.clamp(sims, min=0.0) / temp
            if float(torch.max(logits).item()) <= 0.0:
                weights = torch.ones_like(sims)
            else:
                weights = torch.softmax(logits, dim=0) * float(len(other_indices))
            return other_indices, sims, weights
        weights = torch.ones_like(sims)
        return other_indices, sims, weights

    def _compute_adaptive_residual_lambda(self, *, resid_norm_ratio: float, active_count: int, max_other_sim: float) -> float:
        if active_count <= 1:
            return 0.0
        resid_ratio = min(1.0, max(0.0, float(resid_norm_ratio)))
        sim_term = min(1.0, max(0.0, float(max_other_sim)))
        card_term = min(1.0, max(0.0, float(active_count - 1) / 2.0))
        if resid_ratio < float(getattr(self.config, "GO_ML_ADAPTIVE_MIN_RESID_RATIO", 0.15)) and sim_term < 0.25:
            return 0.0
        lam = 0.15 + 0.35 * card_term + 0.30 * sim_term + 0.20 * resid_ratio
        return min(1.0, max(0.0, float(lam)))

    def _should_use_conditional_huber(
        self,
        *,
        active_count: int,
        resid_meta: Dict[str, float],
        step: Optional[int] = None,
    ) -> bool:
        n_other = int(resid_meta.get("n_other", max(0, int(active_count) - 1)))
        if n_other <= 0 or int(active_count) <= 1:
            return False
        scope = str(getattr(self.config, "GO_ML_HUBER_SCOPE", "always")).strip().lower()
        if scope == "always":
            return True
        if scope == "warmup":
            return (step is not None) and (int(step) < int(getattr(self.config, "GO_ML_HUBER_WARMUP_STEPS", 0)))
        return (
            int(active_count) >= int(getattr(self.config, "GO_ML_HUBER_COND_MIN_ACTIVE", 3))
            or float(resid_meta.get("cond_reg", np.nan)) >= float(getattr(self.config, "GO_ML_HUBER_COND_MIN_COND", 25.0))
            or float(resid_meta.get("resid_norm_ratio", 0.0)) >= float(getattr(self.config, "GO_ML_HUBER_COND_MIN_RESID_RATIO", 0.90))
            or float(resid_meta.get("max_other_sim", 0.0)) >= float(getattr(self.config, "GO_ML_HUBER_COND_MIN_OTHER_SIM", 0.35))
        )

    def _compute_robust_residual_weight(
        self,
        signal_vec: torch.Tensor,
        current_mu: torch.Tensor,
        anchor: torch.Tensor,
        *,
        active_count: int,
        resid_meta: Dict[str, float],
        step: Optional[int] = None,
    ) -> float:
        mode = str(getattr(self.config, "GO_ML_ROBUST_MODE", "none")).strip().lower()
        if mode == "none":
            return 1.0
        if not self._should_use_conditional_huber(active_count=active_count, resid_meta=resid_meta, step=step):
            return 1.0
        ref_vec = self._l2_norm((0.5 * current_mu + 0.5 * anchor).view(1, -1)).view(-1)
        signal = self._l2_norm(signal_vec.view(1, -1)).view(-1)
        cos_val = float(torch.clamp(torch.dot(signal, ref_vec), min=-1.0, max=1.0).item())
        dist = max(0.0, 1.0 - cos_val)
        delta = max(1e-4, float(getattr(self.config, "GO_ML_HUBER_DELTA", 0.20)))
        if dist <= delta:
            return 1.0
        return float(delta / max(dist, 1e-8))

    def _compute_multilabel_residual(self, z_img, active_indices, target_c_idx, t_aligned):
        selected_indices, sims, weights = self._select_residual_confounders(active_indices, target_c_idx, t_aligned)
        if not selected_indices:
            meta = {
                "n_other": 0,
                "n_selected": 0,
                "max_other_sim": 0.0,
                "mean_other_sim": 0.0,
                "resid_norm_ratio": 1.0,
            }
            return z_img, 1.0, meta

        T_sub = t_aligned[selected_indices]
        if int(weights.numel()) == int(T_sub.shape[0]):
            T_proj = weights.view(-1, 1) * T_sub
        else:
            T_proj = T_sub
            weights = torch.ones(int(T_sub.shape[0]), device=self.device, dtype=T_sub.dtype)
        K = int(T_proj.shape[0])
        G = torch.matmul(T_proj, T_proj.T)
        use_go_ml = bool(getattr(self.config, "ENABLE_GO_MULTILABEL_PROJECTION", False))
        tau = self._select_projection_tau(T_proj, z_img) if use_go_ml else float(self.config.MULTI_LABEL_RIDGE)
        try:
            eigvals = torch.linalg.eigvalsh(G).real
            lam_max = float(torch.max(eigvals).item())
            lam_min = float(torch.min(eigvals).item())
            cond_reg = float((lam_max + tau) / max(lam_min + tau, 1e-8))
        except RuntimeError:
            cond_reg = np.nan
        Reg = tau * torch.eye(K, device=self.device, dtype=T_proj.dtype)
        rhs = torch.matmul(T_proj, z_img)
        try:
            alpha = torch.linalg.solve(G + Reg, rhs)
        except RuntimeError:
            meta = {
                "n_other": int(len(active_indices) - 1),
                "n_selected": int(len(selected_indices)),
                "max_other_sim": float(torch.max(sims).item()) if int(sims.numel()) > 0 else 0.0,
                "mean_other_sim": float(torch.mean(sims).item()) if int(sims.numel()) > 0 else 0.0,
                "resid_norm_ratio": 1.0,
                "cond_reg": cond_reg,
            }
            return z_img, 1.0, meta

        resid_raw = z_img - torch.matmul(alpha, T_proj)
        resid_norm = float(torch.norm(resid_raw).item())
        z_norm = max(float(torch.norm(z_img).item()), 1e-8)
        resid_ratio = float(min(1.0, max(0.0, resid_norm / z_norm)))
        meta = {
            "n_other": int(len(active_indices) - 1),
            "n_selected": int(len(selected_indices)),
            "max_other_sim": float(torch.max(sims).item()) if int(sims.numel()) > 0 else 0.0,
            "mean_other_sim": float(torch.mean(sims).item()) if int(sims.numel()) > 0 else 0.0,
            "resid_norm_ratio": resid_ratio,
            "cond_reg": cond_reg,
        }
        if not np.isfinite(resid_norm) or resid_norm <= 1e-8:
            return z_img, 1.0, meta
        return self._l2_norm(resid_raw), resid_norm, meta

    def _labels_to_active_mask(self, y_batch, n_cls: int) -> torch.Tensor:
        y_arr = np.asarray(y_batch)
        if y_arr.ndim == 1:
            out = torch.zeros((len(y_arr), n_cls), dtype=torch.bool, device=self.device)
            pos = np.where(y_arr > 0)[0]
            if n_cls > 0 and len(pos) > 0:
                out[pos, 0] = True
            return out
        n_use = min(int(y_arr.shape[1]), int(n_cls))
        out = torch.zeros((int(y_arr.shape[0]), n_cls), dtype=torch.bool, device=self.device)
        if n_use > 0:
            out[:, :n_use] = torch.tensor(y_arr[:, :n_use] > 0, dtype=torch.bool, device=self.device)
        return out

    def _ensure_disc_axis_negative_state(self, n_cls: int, dtype: torch.dtype, t_aligned: torch.Tensor) -> None:
        if (
            not isinstance(self.disc_neg_centroids, torch.Tensor)
            or int(self.disc_neg_centroids.shape[0]) != int(n_cls)
        ):
            self.disc_neg_centroids = t_aligned[:n_cls].detach().clone().to(self.device)
            self.disc_neg_counts = torch.zeros(n_cls, device=self.device, dtype=dtype)

    def _update_disc_axis_negative_centroids(self, z_batch: torch.Tensor, active_mask: torch.Tensor, t_aligned: torch.Tensor) -> None:
        if not bool(getattr(self.config, "ENABLE_DISC_AXIS_PROCRUSTES", True)):
            return
        n_cls = int(t_aligned.shape[0])
        self._ensure_disc_axis_negative_state(n_cls, z_batch.dtype, t_aligned)
        for i in range(int(z_batch.shape[0])):
            neg_indices = torch.where(~active_mask[i])[0].tolist()
            img_vec = z_batch[i]
            for c_idx in neg_indices:
                self.disc_neg_counts[c_idx] += 1.0
                eta = 1.0 / (float(self.config.KAPPA_EMA) + float(self.disc_neg_counts[c_idx].item()))
                cur = self.disc_neg_centroids[c_idx]
                self.disc_neg_centroids[c_idx] = self._l2_norm((1.0 - eta) * cur + eta * img_vec)

    def _update_centroids_gt_support(
        self,
        z_batch: torch.Tensor,
        gt_labels,
        t_aligned: torch.Tensor,
        *,
        step: Optional[int] = None,
    ) -> Tuple[int, int, int]:
        bsz = int(z_batch.shape[0])
        n_cls = int(t_aligned.shape[0])
        if self.prior_counts is None or int(self.prior_counts.shape[0]) != n_cls:
            self.prior_counts = torch.zeros(n_cls, device=self.device, dtype=z_batch.dtype)
        if self.rejected_counts is None or int(self.rejected_counts.shape[0]) != n_cls:
            self.rejected_counts = torch.zeros(n_cls, device=self.device, dtype=z_batch.dtype)
        if self.support_counts is None or int(self.support_counts.shape[0]) != n_cls:
            self.support_counts = torch.zeros(n_cls, device=self.device, dtype=z_batch.dtype)
        if gt_labels is None:
            raise RuntimeError("GT-only mainline requires labels for support updates.")

        gt_active_mask = self._labels_to_active_mask(gt_labels, n_cls=n_cls)

        update_count = 0
        support_labels = 0

        for i in range(bsz):
            active_indices = torch.where(gt_active_mask[i])[0].tolist()
            if not active_indices:
                continue

            img_vec = z_batch[i]

            for c_idx in active_indices:
                self.prior_counts[c_idx] += 1.0
                support_labels += 1
                self.support_counts[c_idx] += 1.0
                z_resid, resid_norm, resid_meta = self._compute_multilabel_residual(img_vec, active_indices, c_idx, t_aligned)
                eta = 1.0 / (float(self.config.KAPPA_EMA) + float(self.support_counts[c_idx].item()))
                evidence_weight = 1.0
                if bool(getattr(self.config, "ENABLE_GO_MULTILABEL_PROJECTION", False)) and bool(
                    getattr(self.config, "GO_ML_USE_RESIDUAL_NORM_WEIGHT", True)
                ):
                    evidence_weight *= max(1e-4, float(resid_norm))
                current_mu = self.image_centroids[c_idx]
                anchor = t_aligned[c_idx]
                signal_mode = str(getattr(self.config, "GO_ML_SIGNAL_MODE", "residual")).strip().lower()
                signal_vec = img_vec
                residual_lambda = 0.0
                if bool(getattr(self.config, "ENABLE_GO_MULTILABEL_PROJECTION", False)):
                    if signal_mode == "residual":
                        signal_vec = z_resid
                        residual_lambda = 1.0
                    elif signal_mode == "adaptive":
                        residual_lambda = self._compute_adaptive_residual_lambda(
                            resid_norm_ratio=float(resid_meta.get("resid_norm_ratio", 0.0)),
                            active_count=len(active_indices),
                            max_other_sim=float(resid_meta.get("max_other_sim", 0.0)),
                        )
                        if residual_lambda >= 1.0 - 1e-8:
                            signal_vec = z_resid
                        elif residual_lambda <= 1e-8:
                            signal_vec = img_vec
                        else:
                            signal_vec = self._l2_norm(
                                ((1.0 - residual_lambda) * img_vec + residual_lambda * z_resid).view(1, -1)
                            ).view(-1)
                robust_scale = self._compute_robust_residual_weight(
                    signal_vec,
                    current_mu,
                    anchor,
                    active_count=len(active_indices),
                    resid_meta=resid_meta,
                    step=step,
                )
                evidence_weight *= max(1e-4, float(robust_scale))
                kappa0 = float(self.config.KAPPA0)
                target = (kappa0 * anchor + evidence_weight * signal_vec) / (kappa0 + evidence_weight + 1e-8)
                self.image_centroids[c_idx] = self._l2_norm((1.0 - eta) * current_mu + eta * target)
                update_count += 1

        rejected_labels = 0
        self._update_disc_axis_negative_centroids(z_batch, gt_active_mask, t_aligned)
        return update_count, rejected_labels, support_labels

    def _solve_procrustes(self):
        N = self.support_counts
        mask = self._get_alignment_active_mask()
        if mask.sum() == 0:
            return self.current_R

        t_ref = self._get_alignment_text_base()
        w = self._get_alignment_class_weights(mask)
        mu = self._l2_norm(self.image_centroids)
        t_raw = self._l2_norm(t_ref)
        if bool(getattr(self.config, "ENABLE_DISC_AXIS_PROCRUSTES", True)):
            if isinstance(self.disc_neg_centroids, torch.Tensor) and self.disc_neg_centroids.shape == mu.shape:
                lam = max(0.0, float(getattr(self.config, "DISC_AXIS_NEG_LAMBDA", 0.25)))
                mu_neg = self._l2_norm(self.disc_neg_centroids)
                mu_axis = self._l2_norm(mu - lam * mu_neg)
                m_cov = torch.matmul((w.view(-1, 1) * mu_axis).T, t_raw)
                u, _, vh = torch.linalg.svd(m_cov)
                diag = torch.ones(m_cov.shape[1], device=self.device)
                if torch.det(torch.matmul(u, vh)) < 0:
                    diag[-1] = -1
                return torch.matmul(torch.matmul(u, torch.diag(diag)), vh)

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

    # === [澧炲己鐗圿 璇婃柇鍑芥暟 ===
    def _diagnose_and_log_dynamic(self, R_cand, step, phase="Adapt", emit_log: Optional[bool] = None):
        mask = self._get_alignment_active_mask()
        n_act = int(mask.sum().item())
        if n_act == 0:
            return {"passed": False, "dS": 0.0, "n_act": 0}

        T = self._get_alignment_text_base()
        Mu = self.image_centroids
        T_rot = torch.matmul(T, R_cand.T)
        I = torch.eye(R_cand.shape[0], device=self.device)
        ortho_err = torch.norm(torch.matmul(R_cand, R_cand.T) - I).item()

        Sim_post = torch.matmul(Mu, T_rot.T)
        Sim_pre = torch.matmul(Mu, T.T)
        sim_before_vec = Sim_pre.diag()[mask]
        sim_before_post_vec = Sim_post.diag()[mask]
        sim_before = float(sim_before_vec.mean().item())
        valid_sim = float(sim_before_post_vec.mean().item())
        diag_gain_vec = sim_before_post_vec - sim_before_vec
        min_diag_gain = float(diag_gain_vec.min().item())
        neg_diag_gain_mass = float(torch.clamp(-diag_gain_vec, min=0.0).sum().item())
        neg_diag_gain_count = int((diag_gain_vec < 0).sum().item())

        idx = torch.where(mask)[0]
        if int(idx.numel()) > 1:
            Sim_pre_act = Sim_pre.index_select(0, idx).index_select(1, idx)
            Sim_post_act = Sim_post.index_select(0, idx).index_select(1, idx)
            off_diag_mask = ~torch.eye(int(idx.numel()), dtype=torch.bool, device=self.device)
            off_diag_pre_mean = float(Sim_pre_act[off_diag_mask].mean().item())
            off_diag_post_mean = float(Sim_post_act[off_diag_mask].mean().item())
            off_diag_delta = float(off_diag_post_mean - off_diag_pre_mean)
        else:
            Sim_pre_act = None
            off_diag_pre_mean = 0.0
            off_diag_post_mean = 0.0
            off_diag_delta = 0.0

        delta = float(valid_sim - sim_before)
        if bool(getattr(self.config, "GATE_USE_RHO_QUANTILE", False)) and int(sim_before_vec.numel()) > 0:
            q = min(1.0, max(0.0, float(getattr(self.config, "RHO_QUANTILE", 0.70))))
            rho_eff = float(torch.quantile(sim_before_vec.detach(), q).item())
        else:
            rho_eff = self._effective_rho()
        if not np.isfinite(rho_eff):
            rho_eff = self._effective_rho()

        offdiag_limit = self._effective_gate_max_offdiag_delta()
        gain_frac = max(0.0, float(getattr(self.config, "CAPAV1_DYNAMIC_OFFDIAG_FRAC", 0.30)))
        adaptive_limit = gain_frac * max(delta, 0.0)
        if offdiag_limit > 0.0:
            offdiag_limit = max(0.0, min(offdiag_limit, adaptive_limit) if adaptive_limit > 0.0 else offdiag_limit)
        else:
            offdiag_limit = max(0.0, adaptive_limit)
        if bool(self.eval_runtime.get("offdiag_gate", False)):
            offdiag_ok = off_diag_delta <= offdiag_limit
        else:
            offdiag_ok = True

        eps_eff = float(self.config.EPSILON)
        if bool(getattr(self.config, "PROF_DYNAMIC_EPSILON", False)) and Sim_pre_act is not None and int(idx.numel()) > 1:
            hardest_nonmatch = Sim_pre_act.masked_fill(
                torch.eye(int(idx.numel()), dtype=torch.bool, device=self.device),
                -1e9,
            ).max(dim=1).values
            mean_gap = float((sim_before_vec - hardest_nonmatch).mean().item())
            eps_eff = max(1e-6, float(getattr(self.config, "PROF_EPSILON_GAP_PCT", 0.015)) * max(mean_gap, 1e-6))

        rho_ok = valid_sim <= rho_eff
        gain_to_offdiag = float("inf") if off_diag_delta <= 1e-8 else float(delta / max(off_diag_delta, 1e-8))
        if not rho_ok:
            active_margin = max(0, int(getattr(self.config, "CAPAV1_RHO_BYPASS_ACTIVE_MARGIN", 2)))
            relaxed_ratio = max(1.0, float(getattr(self.config, "CAPAV1_RELAXED_GAIN_OFFDIAG_RATIO", 2.5)))
            relaxed_min_active = min(len(mask), int(self.config.MIN_CLASSES_FOR_ADAPTATION) + active_margin)
            rho_ok = (gain_to_offdiag >= relaxed_ratio) and (n_act >= int(relaxed_min_active))

        passed = (delta >= eps_eff) and rho_ok and offdiag_ok

        if emit_log is None:
            emit_log = bool(self.config.VERBOSE)
        if emit_log:
            tqdm.write(
                f" Diagnostic Step {step} ({phase}) | Active {n_act}/{len(mask)} "
                f"| dS={delta:+.4f} | eps*={eps_eff:.4f} | dOffDiag={off_diag_delta:+.4f} "
                f"| dOff*={offdiag_limit:.4f} | rho*={rho_eff:.4f} | OrthoErr={ortho_err:.2e} | Passed={passed}"
            )

        return {
            "passed": passed,
            "dS": delta,
            "s_after": valid_sim,
            "n_act": n_act,
            "dOffDiag": off_diag_delta,
            "dOffDiagLimit": offdiag_limit,
            "gain_to_offdiag": gain_to_offdiag,
            "min_diag_gain": min_diag_gain,
            "neg_diag_gain_mass": neg_diag_gain_mass,
            "neg_diag_gain_count": neg_diag_gain_count,
            "rho_eff": rho_eff,
            "eps_eff": eps_eff,
            "off_diag_pre_mean": off_diag_pre_mean,
            "off_diag_post_mean": off_diag_post_mean,
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
        N = self.support_counts
        mask = self._get_alignment_active_mask()
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
        aligned_text = self._refresh_aligned_text()
        state_dict = {
            "line_mode": "capav1_gt",
            "centroids": self.image_centroids.cpu(),
            "centroids_last_good": self.centroids_last_good.cpu() if isinstance(self.centroids_last_good, torch.Tensor) else None,
            "R": self.R_frozen.cpu(),
            "R_last_good": self.R_last_good.cpu() if isinstance(self.R_last_good, torch.Tensor) else None,
            "support_counts": self.support_counts.cpu(),
            "rejected_counts": self.rejected_counts.cpu() if isinstance(self.rejected_counts, torch.Tensor) else None,
            "prior_counts": self.prior_counts.cpu() if isinstance(self.prior_counts, torch.Tensor) else None,
            "alignment_stats": self.final_alignment_stats,
            "max_leverage": self.max_leverage_info,
            "t_align_base": T_base.detach().cpu(),
            "t_mixed": T_base.detach().cpu(),
            "t_raw_pooled": self.t_raw_pooled.detach().cpu(),
            "t_raw_text": self.t_raw_text.detach().cpu() if isinstance(self.t_raw_text, torch.Tensor) else None,
            "t_processed_text": self.t_processed_text.detach().cpu() if isinstance(self.t_processed_text, torch.Tensor) else None,
            "t_aligned_text": aligned_text.detach().cpu() if isinstance(aligned_text, torch.Tensor) else None,
            "guardian": guardian_blob,
        }
        with open(state_path, "wb") as f:
            pickle.dump(state_dict, f)
        self._log(f"[Saved] {state_path}")

    def _orthogonalize_rotation(self, M: torch.Tensor) -> torch.Tensor:
        U, _, Vh = torch.linalg.svd(M)
        diag = torch.ones(M.shape[0], device=M.device, dtype=M.dtype)
        if torch.det(torch.matmul(U, Vh)) < 0:
            diag[-1] = -1
        return torch.matmul(torch.matmul(U, torch.diag(diag)), Vh)

    def _alignment_stats_score(self, stats: Dict[str, float]) -> float:
        dS = float(stats.get("dS", 0.0))
        dOff = max(0.0, float(stats.get("dOffDiag", 0.0)))
        dOff_lim = max(0.0, float(stats.get("dOffDiagLimit", 0.0)))
        off_violation = max(0.0, dOff - dOff_lim)
        min_diag_gain = float(stats.get("min_diag_gain", 0.0))
        neg_gain_mass = max(0.0, float(stats.get("neg_diag_gain_mass", 0.0)))
        rho_gap = max(0.0, float(stats.get("s_after", 0.0)) - float(stats.get("rho_eff", self.config.RHO)))
        eps_gap = max(0.0, float(stats.get("eps_eff", self.config.EPSILON)) - dS)
        ratio = float(stats.get("gain_to_offdiag", 0.0))
        if not np.isfinite(ratio):
            ratio = 5.0
        ratio_bonus = min(max(ratio, 0.0), 5.0)
        n_act_bonus = 0.01 * min(int(stats.get("n_act", 0)), len(self.config.ORDERED_CLASS_NAMES))
        pass_bonus = 0.05 if bool(stats.get("passed", False)) else 0.0
        min_gain_penalty = max(0.0, -min_diag_gain)
        return (
            dS
            - 0.75 * dOff
            - 1.25 * off_violation
            - 0.80 * neg_gain_mass
            - 0.60 * min_gain_penalty
            - 0.50 * rho_gap
            - 0.50 * eps_gap
            + 0.01 * ratio_bonus
            + n_act_bonus
            + pass_bonus
        )

    def _is_soft_fallback_acceptable(self, stats: Dict[str, float]) -> bool:
        if int(stats.get("n_act", 0)) < int(self.config.MIN_CLASSES_FOR_ADAPTATION):
            return False
        min_gain = max(float(getattr(self.config, "CAPAV1_SOFT_FALLBACK_MIN_GAIN", 0.03)), 5.0 * float(self.config.EPSILON))
        dS = float(stats.get("dS", 0.0))
        dOff = max(0.0, float(stats.get("dOffDiag", 0.0)))
        dOff_lim = max(0.0, float(stats.get("dOffDiagLimit", 0.0)))
        offdiag_mult = max(1.0, float(getattr(self.config, "CAPAV1_SOFT_FALLBACK_OFFDIAG_MULT", 1.50)))
        max_soft_offdiag = max(0.03, dOff_lim * offdiag_mult, dOff_lim + 0.02)
        ratio = float(stats.get("gain_to_offdiag", 0.0))
        if not np.isfinite(ratio):
            ratio = 5.0
        return (dS >= min_gain) and (dOff <= max_soft_offdiag) and (ratio >= 1.0)

    def _select_guarded_alignment_candidate(
        self,
        R_full: torch.Tensor,
        step: int,
        *,
        phase: str = "Adapt",
        allow_soft_fallback: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        full_stats = self._diagnose_and_log_dynamic(R_full, step, phase=phase, emit_log=False)
        full_stats["alpha"] = 1.0
        full_stats["selection_mode"] = "full"
        full_stats["mix_base"] = "full"
        full_stats["score"] = self._alignment_stats_score(full_stats)
        best_hard = (full_stats["score"], R_full, dict(full_stats)) if bool(full_stats.get("passed", False)) else None
        hard_candidates: List[Tuple[float, torch.Tensor, Dict[str, float]]] = []
        if best_hard is not None:
            hard_candidates.append((full_stats["score"], R_full, dict(full_stats)))
        best_soft = None
        if allow_soft_fallback and self._is_soft_fallback_acceptable(full_stats):
            best_soft = (full_stats["score"], R_full, dict(full_stats))
        candidate_dump: List[Dict[str, float]] = [dict(full_stats)]

        slerp_full = self._build_slerp_rotation_candidate(R_full, 1.0)
        if slerp_full is not None:
            R_slerp_full, lam_full = slerp_full
            s_stats = self._diagnose_and_log_dynamic(R_slerp_full, step, phase=phase, emit_log=False)
            s_stats["alpha"] = 1.0
            s_stats["mix_base"] = "slerp"
            s_stats["selection_mode"] = "guarded_slerp"
            s_stats["slerp_lambda"] = float(lam_full)
            s_stats["score"] = self._alignment_stats_score(s_stats)
            candidate_dump.append(dict(s_stats))
            if bool(s_stats.get("passed", False)):
                hard_candidates.append((s_stats["score"], R_slerp_full, dict(s_stats)))
                if (best_hard is None) or (s_stats["score"] > best_hard[0]):
                    best_hard = (s_stats["score"], R_slerp_full, dict(s_stats))
            if allow_soft_fallback and self._is_soft_fallback_acceptable(s_stats):
                if (best_soft is None) or (s_stats["score"] > best_soft[0]):
                    best_soft = (s_stats["score"], R_slerp_full, dict(s_stats))

        identity = torch.eye(R_full.shape[0], device=self.device, dtype=R_full.dtype)
        anchor = self.current_R if isinstance(self.current_R, torch.Tensor) and self.current_R.shape == R_full.shape else identity
        alpha_list = []
        alpha_source = list(getattr(self.config, "CAPAV1_GUARDED_ALPHAS", [1.0, 0.85, 0.70, 0.55, 0.40, 0.25]))
        for alpha in alpha_source:
            a = float(alpha)
            if a <= 0.0 or a > 1.0:
                continue
            if not any(abs(a - prev) < 1e-6 for prev in alpha_list):
                alpha_list.append(a)
        if not any(abs(1.0 - prev) < 1e-6 for prev in alpha_list):
            alpha_list.insert(0, 1.0)
        alpha_list = sorted(alpha_list, reverse=True)

        for alpha in alpha_list:
            if abs(alpha - 1.0) < 1e-6:
                continue
            mixed = (1.0 - alpha) * anchor + alpha * R_full
            R_try = self._orthogonalize_rotation(mixed)
            stats = self._diagnose_and_log_dynamic(R_try, step, phase=phase, emit_log=False)
            stats["alpha"] = alpha
            stats["mix_base"] = "anchor"
            stats["selection_mode"] = "guarded"
            stats["score"] = self._alignment_stats_score(stats)
            candidate_dump.append(dict(stats))
            if bool(stats.get("passed", False)):
                hard_candidates.append((stats["score"], R_try, dict(stats)))
                if (best_hard is None) or (stats["score"] > best_hard[0]):
                    best_hard = (stats["score"], R_try, dict(stats))
            if allow_soft_fallback and self._is_soft_fallback_acceptable(stats):
                if (best_soft is None) or (stats["score"] > best_soft[0]):
                    best_soft = (stats["score"], R_try, dict(stats))

            slerp_try = self._build_slerp_rotation_candidate(R_try, alpha)
            if slerp_try is not None:
                R_slerp, lam = slerp_try
                s_stats = self._diagnose_and_log_dynamic(R_slerp, step, phase=phase, emit_log=False)
                s_stats["alpha"] = alpha
                s_stats["mix_base"] = "slerp"
                s_stats["selection_mode"] = "guarded_slerp"
                s_stats["slerp_lambda"] = float(lam)
                s_stats["score"] = self._alignment_stats_score(s_stats)
                candidate_dump.append(dict(s_stats))
                if bool(s_stats.get("passed", False)):
                    hard_candidates.append((s_stats["score"], R_slerp, dict(s_stats)))
                    if (best_hard is None) or (s_stats["score"] > best_hard[0]):
                        best_hard = (s_stats["score"], R_slerp, dict(s_stats))
                if allow_soft_fallback and self._is_soft_fallback_acceptable(s_stats):
                    if (best_soft is None) or (s_stats["score"] > best_soft[0]):
                        best_soft = (s_stats["score"], R_slerp, dict(s_stats))

        if hard_candidates:
            tol = max(0.0, float(getattr(self.config, "CAPAV1_SMALLER_ALPHA_SCORE_TOL", 0.015)))
            min_gain_frac = min(1.0, max(0.0, float(getattr(self.config, "CAPAV1_SMALLER_ALPHA_MIN_GAIN_FRAC", 0.40))))
            best_hard_score = max(float(item[0]) for item in hard_candidates)
            max_hard_gain = max(float(item[2].get("dS", 0.0)) for item in hard_candidates)
            conservative_pool = [
                item
                for item in hard_candidates
                if float(item[0]) >= (best_hard_score - tol)
                and float(item[2].get("dS", 0.0)) >= (max_hard_gain * min_gain_frac)
            ]
            if conservative_pool:
                conservative_pool.sort(
                    key=lambda item: (
                        float(item[2].get("alpha", 1.0)),
                        -float(item[0]),
                    )
                )
                best_hard = conservative_pool[0]
                best_hard[2]["selection_mode"] = "guarded_conservative"

        chosen = best_hard
        if chosen is None and allow_soft_fallback and best_soft is not None:
            chosen = best_soft
            chosen[2]["soft_fallback"] = True
        if chosen is None:
            chosen = (full_stats["score"], R_full, dict(full_stats))

        _, R_sel, stats_sel = chosen
        stats_sel.setdefault("soft_fallback", False)
        if self.config.VERBOSE:
            if bool(getattr(self.config, "CAPAV1_GUARDED_DUMP", False)):
                for item in sorted(candidate_dump, key=lambda x: float(x.get("alpha", 1.0)), reverse=True):
                    tqdm.write(
                        " [GuardedCand] "
                        f"a={float(item.get('alpha', 1.0)):.2f} "
                        f"| base={str(item.get('mix_base', 'full'))} "
                        f"| pass={bool(item.get('passed', False))} "
                        f"| dS={float(item.get('dS', 0.0)):+.4f} "
                        f"| dOff={float(item.get('dOffDiag', 0.0)):+.4f} "
                        f"| minGain={float(item.get('min_diag_gain', 0.0)):+.4f} "
                        f"| negMass={float(item.get('neg_diag_gain_mass', 0.0)):.4f} "
                        f"| score={float(item.get('score', 0.0)):+.4f}"
                    )
            diag_stats = self._diagnose_and_log_dynamic(R_sel, step, phase=phase, emit_log=True)
            diag_stats["alpha"] = float(stats_sel.get("alpha", 1.0))
            diag_stats["mix_base"] = str(stats_sel.get("mix_base", "full"))
            diag_stats["selection_mode"] = str(stats_sel.get("selection_mode", "full"))
            diag_stats["score"] = float(stats_sel.get("score", self._alignment_stats_score(diag_stats)))
            diag_stats["soft_fallback"] = bool(stats_sel.get("soft_fallback", False))
            stats_sel = diag_stats
            tqdm.write(
                f" [GuardedR] alpha={float(stats_sel.get('alpha', 1.0)):.2f} "
                f"| base={stats_sel.get('mix_base', 'full')} "
                f"| mode={stats_sel.get('selection_mode', 'full')} "
                f"| score={float(stats_sel.get('score', 0.0)):+.4f} "
                f"| soft={bool(stats_sel.get('soft_fallback', False))}"
            )
        return R_sel, stats_sel

    def _commit_alignment_candidate(self, R_cand: torch.Tensor, stats: Dict[str, float], gate_pass: bool) -> None:
        self.current_R = R_cand
        self.R_frozen = R_cand
        self._snapshot_last_good_state()
        self._refresh_aligned_text()

        t_base_eval = self._get_alignment_text_base()
        T_rot = torch.matmul(t_base_eval, R_cand.T)
        Sim = torch.matmul(self.image_centroids, T_rot.T)
        Sim_pre = torch.matmul(self.image_centroids, t_base_eval.T)
        off_diag_mask = ~torch.eye(Sim.shape[0], dtype=torch.bool, device=self.device)
        self.final_alignment_stats = {
            "dS_gain": float(stats.get("dS", 0.0)),
            "n_act": int(stats.get("n_act", 0)),
            "off_diag_mean": Sim[off_diag_mask].mean().item(),
            "off_diag_max": Sim[off_diag_mask].max().item(),
            "off_diag_pre_mean": Sim_pre[off_diag_mask].mean().item(),
            "support_count": float(self.support_counts.sum().item()) if self.support_counts is not None else 0.0,
            "gate_pass": bool(gate_pass),
            "selection_alpha": float(stats.get("alpha", 1.0)),
            "selection_mix_base": str(stats.get("mix_base", "full")),
            "selection_mode": str(stats.get("selection_mode", "full")),
            "selection_score": float(stats.get("score", 0.0)),
            "selection_slerp_lambda": float(stats.get("slerp_lambda", 0.0)),
            "soft_fallback": bool(stats.get("soft_fallback", False)),
        }

    def run_pipeline(self, *, run_stage4: bool = True):
        self._ensure_early_text_prompt_support()
        z_cal = self._prepare_shared_feature_space()
        if not self._mode_uses_alignment():
            # Legacy deployment-only evaluation sweep kept dormant for the two-mode main path.
            if run_stage4 and self._mode_uses_guarded_alignment():
                self._run_evaluation()
                self._run_scale_sweep([8, 12, 16, 24, 32, 40])
            return

        d = self.t_raw_text.shape[1]

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
        self.guardian_stage2_ref_mean = np.nan
        self.guardian_stage2_stat = 0.0
        self.guardian_stage2_last = np.nan
        self.guardian_stage2_steps = 0
        saved_state = False
        best_soft_candidate_R = None
        best_soft_candidate_stats = None
        best_soft_candidate_score = -float("inf")

        self._log(f"[Stage III] GT Support Adaptation (warmup={self.config.WARMUP_BATCHES} batches)")
        z_train, y_train, _ = self._load_data(self.config.TRAIN_DATA_PATH, split_override=0)
        z_train = self._prepare_eval_embeddings(z_train)
        if y_train is None:
            raise RuntimeError("capav1_gt requires labels in TRAIN_DATA_PATH for GT evidence routing.")
        
        B = self.config.TRAIN_BATCH_SIZE
        
        pbar = tqdm(range(0, len(z_train), B), desc=" Init Warmup", disable=(not self.config.VERBOSE),
                   bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
        
        for i in pbar:
            if self.is_frozen: break
            z_b = z_train[i:i+B]
            y_b = None if y_train is None else y_train[i:i+B]
            step = i // B
            
            is_warmup = step < self.config.WARMUP_BATCHES
            
            if is_warmup:
                R_inference = torch.eye(d, device=self.device)
                phase_name = "WarmUp"
            else:
                R_inference = self.current_R
                phase_name = "Adapt"
                
            t_base = self._get_alignment_text_base()
            t_aligned = self._l2_norm(torch.matmul(t_base, R_inference.T))
            logits = self._effective_eval_scale() * torch.matmul(z_b, t_aligned.T) + self.b_c
            train_T = self.T_opt if self._mode_uses_calibration() else 1.0
            probs = torch.sigmoid(logits / max(float(train_T), 1e-4))

            g_info = self._guardian_update_from_window(step, probs)
            if g_info is not None and bool(g_info.get("changed", False)):
                if bool(g_info.get("frozen", False)):
                    self._log(
                        f"[GO] Alarm -> freeze at step={step}, PSI={float(g_info['psi']):.4f}, "
                        f"Stage2={float(g_info.get('stage2_stat', np.nan)):.4f}, rollback to R_last_good.",
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
            elif g_info is not None and bool(g_info.get("stage2_alarm", False)) and bool(getattr(self.config, "ENABLE_GO_GUARDIAN_STAGE2", False)):
                self._log(
                    f"[GO][Stage2] Alarm candidate at step={step}, stat={float(g_info.get('stage2_stat', np.nan)):.4f}.",
                    always=True,
                )

            if self._guardian_is_frozen():
                n_updates, n_rejected, n_support = 0, 0, 0
                if isinstance(self.R_last_good, torch.Tensor):
                    self.current_R = self.R_last_good.clone()
                    self.R_frozen = self.R_last_good.clone()
                    if isinstance(self.centroids_last_good, torch.Tensor):
                        self.image_centroids = self.centroids_last_good.clone()
            else:
                n_updates, n_rejected, n_support = self._update_centroids_gt_support(
                    z_b,
                    y_b,
                    t_aligned,
                    step=step,
                )
                self._set_prior_bias_from_counts(self.prior_counts)
            
            # === 瀹炴椂璁＄畻 Mask 鍜?璁℃暟 ===
            mask = self._get_alignment_active_mask()
            n_act = mask.sum().item()
            n_total = len(self.config.ORDERED_CLASS_NAMES)
            
            pbar_metrics = {
                "Ph": phase_name, 
                "N_act": f"{n_act}/{n_total}",
                "Upd": f"{n_updates}",
                "SUP": f"{n_support}",
                "REJ": f"{n_rejected}",
                "G": str(self.guardian_status),
                "PSI": f"{float(self.guardian_last_psi):.4f}" if np.isfinite(self.guardian_last_psi) else "NA",
                "G2": f"{float(getattr(self, 'guardian_stage2_last', np.nan)):.4f}" if np.isfinite(float(getattr(self, 'guardian_stage2_last', np.nan))) else "NA",
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
                    base_ref = self._get_mode_alignment_reference()
                    self.t_align_base = base_ref.clone() if isinstance(base_ref, torch.Tensor) else self._get_alignment_text_base().clone()
                    # 鏈夎冻澶熺被鍒椂锛岃绠?R锛坃solve_procrustes 鍐呴儴浼氳嚜鍔ㄥ拷鐣ユ牱鏈笉瓒崇殑绫诲埆锛?
                    R_full = self._solve_procrustes()
                    if self._mode_uses_guarded_alignment():
                        R_cand, stats = self._select_guarded_alignment_candidate(
                            R_full,
                            step,
                            phase=phase_name,
                            allow_soft_fallback=True,
                        )
                    else:
                        R_cand = R_full
                        stats = self._diagnose_and_log_dynamic(R_full, step, phase=phase_name, emit_log=False)
                        stats["alpha"] = 1.0
                        stats["selection_mode"] = "pure_full"
                        stats["mix_base"] = "full"
                        stats["score"] = self._alignment_stats_score(stats)
                        stats["soft_fallback"] = False
                        stats["passed"] = True
                    pbar_metrics.update(
                        {
                            "dS": f"{stats['dS']:.4f}",
                            "dOff": f"{float(stats.get('dOffDiag', 0.0)):+.4f}",
                            "rho*": f"{float(stats.get('rho_eff', self.config.RHO)):.3f}",
                            "a": f"{float(stats.get('alpha', 1.0)):.2f}",
                        }
                    )
                    
                    if stats["passed"]:
                        self._commit_alignment_candidate(R_cand, stats, gate_pass=True)
                        coverage_ratio = n_act / n_total
                        if self._mode_uses_guarded_alignment():
                            if coverage_ratio >= 0.8:
                                if bool(getattr(self.config, "AUDIT_DISABLE_EARLY_FREEZE", False)):
                                    self._log(
                                        f"[Gate Passed][Audit] Early freeze suppressed at step={step} "
                                        f"(coverage={coverage_ratio:.1%}, 螖s={stats['dS']:.4f}); continuing full train pass."
                                    )
                                else:
                                    self.is_frozen = True
                                    pbar.set_postfix(pbar_metrics)
                                    self._log(f"[Gate Passed] Freeze R at step={step} (coverage={coverage_ratio:.1%}, 螖s={stats['dS']:.4f})")
                                    self._save_and_report_per_class()
                                    saved_state = True
                                    pbar.close()
                                    break
                            else:
                                self._log(f" [Update] R updated with {n_act}/{n_total} classes ({coverage_ratio:.1%}).")
                        else:
                            self._log(f" [PureCAPA] Updated aligned R with {n_act}/{n_total} classes ({coverage_ratio:.1%}).")
                    elif bool(stats.get("soft_fallback", False)):
                        soft_score = float(stats.get("score", self._alignment_stats_score(stats)))
                        if soft_score > best_soft_candidate_score:
                            best_soft_candidate_score = soft_score
                            best_soft_candidate_R = R_cand.clone()
                            best_soft_candidate_stats = dict(stats)
                            self._log(
                                f" [SoftFallback] Tracking best GT candidate at step={step} "
                                f"(alpha={float(stats.get('alpha', 1.0)):.2f}, dS={float(stats.get('dS', 0.0)):+.4f}, "
                                f"dOff={float(stats.get('dOffDiag', 0.0)):+.4f})."
                            )
                    elif step == self.config.WARMUP_BATCHES:
                        self._log(f" [Gate Failed] Warmup insufficient (螖s={stats['dS']:.4f}).")

            if self.config.VERBOSE:
                pbar.set_postfix(pbar_metrics)
            
        if not self.is_frozen:
            if self._mode_uses_guarded_alignment() and isinstance(best_soft_candidate_R, torch.Tensor) and isinstance(best_soft_candidate_stats, dict):
                self._commit_alignment_candidate(best_soft_candidate_R, best_soft_candidate_stats, gate_pass=False)
                self._log(
                    f"[WARN] Loop finished without freezing; using best GT soft fallback "
                    f"(alpha={float(best_soft_candidate_stats.get('alpha', 1.0)):.2f}, dS={float(best_soft_candidate_stats.get('dS', 0.0)):+.4f}).",
                    always=True,
                )
            else:
                self._log("[WARN] Loop finished without freezing; using latest R for evaluation.")
                if self.current_R is None:
                    self.current_R = torch.eye(d, device=self.device)
                self.R_frozen = self.current_R
                self._snapshot_last_good_state()
        final_prior_counts = self.prior_counts if self.prior_counts is not None else self.support_counts
        self._set_prior_bias_from_counts(final_prior_counts)
        self._refresh_aligned_text()
        if not saved_state:
            self._save_and_report_per_class()

        # Legacy deployment-only evaluation sweep kept dormant for the two-mode main path.
        if run_stage4 and self._mode_uses_guarded_alignment():
            self._run_evaluation()
            self._run_scale_sweep([8, 12, 16, 24, 32, 40])

    def _compute_ece(self, y_true, y_prob, n_bins=10):
        """
        璁＄畻 Expected Calibration Error (ECE)銆?
        琛￠噺鐢变簬 TTA 瀵艰嚧鐨勬ā鍨嬭繃搴﹁嚜淇?Overconfidence)绋嬪害銆?
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        # 纭繚杞负 1D 鏁扮粍杩涜鍏ㄥ眬璁＄畻 (Micro-level ECE)
        y_true_flat = y_true.flatten()
        y_prob_flat = y_prob.flatten()
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # 钀藉湪杩欎釜缃俊搴﹀尯闂村唴鐨勬牱鏈?
            in_bin = (y_prob_flat > bin_lower) & (y_prob_flat <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true_flat[in_bin].mean()
                avg_confidence_in_bin = y_prob_flat[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece * 100.0  # 杞负鐧惧垎姣?

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
            calib_T=calib_T,
            scoring_mode=mode,
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
            probs_rank = self._predict_probs(
                logits, calib_T=1.0, is_multi=True, dataset_name=dataset_name, scoring_mode=mode, ranking=True
            ).cpu().numpy()
            probs_cal = self._predict_probs(
                logits, calib_T=calib_T, is_multi=True, dataset_name=dataset_name, scoring_mode=mode, ranking=False
            ).cpu().numpy()
            stats["Top1_Median"] = np.median(probs_cal.max(axis=1))

            try:
                if y_test.ndim != 2:
                    raise ValueError(f"Expected multilabel y_test as 2D array, got shape={getattr(y_test, 'shape', None)}")
                n_labels = int(y_test.shape[1])
                n_proto = int(probs_rank.shape[1])
                n_eval = max(1, min(n_labels, n_proto))

                # Assume label columns align with ORDERED_CLASS_NAMES[:n_eval]
                eval_indices = list(range(n_eval))
                probs_rank_eval = probs_rank[:, eval_indices]
                probs_cal_eval = probs_cal[:, eval_indices]
                y_eval = y_test[:, eval_indices]

                aucs = [roc_auc_score(y_eval[:, i], probs_rank_eval[:, i])
                        for i in range(n_eval) if len(np.unique(y_eval[:, i])) > 1]
                if aucs:
                    stats["Macro-AUC"] = np.mean(aucs)
                try:
                    stats["Micro-AUC"] = roc_auc_score(y_eval, probs_rank_eval, average="micro")
                except Exception:
                    pass

                stats["ECE"] = self._compute_ece(y_eval, probs_cal_eval)
                stats["Acc"] = f1_score(y_eval, (probs_cal_eval > 0.5).astype(int), average="macro", zero_division=0)
                stats["Brier"] = float(np.mean((probs_cal_eval - y_eval) ** 2))

                if "DEBUG_AUC_CHECK" in os.environ:
                    try:
                        aucs_T1 = [roc_auc_score(y_eval[:, i], probs_rank_eval[:, i])
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

    def _compute_metrics_from_prob_arrays(
        self,
        y_true,
        probs_rank,
        probs_cal,
        *,
        is_multi: bool,
    ) -> Dict[str, float]:
        stats = {
            "Macro-AUC": np.nan,
            "Micro-AUC": np.nan,
            "ECE": np.nan,
            "Acc": 0.0,
            "Top1_Median": 0.0,
            "Brier": np.nan,
        }
        if is_multi:
            y_arr = np.asarray(y_true)
            s_rank = np.asarray(probs_rank, dtype=np.float64)
            s_cal = np.asarray(probs_cal, dtype=np.float64)
            if y_arr.ndim != 2 or s_rank.ndim != 2 or s_cal.ndim != 2:
                return stats
            n_use = max(1, min(y_arr.shape[1], s_rank.shape[1], s_cal.shape[1]))
            y_eval = y_arr[:, :n_use]
            rank_eval = s_rank[:, :n_use]
            cal_eval = s_cal[:, :n_use]
            aucs = []
            for i in range(n_use):
                if len(np.unique(y_eval[:, i])) < 2:
                    continue
                try:
                    aucs.append(float(roc_auc_score(y_eval[:, i], rank_eval[:, i])))
                except Exception:
                    continue
            if aucs:
                stats["Macro-AUC"] = float(np.mean(aucs))
            try:
                stats["Micro-AUC"] = float(roc_auc_score(y_eval, rank_eval, average="micro"))
            except Exception:
                pass
            stats["ECE"] = self._compute_ece(y_eval, cal_eval)
            stats["Acc"] = float(f1_score(y_eval, (cal_eval > 0.5).astype(int), average="macro", zero_division=0))
            stats["Top1_Median"] = float(np.median(cal_eval.max(axis=1)))
            stats["Brier"] = float(np.mean((cal_eval - y_eval) ** 2))
            return stats

        y_bin = np.asarray(y_true).reshape(-1)
        s_rank = np.asarray(probs_rank, dtype=np.float64).reshape(-1)
        s_cal = np.asarray(probs_cal, dtype=np.float64).reshape(-1)
        n = min(len(y_bin), len(s_rank), len(s_cal))
        if n <= 0:
            return stats
        y_bin = y_bin[:n]
        s_rank = s_rank[:n]
        s_cal = s_cal[:n]
        if len(np.unique(y_bin)) >= 2:
            try:
                auc_val = float(roc_auc_score(y_bin, s_rank))
                stats["Macro-AUC"] = auc_val
                stats["Micro-AUC"] = auc_val
            except Exception:
                pass
        stats["ECE"] = self._compute_ece(y_bin, s_cal)
        stats["Acc"] = float(accuracy_score(y_bin, (s_cal > 0.5).astype(int)))
        stats["Top1_Median"] = float(np.median(s_cal))
        stats["Brier"] = float(np.mean((s_cal - y_bin) ** 2))
        return stats

    def _load_saved_state_for_evaluation(self) -> Dict[str, object]:
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
        centroids_last_good = state.get("centroids_last_good", None)
        self.centroids_last_good = centroids_last_good.to(self.device) if isinstance(centroids_last_good, torch.Tensor) else None
        counts_blob = state.get("support_counts", state.get("counts"))
        counts = counts_blob.to(self.device)
        self.support_counts = counts.clone()
        counts_lq = state.get("rejected_counts", state.get("counts_lq", None))
        self.rejected_counts = counts_lq.to(self.device) if isinstance(counts_lq, torch.Tensor) else None
        prior_counts = state.get("prior_counts", None)
        self.prior_counts = prior_counts.to(self.device) if isinstance(prior_counts, torch.Tensor) else counts.clone()
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

        if self.t_align_base is None:
            base_ref = self._get_mode_alignment_reference()
            self.t_align_base = base_ref.clone() if isinstance(base_ref, torch.Tensor) else self.t_raw_pooled.clone()
        self._set_prior_bias_from_counts(self.prior_counts if self.prior_counts is not None else counts)
        if isinstance(state.get("t_raw_text", None), torch.Tensor):
            self.t_raw_text = state["t_raw_text"].to(self.device)
        if isinstance(state.get("t_processed_text", None), torch.Tensor):
            self.t_processed_text = state["t_processed_text"].to(self.device)
        self._refresh_aligned_text()
        return state

    def _predict_dataset_branch_outputs(
        self,
        dataset_name: str,
        path: str,
        *,
        scoring_mode: Optional[str] = None,
        split_override: int = 2,
    ) -> Dict[str, object]:
        if self.R_frozen is None or self.support_counts is None or self.prior_counts is None:
            self._load_saved_state_for_evaluation()

        mode = self._resolve_scoring_mode(scoring_mode)
        t_base_proc = self.t_zero_shot_base if isinstance(self.t_zero_shot_base, torch.Tensor) else self.t_raw_pooled
        t_align_base_proc = self._get_alignment_text_base()
        t_capa_proc = self._l2_norm(torch.matmul(t_align_base_proc, self.R_frozen.T))

        z_test_raw, y_test, is_multi = self._load_data(path, split_override=int(split_override))
        if y_test is None:
            raise RuntimeError(f"Dataset has no labels: {dataset_name}")
        z_test_proc = self._apply_preprocessing(z_test_raw, self.zI_mean)

        g_cpu = torch.Generator()
        g_cpu.manual_seed(self.config.RANDOM_SEED)
        perm = torch.randperm(len(z_test_raw), generator=g_cpu)
        z_eval_proc = z_test_proc[perm]
        y_eval = y_test[perm.cpu().numpy()]

        z_tau_raw, y_tau, _ = self._load_data(
            self.config.TAU_CALIB_DATA_PATH,
            is_calibration=True,
            split_override=1,
        )
        if y_tau is None:
            raise RuntimeError(f"TAU calibration set has no labels: {self.config.TAU_CALIB_DATA_PATH}")
        z_tau_proc = self._apply_preprocessing(z_tau_raw, self.zI_mean)
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

        logits_tau = self._compose_eval_logits(
            z_tau_proc,
            t_capa_proc,
            scale=float(self.s_opt),
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
            dataset_name=dataset_name,
            is_multi=is_multi,
            scoring_mode=mode,
        )

        logits_capa = self._compose_eval_logits(
            z_eval_proc,
            t_capa_proc,
            scale=float(self.s_opt),
            baseline_t_protos=t_base_proc,
        )
        logits_base = float(self.s_opt) * torch.matmul(z_eval_proc, t_base_proc.T) + self.b_c

        probs_capa_cal = self._predict_probs(
            logits_capa,
            calib_T=tau_cal,
            is_multi=is_multi,
            dataset_name=dataset_name,
            scoring_mode=mode,
            ranking=False,
        ).detach().cpu().numpy()
        probs_capa_rank = self._predict_probs(
            logits_capa,
            calib_T=1.0,
            is_multi=is_multi,
            dataset_name=dataset_name,
            scoring_mode=mode,
            ranking=True,
        ).detach().cpu().numpy()
        probs_base_cal = self._predict_probs(
            logits_base,
            calib_T=tau_cal,
            is_multi=is_multi,
            dataset_name=dataset_name,
            scoring_mode=mode,
            ranking=False,
        ).detach().cpu().numpy()
        probs_base_rank = self._predict_probs(
            logits_base,
            calib_T=1.0,
            is_multi=is_multi,
            dataset_name=dataset_name,
            scoring_mode=mode,
            ranking=True,
        ).detach().cpu().numpy()

        if is_multi:
            n_use = max(1, min(int(y_eval.shape[1]), int(probs_capa_rank.shape[1]), int(probs_base_rank.shape[1])))
            y_eval_use = np.asarray(y_eval)[:, :n_use]
            probs_capa_cal_use = probs_capa_cal[:, :n_use]
            probs_capa_rank_use = probs_capa_rank[:, :n_use]
            probs_base_cal_use = probs_base_cal[:, :n_use]
            probs_base_rank_use = probs_base_rank[:, :n_use]
            conf = probs_capa_cal_use.max(axis=1)
        else:
            y_eval_use = np.asarray(y_eval).reshape(-1)
            probs_capa_cal_use = np.asarray(probs_capa_cal).reshape(-1)
            probs_capa_rank_use = np.asarray(probs_capa_rank).reshape(-1)
            probs_base_cal_use = np.asarray(probs_base_cal).reshape(-1)
            probs_base_rank_use = np.asarray(probs_base_rank).reshape(-1)
            conf = probs_capa_cal_use.reshape(-1)

        stats_capa = self._compute_metrics_from_prob_arrays(
            y_eval_use,
            probs_capa_rank_use,
            probs_capa_cal_use,
            is_multi=is_multi,
        )
        stats_base = self._compute_metrics_from_prob_arrays(
            y_eval_use,
            probs_base_rank_use,
            probs_base_cal_use,
            is_multi=is_multi,
        )
        return {
            "dataset": dataset_name,
            "is_multi": bool(is_multi),
            "y": y_eval_use,
            "tau_cal": float(tau_cal),
            "probs_capa_rank": probs_capa_rank_use,
            "probs_capa_cal": probs_capa_cal_use,
            "probs_base_rank": probs_base_rank_use,
            "probs_base_cal": probs_base_cal_use,
            "confidence": np.asarray(conf, dtype=np.float64).reshape(-1),
            "stats_capa": stats_capa,
            "stats_base": stats_base,
        }

    def _macro_auc_subset(
        self,
        y_true,
        probs_rank,
        *,
        is_multi: bool,
        sample_mask: Optional[np.ndarray] = None,
        class_indices: Optional[List[int]] = None,
    ) -> float:
        y_arr = np.asarray(y_true)
        s_arr = np.asarray(probs_rank, dtype=np.float64)
        if sample_mask is not None:
            mask = np.asarray(sample_mask, dtype=bool).reshape(-1)
            if y_arr.shape[0] != mask.shape[0]:
                n = min(int(y_arr.shape[0]), int(mask.shape[0]))
                y_arr = y_arr[:n]
                s_arr = s_arr[:n]
                mask = mask[:n]
            y_arr = y_arr[mask]
            s_arr = s_arr[mask]
        if len(y_arr) <= 1:
            return np.nan
        if not is_multi:
            y_bin = np.asarray(y_arr).reshape(-1)
            s_bin = np.asarray(s_arr, dtype=np.float64).reshape(-1)
            if len(np.unique(y_bin)) < 2:
                return np.nan
            try:
                return float(roc_auc_score(y_bin, s_bin))
            except Exception:
                return np.nan

        if y_arr.ndim != 2 or s_arr.ndim != 2:
            return np.nan
        if class_indices is None:
            class_indices = list(range(min(int(y_arr.shape[1]), int(s_arr.shape[1]))))
        aucs = []
        for idx in class_indices:
            if idx >= y_arr.shape[1] or idx >= s_arr.shape[1]:
                continue
            y_c = y_arr[:, idx]
            if len(np.unique(y_c)) < 2:
                continue
            try:
                aucs.append(float(roc_auc_score(y_c, s_arr[:, idx])))
            except Exception:
                continue
        return float(np.mean(aucs)) if aucs else np.nan

    def _compute_eval_go_ml_risk_features(
        self,
        z_eval: torch.Tensor,
        y_eval,
        t_protos: torch.Tensor,
    ) -> pd.DataFrame:
        n_cls = int(t_protos.shape[0])
        active_mask = self._labels_to_active_mask(y_eval, n_cls=n_cls)
        support_np = None
        support_med = np.nan
        if isinstance(self.support_counts, torch.Tensor):
            support_np = self.support_counts.detach().cpu().numpy().astype(np.float64)
            support_med = float(np.median(support_np)) if support_np.size > 0 else np.nan
        rows = []
        for i in range(int(z_eval.shape[0])):
            active_indices = torch.where(active_mask[i])[0].tolist()
            n_active = len(active_indices)
            cond_vals = []
            resid_ratio_vals = []
            sim_vals = []
            for c_idx in active_indices:
                _, _, meta = self._compute_multilabel_residual(z_eval[i], active_indices, c_idx, t_protos)
                cond_val = float(meta.get("cond_reg", np.nan))
                ratio_val = float(meta.get("resid_norm_ratio", np.nan))
                sim_val = float(meta.get("max_other_sim", np.nan))
                if np.isfinite(cond_val):
                    cond_vals.append(cond_val)
                if np.isfinite(ratio_val):
                    resid_ratio_vals.append(ratio_val)
                if np.isfinite(sim_val):
                    sim_vals.append(sim_val)
            support_vals = [float(support_np[idx]) for idx in active_indices] if support_np is not None and active_indices else []
            min_support = float(min(support_vals)) if support_vals else np.nan
            rows.append(
                {
                    "sample_idx": i,
                    "active_count": n_active,
                    "cond_reg_max": float(max(cond_vals)) if cond_vals else np.nan,
                    "resid_ratio_max": float(max(resid_ratio_vals)) if resid_ratio_vals else np.nan,
                    "max_other_sim": float(max(sim_vals)) if sim_vals else np.nan,
                    "min_support": min_support,
                    "freq_bucket": (
                        "low"
                        if np.isfinite(min_support) and np.isfinite(support_med) and min_support < support_med
                        else "high"
                    ) if active_indices else "none",
                }
            )
        return pd.DataFrame(rows)

    def _evaluate_prompt_stage(
        self,
        z_test: torch.Tensor,
        y_test,
        is_multi: bool,
        t_protos: torch.Tensor,
        *,
        dataset_name: str,
        prior_counts: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, float], np.ndarray]:
        old_b = None if self.b_c is None else self.b_c.clone()
        if prior_counts is None:
            self.b_c = torch.zeros((1, int(t_protos.shape[0])), device=self.device)
        else:
            self._set_prior_bias_from_counts(prior_counts)

        stats = self._compute_metrics(
            z_test,
            y_test,
            is_multi,
            t_protos,
            scale_override=self.s_opt,
            temperature_override=self.T_opt,
            dataset_name=dataset_name,
            scoring_mode=self._resolve_scoring_mode(None),
            baseline_t_protos=None,
            use_cache=False,
        )
        logits = self.s_opt * torch.matmul(z_test, t_protos.T) + self.b_c
        probs_rank = self._predict_probs(
            logits,
            calib_T=1.0,
            is_multi=is_multi,
            dataset_name=dataset_name,
            scoring_mode=self._resolve_scoring_mode(None),
            ranking=True,
        ).detach().cpu().numpy()

        if old_b is None:
            self.b_c = None
        else:
            self.b_c = old_b
        return stats, probs_rank

    def _per_class_auc_rows(
        self,
        y_true,
        probs_rank: np.ndarray,
        *,
        dataset_name: str,
        stage_name: str,
    ) -> List[Dict[str, object]]:
        y_arr = np.asarray(y_true)
        if y_arr.ndim != 2:
            return []
        n_use = min(int(y_arr.shape[1]), int(probs_rank.shape[1]), len(self.config.ORDERED_CLASS_NAMES))
        rows: List[Dict[str, object]] = []
        for c in range(n_use):
            y_c = y_arr[:, c]
            auc_val = np.nan
            if len(np.unique(y_c)) >= 2:
                try:
                    auc_val = float(roc_auc_score(y_c, probs_rank[:, c]))
                except Exception:
                    auc_val = np.nan
            rows.append(
                {
                    "Dataset": dataset_name,
                    "Stage": stage_name,
                    "Class": str(self.config.ORDERED_CLASS_NAMES[c]),
                    "AUC": auc_val,
                    "Prevalence": float(np.mean(y_c)),
                    "Positives": int(np.sum(y_c > 0)),
                    "Negatives": int(np.sum(y_c <= 0)),
                }
            )
        return rows

    def _prototype_angle_deg(self, A: torch.Tensor, B: torch.Tensor) -> np.ndarray:
        n_use = min(int(A.shape[0]), int(B.shape[0]))
        a = self._l2_norm(A[:n_use])
        b = self._l2_norm(B[:n_use])
        cos = torch.clamp(torch.sum(a * b, dim=1), min=-1.0, max=1.0)
        ang = torch.rad2deg(torch.acos(cos)).detach().cpu().numpy()
        return np.asarray(ang, dtype=np.float64)

    def _build_fixed_alignment_with_external_state(
        self,
        *,
        centroids: torch.Tensor,
        support_counts: torch.Tensor,
        prior_counts: torch.Tensor,
        rejected_counts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        backup = {
            "image_centroids": None if self.image_centroids is None else self.image_centroids.clone(),
            "support_counts": None if self.support_counts is None else self.support_counts.clone(),
            "prior_counts": None if self.prior_counts is None else self.prior_counts.clone(),
            "rejected_counts": None if self.rejected_counts is None else self.rejected_counts.clone(),
            "t_align_base": None if self.t_align_base is None else self.t_align_base.clone(),
            "b_c": None if self.b_c is None else self.b_c.clone(),
        }
        try:
            self.image_centroids = centroids.clone().to(self.device)
            self.support_counts = support_counts.clone().to(self.device)
            self.prior_counts = prior_counts.clone().to(self.device)
            self.rejected_counts = (
                rejected_counts.clone().to(self.device)
                if isinstance(rejected_counts, torch.Tensor)
                else torch.zeros_like(self.support_counts)
            )
            base_ref = self._get_mode_alignment_reference()
            self.t_align_base = base_ref.clone() if isinstance(base_ref, torch.Tensor) else self.t_raw_pooled.clone()
            self._set_prior_bias_from_counts(self.prior_counts)
            R_full = self._solve_procrustes()
            R_sel, stats = self._select_guarded_alignment_candidate(
                R_full,
                step=int(getattr(self.config, "WARMUP_BATCHES", 0)),
                phase="PromptDiag",
                allow_soft_fallback=True,
            )
            t_eval = self._l2_norm(torch.matmul(self.t_align_base, R_sel.T))
            return t_eval, dict(stats)
        finally:
            self.image_centroids = backup["image_centroids"]
            self.support_counts = backup["support_counts"]
            self.prior_counts = backup["prior_counts"]
            self.rejected_counts = backup["rejected_counts"]
            self.t_align_base = backup["t_align_base"]
            self.b_c = backup["b_c"]

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

        # local refinement within 卤20% of best grid point
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

    def _iter_selected_test_datasets(self, datasets: Optional[List[str]] = None) -> List[Tuple[str, str]]:
        requested = None
        if datasets:
            requested = {self._canonical_dataset_name(item) for item in datasets}
        selected: List[Tuple[str, str]] = []
        for name, path in self.config.TEST_DATA_PATHS.items():
            canonical = self._canonical_dataset_name(name)
            if requested is not None and canonical not in requested:
                continue
            selected.append((canonical or str(name), path))
        if requested is not None and not selected:
            raise ValueError(f"No datasets matched {datasets}. Available: {list(self.config.TEST_DATA_PATHS)}")
        return selected

    def _fit_eval_temperature(
        self,
        dataset_name: str,
        *,
        t_eval: torch.Tensor,
        baseline_t_protos: Optional[torch.Tensor],
        scoring_mode: Optional[str] = None,
    ) -> float:
        if not self._mode_uses_calibration():
            return 1.0

        z_tau_raw, y_tau, is_multi_tau = self._load_data(
            self.config.TAU_CALIB_DATA_PATH,
            is_calibration=True,
            split_override=1,
        )
        if y_tau is None:
            raise RuntimeError(f"TAU calibration set has no labels: {self.config.TAU_CALIB_DATA_PATH}")

        z_tau = self._prepare_eval_embeddings(z_tau_raw)
        g_cpu_tau = torch.Generator()
        g_cpu_tau.manual_seed(self.config.RANDOM_SEED)
        perm_tau = torch.randperm(len(z_tau), generator=g_cpu_tau)
        z_tau = z_tau[perm_tau]
        if isinstance(y_tau, np.ndarray):
            y_tau = y_tau[perm_tau.cpu().numpy()]

        n_tau_total = len(z_tau)
        n_tau = max(1, int(self.config.TAU_CALIB_FRAC * n_tau_total))
        if n_tau_total >= 2500:
            n_tau = min(self.config.TAU_CALIB_MAX, max(self.config.TAU_CALIB_MIN, n_tau))
        z_tau = z_tau[:n_tau]
        y_tau = y_tau[:n_tau]

        self._set_eval_bias_for_prototypes(t_eval)
        logits_tau = self._compose_eval_logits(
            z_tau,
            t_eval,
            scale=self._effective_eval_scale(),
            baseline_t_protos=baseline_t_protos,
        ).cpu().numpy()
        tau_cal, _ = self._fit_posthoc_tau(
            logits_tau,
            np.asarray(y_tau),
            dataset_name=dataset_name,
            is_multi=bool(is_multi_tau),
            scoring_mode=scoring_mode,
        )
        return float(tau_cal)

    def _build_eval_audit_summary(
        self,
        *,
        dataset_name: str,
        dataset_path: str,
        split_override: int,
        prototype_source: str,
        temperature: float,
        scale: float,
        run_dir: str,
    ) -> Dict[str, object]:
        return {
            "eval_mode": self._current_eval_mode(),
            "entry_script": str(getattr(self.config, "ENTRY_SCRIPT", ENTRY_SCRIPT_PATH)),
            "dataset": str(dataset_name),
            "dataset_path": str(dataset_path),
            "split": int(split_override),
            "backbone": str(self.config.MODEL_NAME),
            "train_data_path": str(getattr(self.config, "TRAIN_DATA_PATH", "")),
            "cross_modal_data_path": str(getattr(self.config, "CROSS_MODAL_DATA_PATH", "")),
            "calib_data_path": str(getattr(self.config, "CALIB_DATA_PATH", "")),
            "tau_calib_data_path": str(getattr(self.config, "TAU_CALIB_DATA_PATH", "")),
            "data_semantics": {
                "train": "CheXpert-5 target-aware view materialized from data_train.pkl for CAPA adaptation/training",
                "test": "dataset-specific zero-shot evaluation file",
                "cross_modal": "CHEXPERT_MIMIC.pkl for cross-modal retrieval analysis",
                "calibration": "target-aware image-only calibration subset materialized from data_train.pkl",
            },
            "prototype_source": str(prototype_source),
            "preprocessing_on": bool(self._mode_uses_image_preprocessing()),
            "alignment_on": bool(self._mode_uses_alignment()),
            "cache_on": bool(self._is_cache_enabled()),
            "dual_track_on": bool(self._uses_dual_track_inference()),
            "guardian_on": bool(self._is_go_guardian_enabled()),
            "prior_correction_on": bool(self._mode_uses_prior_correction()),
            "calibration_on": bool(self._mode_uses_calibration()),
            "final_logits_source": str(self.eval_runtime.get("final_logits_source", "")),
            "early_text_prompt_support": {
                "enabled": bool(getattr(self.config, "ENABLE_EARLY_TEXT_PROMPT_SUPPORT", True))
                and bool(getattr(self.config, "PROMPT_TEXT_EMBEDDING_GROUPS", None)),
                "top_k": int(getattr(self.config, "EARLY_TEXT_PROMPT_TOP_K", 12)),
                "entry_mode": str(getattr(self.config, "EARLY_TEXT_PROMPT_ENTRY_MODE", "full")),
                "selection_mode": str(getattr(self.config, "EARLY_TEXT_PROMPT_SELECTION_MODE", "margin")),
                "info": dict(getattr(self, "early_text_prompt_support_info", {})),
            },
            "metric_source": str(self.eval_runtime.get("metric_source", f"{Path(__file__).name}::_compute_metrics")),
            "run_dir": str(run_dir),
            "scale": float(scale),
            "temperature": float(temperature),
            "notes": str(self.eval_runtime.get("notes", "")),
        }

    def _shift_gate_enabled(self) -> bool:
        return (
            self._current_eval_mode() == "full_capa"
            and bool(getattr(self.config, "ENABLE_CAPA_SHIFT_GATE", True))
            and (not bool(getattr(self.config, "DISABLE_SHARED_CENTERING", False)))
        )

    def _dataset_calibration_shift_l2(self, dataset_path: str) -> float:
        z_cal, _, _ = self._load_data(self.config.CALIB_DATA_PATH, is_calibration=True, split_override=1)
        cal_mean = self._l2_norm(self._l2_norm(z_cal).mean(dim=0, keepdim=True)).view(-1)
        z_raw, _, _ = self._load_data(dataset_path, split_override=2)
        ds_mean = self._l2_norm(self._l2_norm(z_raw).mean(dim=0, keepdim=True)).view(-1)
        return float(torch.norm(ds_mean - cal_mean).item())

    def _build_shift_gate_fallback_runner(self) -> "CAPA5NotebookRunner":
        cfg = copy.deepcopy(self.config)
        cfg.SAVE_DIR = os.path.join(self.config.SAVE_DIR, "shift_gate_no_center")
        cfg.PRINT_SUMMARY = False
        cfg.VERBOSE = False
        cfg.DEBUG = False
        cfg.DISABLE_SHARED_CENTERING = True
        cfg.ENABLE_CAPA_SHIFT_GATE = False
        cfg.ENABLE_PROMPT_BANK_READOUT = False
        cfg.ENABLE_EARLY_TEXT_PROMPT_SUPPORT = False
        if hasattr(cfg, "PROMPT_TEXT_EMBEDDING_GROUPS"):
            delattr(cfg, "PROMPT_TEXT_EMBEDDING_GROUPS")
        cfg.ENABLE_RESIDUAL_LOCAL_SLERP = False
        cfg.ENABLE_DISC_AXIS_PROCRUSTES = False
        runner = CAPA5NotebookRunner(cfg)
        runner._init_state()
        runner.eval_runtime = runner._build_eval_runtime()
        runner.run_pipeline(run_stage4=False)
        return runner

    def run_eval_mode_report(
        self,
        *,
        datasets: Optional[List[str]] = None,
        split_override: int = 2,
        scoring_mode: Optional[str] = None,
    ) -> pd.DataFrame:
        self._init_state()
        self.eval_runtime = self._build_eval_runtime()
        self._log_eval_mode_summary(always=True)

        if self._mode_uses_alignment():
            self.run_pipeline(run_stage4=False)
        else:
            self._prepare_shared_feature_space()

        mode = self._current_eval_mode()
        score_mode = self._resolve_scoring_mode(scoring_mode)
        t_eval, baseline_t_protos, prototype_source = self._get_eval_prototype_bundle()
        scale = self._effective_eval_scale()
        self._log(
            "[DataLayout] "
            f"train={self.config.TRAIN_DATA_PATH} | "
            f"cross_modal={getattr(self.config, 'CROSS_MODAL_DATA_PATH', '')} | "
            f"calib={self.config.CALIB_DATA_PATH} | "
            f"tau_calib={self.config.TAU_CALIB_DATA_PATH}",
            always=True,
        )

        audit_dir = os.path.join(self.config.SAVE_DIR, "audit")
        os.makedirs(audit_dir, exist_ok=True)
        report_rows: List[Dict[str, object]] = []
        fallback_runner: Optional[CAPA5NotebookRunner] = None

        for dataset_name, path in self._iter_selected_test_datasets(datasets):
            if not os.path.exists(path):
                continue
            eval_runner: CAPA5NotebookRunner = self
            eval_t = t_eval
            eval_baseline_t = baseline_t_protos
            eval_prototype_source = prototype_source
            path_choice = "centered"
            shift_l2 = np.nan
            if self._shift_gate_enabled():
                shift_l2 = self._dataset_calibration_shift_l2(path)
                if shift_l2 > float(getattr(self.config, "CAPA_SHIFT_GATE_THRESHOLD", 0.22)):
                    if fallback_runner is None:
                        fallback_runner = self._build_shift_gate_fallback_runner()
                    eval_runner = fallback_runner
                    eval_t, eval_baseline_t, eval_prototype_source = eval_runner._get_eval_prototype_bundle()
                    path_choice = "no_center"

            z_test_raw, y_test, is_multi = eval_runner._load_data(path, split_override=split_override)
            if y_test is None:
                continue
            z_test = eval_runner._prepare_eval_embeddings(z_test_raw)
            eval_runner._set_eval_bias_for_prototypes(eval_t)
            eval_scale = eval_runner._effective_eval_scale()
            if eval_runner._mode_uses_prompt_bank_readout():
                tau_eval = float(getattr(eval_runner.config, "PROMPT_BANK_READOUT_CALIB_T", 2.0))
                eval_runner.last_dualtrack_eval_info = {}
                logits = eval_runner._compose_prompt_bank_readout_logits(z_test)
                stats = eval_runner._compute_metrics_from_logits(
                    logits,
                    y_test,
                    bool(is_multi),
                    calib_T=tau_eval,
                    dataset_name=dataset_name,
                    scoring_mode=score_mode,
                )
                eval_prototype_source = f"{eval_prototype_source}+logit_prompt_bank"
            else:
                tau_eval = eval_runner._fit_eval_temperature(
                    dataset_name,
                    t_eval=eval_t,
                    baseline_t_protos=eval_baseline_t,
                    scoring_mode=score_mode,
                )
                stats = eval_runner._compute_metrics(
                    z_test,
                    y_test,
                    is_multi,
                    eval_t,
                    scale_override=eval_scale,
                    temperature_override=tau_eval,
                    dataset_name=dataset_name,
                    scoring_mode=score_mode,
                    use_cache=eval_runner._is_cache_enabled(),
                    baseline_t_protos=eval_baseline_t,
                )
            audit = eval_runner._build_eval_audit_summary(
                dataset_name=dataset_name,
                dataset_path=path,
                split_override=split_override,
                prototype_source=eval_prototype_source,
                temperature=tau_eval,
                scale=eval_scale,
                run_dir=eval_runner.config.SAVE_DIR,
            )
            audit.update(
                {
                    "route": "early_text_support_disc_axis_residual_local_prompt_bank_shift_gate"
                    if self._current_eval_mode() == "full_capa"
                    else self._current_eval_mode(),
                    "path_choice": path_choice,
                    "shift_l2": float(shift_l2) if np.isfinite(shift_l2) else None,
                    "shift_threshold": float(getattr(self.config, "CAPA_SHIFT_GATE_THRESHOLD", 0.22)),
                    "disc_axis_on": bool(getattr(eval_runner.config, "ENABLE_DISC_AXIS_PROCRUSTES", True)),
                    "disc_axis_neg_lambda": float(getattr(eval_runner.config, "DISC_AXIS_NEG_LAMBDA", 0.25)),
                    "residual_local_on": bool(getattr(eval_runner.config, "ENABLE_RESIDUAL_LOCAL_SLERP", True)),
                    "early_text_prompt_support_on": bool(getattr(eval_runner.config, "ENABLE_EARLY_TEXT_PROMPT_SUPPORT", True))
                    and bool(getattr(eval_runner.config, "PROMPT_TEXT_EMBEDDING_GROUPS", None)),
                    "early_text_prompt_top_k": int(getattr(eval_runner.config, "EARLY_TEXT_PROMPT_TOP_K", 12)),
                    "early_text_prompt_entry_mode": str(getattr(eval_runner.config, "EARLY_TEXT_PROMPT_ENTRY_MODE", "full")),
                    "early_text_prompt_selection_mode": str(getattr(eval_runner.config, "EARLY_TEXT_PROMPT_SELECTION_MODE", "margin")),
                    "early_text_prompt_support_info": dict(getattr(eval_runner, "early_text_prompt_support_info", {})),
                    "prompt_bank_readout_on": bool(eval_runner._mode_uses_prompt_bank_readout()),
                    "shared_centering_disabled": bool(getattr(eval_runner.config, "DISABLE_SHARED_CENTERING", False)),
                    "residual_local_slerp_info": list(getattr(eval_runner, "residual_local_slerp_info", [])),
                }
            )
            audit_path = os.path.join(audit_dir, f"{mode}_{dataset_name.lower()}_audit.json")
            with open(audit_path, "w", encoding="utf-8") as f:
                json.dump(audit, f, indent=2, ensure_ascii=True, default=_json_default)

            row = {
                "mode": mode,
                "dataset": dataset_name,
                "split": int(split_override),
                "scoring_mode": score_mode,
                "macro_auc": float(stats.get("Macro-AUC", np.nan)),
                "micro_auc": float(stats.get("Micro-AUC", np.nan)),
                "ece": float(stats.get("ECE", np.nan)),
                "acc": float(stats.get("Acc", np.nan)),
                "brier": float(stats.get("Brier", np.nan)),
                "temperature": float(tau_eval),
                "scale": float(eval_scale),
                "run_dir": str(self.config.SAVE_DIR),
                "prototype_source": eval_prototype_source,
                "final_logits_source": str(audit["final_logits_source"]),
                "path_choice": path_choice,
                "shift_l2": float(shift_l2) if np.isfinite(shift_l2) else np.nan,
                "shift_threshold": float(getattr(self.config, "CAPA_SHIFT_GATE_THRESHOLD", 0.22)),
                "dual_track_used": bool(eval_runner.last_dualtrack_eval_info.get("enabled", False)),
                "cache_used": bool(eval_runner._is_cache_enabled()),
                "guardian_used": bool(eval_runner._is_go_guardian_enabled()),
                "prior_correction_used": bool(eval_runner._mode_uses_prior_correction()),
                "calibration_used": bool(eval_runner._mode_uses_calibration()),
                "prompt_bank_readout_used": bool(eval_runner._mode_uses_prompt_bank_readout()),
                "notes": str(self.eval_runtime.get("notes", "")),
                "audit_summary_path": audit_path,
            }
            if eval_runner.last_dualtrack_eval_info:
                row["dualtrack_aligned_rate"] = float(eval_runner.last_dualtrack_eval_info.get("aligned_rate", np.nan))
                row["dualtrack_agree_rate"] = float(eval_runner.last_dualtrack_eval_info.get("agree_rate", np.nan))
            if eval_runner.last_cache_eval_info:
                row["cache_usage_rate"] = float(eval_runner.last_cache_eval_info.get("usage_rate", 0.0))
                row["cache_mean_alpha"] = float(eval_runner.last_cache_eval_info.get("mean_alpha", 0.0))
            report_rows.append(row)

        df = pd.DataFrame(report_rows)
        out_csv = os.path.join(self.config.SAVE_DIR, f"eval_mode_report_{mode}.csv")
        df.to_csv(out_csv, index=False)
        out_json = os.path.join(self.config.SAVE_DIR, f"eval_mode_report_{mode}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report_rows, f, indent=2, ensure_ascii=True, default=_json_default)
        self._log(f"[Saved] {out_csv}", always=True)
        self._log(f"[Saved] {out_json}", always=True)
        return df

    def _get_gate_active_mask(self, n_cls: int) -> torch.Tensor:
        return self._get_alignment_active_mask(n_cls=n_cls)

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

        sim_pre_full = torch.matmul(mu, t_base_norm.T)
        sim_post_full = torch.matmul(mu, t_rot_norm.T)
        sim_pre = sim_pre_full.diag()
        sim_post = sim_post_full.diag()
        sim_before = sim_pre[mask].mean()
        sim_after = sim_post[mask].mean()
        sim_gain = sim_after - sim_before

        idx = torch.where(mask)[0]
        n_act = int(idx.numel())
        if n_act <= 1:
            off_diag_pre = torch.tensor([0.0], device=self.device)
            off_diag_post = torch.tensor([0.0], device=self.device)
        else:
            sim_pre_act = sim_pre_full.index_select(0, idx).index_select(1, idx)
            sim_post_act = sim_post_full.index_select(0, idx).index_select(1, idx)
            off_mask = ~torch.eye(n_act, dtype=torch.bool, device=self.device)
            off_diag_pre = sim_pre_act[off_mask]
            off_diag_post = sim_post_act[off_mask]

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
        t_base = self.t_zero_shot_base if isinstance(self.t_zero_shot_base, torch.Tensor) else self.t_raw_pooled
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
        centroids_last_good = state.get("centroids_last_good", None)
        self.centroids_last_good = centroids_last_good.to(self.device) if isinstance(centroids_last_good, torch.Tensor) else None
        counts_blob = state.get("support_counts", state.get("counts"))
        counts = counts_blob.to(self.device)
        self.support_counts = counts.clone()
        counts_lq = state.get("rejected_counts", state.get("counts_lq", None))
        self.rejected_counts = counts_lq.to(self.device) if isinstance(counts_lq, torch.Tensor) else None
        prior_counts = state.get("prior_counts", None)
        self.prior_counts = prior_counts.to(self.device) if isinstance(prior_counts, torch.Tensor) else counts.clone()
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

        if self.t_align_base is None:
            base_ref = self._get_mode_alignment_reference()
            self.t_align_base = base_ref.clone() if isinstance(base_ref, torch.Tensor) else self.t_raw_pooled.clone()

        eval_scale = float(self.s_opt)
        self._set_prior_bias_from_counts(self.prior_counts if self.prior_counts is not None else counts)

        # Keep alignment-space (processed) heads for both metrics and Sim reporting.
        t_base_proc = self.t_zero_shot_base if isinstance(self.t_zero_shot_base, torch.Tensor) else self.t_raw_pooled
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
            cache_info = dict(self.last_cache_eval_info)
            dualtrack_info = dict(self.last_dualtrack_eval_info)

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
                "Line": "capav1_gt",
                "Dataset": d_name,
                "Param_Profile": self.config.PARAM_PROFILE,
                "Label_Space": self.config.LABEL_SPACE,
                "ScoringMode": mode,
                "Calibrator": "Temp (Scalar)",
                "Scale": eval_scale,
                "Tau": tau_cal,
                "Init_Temperature": float(self.config.INIT_TEMPERATURE),
                "N_Min_Support": int(self.config.N_MIN_SUPPORT_FOR_ACTIVE),
                "Min_Classes_Adapt": int(self.config.MIN_CLASSES_FOR_ADAPTATION),
                "Cache_Mode": str(getattr(self.config, "CACHE_MODE", "off")),
                "Cache_Alpha_Max": float(self.config.CACHE_ALPHA_MAX),
                "Cache_TopK": int(self.config.CACHE_TOPK),
                "Cache_Temp": float(self.config.CACHE_TEMP),
                "Structured_Prompt_Bank": bool(getattr(self.config, "ENABLE_STRUCTURED_PROMPT_BANK", False)),
                "Prompt_Bank_Profile": str(getattr(self.config, "PROMPT_BANK_PROFILE", "v2")),
                "Prompt_Pooling_Mode": str(getattr(self.config, "PROMPT_POOLING_MODE", "mean")),
                "Prompt_Legacy_Mix": float(getattr(self.config, "PROMPT_LEGACY_MIX", 0.0)),
                "Prompt_Class_Mix_Profile": str(getattr(self.config, "PROMPT_CLASS_MIX_PROFILE", "none")),
                "Early_Text_Prompt_Support": bool(getattr(self.config, "ENABLE_EARLY_TEXT_PROMPT_SUPPORT", True))
                and bool(getattr(self.config, "PROMPT_TEXT_EMBEDDING_GROUPS", None)),
                "Early_Text_Prompt_TopK": int(getattr(self.config, "EARLY_TEXT_PROMPT_TOP_K", 12)),
                "Early_Text_Prompt_Entry_Mode": str(getattr(self.config, "EARLY_TEXT_PROMPT_ENTRY_MODE", "full")),
                "Early_Text_Prompt_Selection": str(getattr(self.config, "EARLY_TEXT_PROMPT_SELECTION_MODE", "margin")),
                "Guarded_Slerp": bool(getattr(self.config, "ENABLE_CAPAV1_GUARDED_SLERP", False)),
                "Guarded_Slerp_Lambda_Max": float(getattr(self.config, "CAPAV1_GUARDED_SLERP_LAMBDA_MAX", 0.0)),
                "GO_Stage2": bool(getattr(self.config, "ENABLE_GO_GUARDIAN_STAGE2", False)),
                "Prompt_Coreset": bool(getattr(self.config, "ENABLE_PROMPT_CORESET", False)),
                "Prompt_Coreset_Size": int(getattr(self.config, "PROMPT_CORESET_SIZE", self.config.M)),
                "Prompt_Bucket_Keep": int(getattr(self.config, "PROMPT_BUCKET_KEEP", 2)),
                "DualTrack_Abstain": bool(getattr(self.config, "CAPAV1_DUALTRACK_ENABLE_ABSTAIN", False)),
                "DualTrack_Abstain_Rate": dualtrack_info.get("abstain_rate", 0.0),
                "DualTrack_Aligned_Rate": dualtrack_info.get("aligned_rate", np.nan),
                "DualTrack_Agree_Rate": dualtrack_info.get("agree_rate", np.nan),
                "Cache_Dataset_Gate": bool(cache_info.get("dataset_gate", False)),
                "Cache_PSI_Top1": cache_info.get("psi_top1", np.nan),
                "Cache_PSI_TopKMean": cache_info.get("psi_topk_mean", np.nan),
                "Cache_PSI_Entropy": cache_info.get("psi_entropy", np.nan),
                "Cache_Usage_Rate": cache_info.get("usage_rate", 0.0),
                "Cache_Mean_Alpha": cache_info.get("mean_alpha", 0.0),
                "Cache_Agree_Rate": cache_info.get("agree_rate", np.nan),
                "Cache_Mean_Top1Sim": cache_info.get("mean_top1_sim", np.nan),
                "Cache_Mean_Purity": cache_info.get("mean_purity", np.nan),
                "Cache_Mean_Entropy": cache_info.get("mean_entropy", np.nan),
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

    # === 鏂板杈呭姪鍑芥暟锛歅SI 璁＄畻 ===
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
        
        # 鍔犺浇鍐荤粨鐘舵€?
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
            
            # 2. Dynamic Config (鍏抽敭淇敼: 閽堝灏忔暟鎹泦鍑忓皯鍒嗘《鏁?
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
            
            # A. 鏀堕泦缃俊搴?
            all_confidences = []
            BATCH_SIZE = 256
            for i in range(0, len(z_test), BATCH_SIZE):
                z_b = z_test[i:i+BATCH_SIZE]
                logits = scale * torch.matmul(z_b, t_capa.T) + self.b_c
                probs = torch.sigmoid(logits / self.T_opt)
                confs = probs.max(dim=1).values.detach().cpu().numpy()
                all_confidences.extend(confs)
            
            all_confidences = np.array(all_confidences)
            
            # B. 寤虹珛鍩哄噯
            n_init_samples = WINDOW_SIZE * INIT_WINDOWS
            if len(all_confidences) < n_init_samples + WINDOW_SIZE:
                print(f"   [Skip] Not enough data.")
                continue
                
            baseline_data = all_confidences[:n_init_samples]
            monitor_data = all_confidences[n_init_samples:]
            
            # C. 鐩戞帶
            psi_history = []
            n_windows = int(np.ceil(len(monitor_data) / WINDOW_SIZE))
            
            for i in range(n_windows):
                start = i * WINDOW_SIZE
                end = min((i + 1) * WINDOW_SIZE, len(monitor_data))
                if end - start < 10: break 
                
                curr_window_data = monitor_data[start:end]
                
                # 浣跨敤鍔ㄦ€佽瀹氱殑 bin 鏁伴噺
                psi_val = self._compute_psi(baseline_data, curr_window_data, n_bins=PSI_BINS)
                psi_history.append(psi_val)
            
            if not psi_history:
                print("   [Skip] No valid windows.")
                continue

            # D. 缁熻
            psi_arr = np.array(psi_history)
            psi_min = psi_arr.min()
            psi_med = np.median(psi_arr)
            psi_max = psi_arr.max()
            
            # 绋嶅井鏀惧灏忔牱鏈殑 Max 闃堝€?(Median 鎵嶆槸鍏抽敭)
            threshold = 0.4 if n_samples < 2000 else 0.25
            status = "Stable" if psi_med < 0.25 else "Drift" # 涓昏鐪嬩腑浣嶆暟
            
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
                # Legacy experimental branch: intentionally rebinds a test file as
                # train/calib/tau via split_override. This does NOT follow the main
                # dataset semantics used by eval_mode_report / eval_mode_comparison.
                cfg_ds.TRAIN_DATA_PATH = path
                cfg_ds.CALIB_DATA_PATH = path
                cfg_ds.TAU_CALIB_DATA_PATH = path
                cfg_ds.SAVE_DIR = os.path.join(self.config.SAVE_DIR, f"per_dataset_capa_{key.lower()}")
                os.makedirs(cfg_ds.SAVE_DIR, exist_ok=True)
                self._log(
                    f"[PerDatasetCAPA][Legacy] Rebinding {path} as train/calib/tau for diagnostic use only.",
                    always=True,
                )

                runner_ds = CAPA5NotebookRunner(cfg_ds)
                runner_ds._build_prototypes()
                if runner_ds.support_counts is None:
                    runner_ds.support_counts = torch.ones(
                        len(runner_ds.config.ORDERED_CLASS_NAMES),
                        device=runner_ds.device,
                    ) * runner_ds.config.N_MIN_SUPPORT_FOR_ACTIVE
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
            "Dataset    螖AUC(mixed)  螖AUC(softmax)  softmax-mixed  "
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

    def _run_evaluation(self):
        gate_pass = bool(self.final_alignment_stats.get("gate_pass", False)) if isinstance(self.final_alignment_stats, dict) else False
        soft_fallback = bool(self.final_alignment_stats.get("soft_fallback", False)) if isinstance(self.final_alignment_stats, dict) else False
        if gate_pass or self.is_frozen:
            status = "ACCEPTED (Frozen R*)"
        elif soft_fallback:
            status = "SOFT-FALLBACK (Guarded R)"
        elif isinstance(self.R_frozen, torch.Tensor):
            I_eval = torch.eye(self.R_frozen.shape[0], device=self.R_frozen.device, dtype=self.R_frozen.dtype)
            if torch.norm(self.R_frozen - I_eval).item() > 1e-3:
                status = "UPDATED (Unfrozen R)"
            else:
                status = "REJECTED (Identity)"
        else:
            status = "REJECTED (Identity)"
        self._log(f"\n[Stage IV] Evaluation ({status})")
        
        t_base = self.t_zero_shot_base if isinstance(self.t_zero_shot_base, torch.Tensor) else self.t_raw_pooled
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
                "Line": "capav1_gt",
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

def _parse_float_csv(text: Optional[str]) -> Optional[List[float]]:
    if text is None:
        return None
    items: List[float] = []
    for raw in str(text).split(","):
        token = raw.strip()
        if not token:
            continue
        items.append(float(token))
    return items if items else None


def run_prompt_stage_isolation_analysis(config: CAPA5Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = os.path.join(config.SAVE_DIR, "prompt_stage_isolation")
    os.makedirs(out_dir, exist_ok=True)

    legacy_cfg = copy.deepcopy(config)
    legacy_cfg.DEBUG = False
    legacy_cfg.VERBOSE = False
    legacy_cfg.PRINT_SUMMARY = False
    legacy_cfg.CACHE_MODE = "off"
    legacy_cfg.ENABLE_STRUCTURED_PROMPT_BANK = False
    legacy_cfg.PROMPT_BANK_PROFILE = "v3"
    legacy_cfg.PROMPT_POOLING_MODE = "mean"
    legacy_cfg.PROMPT_LEGACY_MIX = 0.0
    legacy_cfg.PROMPT_CLASS_MIX_PROFILE = "none"
    legacy_cfg.ENABLE_PROMPT_CORESET = False
    legacy_cfg.SAVE_DIR = os.path.join(out_dir, "legacy")

    structured_cfg = copy.deepcopy(config)
    structured_cfg.DEBUG = False
    structured_cfg.VERBOSE = False
    structured_cfg.PRINT_SUMMARY = False
    structured_cfg.CACHE_MODE = "off"
    structured_cfg.SAVE_DIR = os.path.join(out_dir, "structured")

    legacy_runner = CAPA5NotebookRunner(legacy_cfg)
    legacy_runner.run_pipeline()
    structured_runner = CAPA5NotebookRunner(structured_cfg)
    structured_runner.run_pipeline()

    legacy_raw_t = legacy_runner.t_zero_shot_base if isinstance(legacy_runner.t_zero_shot_base, torch.Tensor) else legacy_runner.t_raw_pooled
    legacy_full_t = legacy_runner._l2_norm(torch.matmul(legacy_runner._get_alignment_text_base(), legacy_runner.R_frozen.T))
    structured_raw_t = structured_runner.t_raw_pooled.clone()
    structured_fixed_t, fixed_stats = structured_runner._build_fixed_alignment_with_external_state(
        centroids=legacy_runner.image_centroids,
        support_counts=legacy_runner.support_counts,
        prior_counts=legacy_runner.prior_counts,
        rejected_counts=legacy_runner.rejected_counts,
    )
    structured_full_t = structured_runner._l2_norm(torch.matmul(structured_runner._get_alignment_text_base(), structured_runner.R_frozen.T))

    angle_raw = legacy_runner._prototype_angle_deg(legacy_raw_t, structured_raw_t)
    angle_fixed = legacy_runner._prototype_angle_deg(legacy_full_t, structured_fixed_t)
    angle_full = legacy_runner._prototype_angle_deg(legacy_full_t, structured_full_t)

    summary_rows: List[Dict[str, object]] = []
    mimic_rows: List[Dict[str, object]] = []

    for d_name, path in config.TEST_DATA_PATHS.items():
        if not os.path.exists(path):
            continue

        z_legacy_raw, y_legacy, is_multi_legacy = legacy_runner._load_data(path, split_override=2)
        if y_legacy is None:
            continue
        z_legacy = legacy_runner._apply_preprocessing(z_legacy_raw, legacy_runner.zI_mean)

        z_struct_raw, y_struct, is_multi_struct = structured_runner._load_data(path, split_override=2)
        z_struct = structured_runner._apply_preprocessing(z_struct_raw, structured_runner.zI_mean)

        stats_legacy_raw, probs_legacy_raw = legacy_runner._evaluate_prompt_stage(
            z_legacy, y_legacy, is_multi_legacy, legacy_raw_t, dataset_name=d_name, prior_counts=None
        )
        stats_legacy_full, probs_legacy_full = legacy_runner._evaluate_prompt_stage(
            z_legacy,
            y_legacy,
            is_multi_legacy,
            legacy_full_t,
            dataset_name=d_name,
            prior_counts=legacy_runner.prior_counts,
        )
        stats_struct_raw, probs_struct_raw = structured_runner._evaluate_prompt_stage(
            z_struct, y_struct, is_multi_struct, structured_raw_t, dataset_name=d_name, prior_counts=None
        )
        stats_struct_fixed, probs_struct_fixed = structured_runner._evaluate_prompt_stage(
            z_struct,
            y_struct,
            is_multi_struct,
            structured_fixed_t,
            dataset_name=d_name,
            prior_counts=legacy_runner.prior_counts,
        )
        stats_struct_full, probs_struct_full = structured_runner._evaluate_prompt_stage(
            z_struct,
            y_struct,
            is_multi_struct,
            structured_full_t,
            dataset_name=d_name,
            prior_counts=structured_runner.prior_counts,
        )

        summary_rows.append(
            {
                "Dataset": d_name,
                "Legacy_Raw_AUC_Macro": float(stats_legacy_raw.get("Macro-AUC", np.nan)),
                "Structured_Raw_AUC_Macro": float(stats_struct_raw.get("Macro-AUC", np.nan)),
                "Delta_Raw_vs_Legacy_Raw": float(stats_struct_raw.get("Macro-AUC", np.nan) - stats_legacy_raw.get("Macro-AUC", np.nan)),
                "Legacy_Full_AUC_Macro": float(stats_legacy_full.get("Macro-AUC", np.nan)),
                "Structured_Fixed_AUC_Macro": float(stats_struct_fixed.get("Macro-AUC", np.nan)),
                "Delta_Fixed_vs_Legacy_Full": float(stats_struct_fixed.get("Macro-AUC", np.nan) - stats_legacy_full.get("Macro-AUC", np.nan)),
                "Structured_Full_AUC_Macro": float(stats_struct_full.get("Macro-AUC", np.nan)),
                "Delta_Full_vs_Legacy_Full": float(stats_struct_full.get("Macro-AUC", np.nan) - stats_legacy_full.get("Macro-AUC", np.nan)),
                "Fixed_Selection_Alpha": float(fixed_stats.get("alpha", 1.0)),
                "Fixed_Selection_Mode": str(fixed_stats.get("selection_mode", "")),
                "Fixed_Selection_Base": str(fixed_stats.get("mix_base", "")),
            }
        )

        if legacy_runner._canonical_dataset_name(d_name) == "MIMIC":
            raw_legacy_rows = legacy_runner._per_class_auc_rows(y_legacy, probs_legacy_raw, dataset_name=d_name, stage_name="legacy_raw")
            raw_struct_rows = structured_runner._per_class_auc_rows(y_struct, probs_struct_raw, dataset_name=d_name, stage_name="structured_raw")
            full_legacy_rows = legacy_runner._per_class_auc_rows(y_legacy, probs_legacy_full, dataset_name=d_name, stage_name="legacy_full")
            fixed_struct_rows = structured_runner._per_class_auc_rows(y_struct, probs_struct_fixed, dataset_name=d_name, stage_name="structured_fixed")
            full_struct_rows = structured_runner._per_class_auc_rows(y_struct, probs_struct_full, dataset_name=d_name, stage_name="structured_full")

            stage_maps: Dict[str, Dict[str, float]] = {}
            for bundle in [raw_legacy_rows, raw_struct_rows, full_legacy_rows, fixed_struct_rows, full_struct_rows]:
                for row in bundle:
                    cls = str(row["Class"])
                    stage = str(row["Stage"])
                    stage_maps.setdefault(cls, {})[stage] = float(row.get("AUC", np.nan))

            for idx, cls_name in enumerate(legacy_runner.config.ORDERED_CLASS_NAMES):
                row_map = stage_maps.get(str(cls_name), {})
                auc_legacy_raw = float(row_map.get("legacy_raw", np.nan))
                auc_struct_raw = float(row_map.get("structured_raw", np.nan))
                auc_legacy_full = float(row_map.get("legacy_full", np.nan))
                auc_struct_fixed = float(row_map.get("structured_fixed", np.nan))
                auc_struct_full = float(row_map.get("structured_full", np.nan))
                mimic_rows.append(
                    {
                        "Dataset": d_name,
                        "Class": str(cls_name),
                        "AUC_Legacy_Raw": auc_legacy_raw,
                        "AUC_Structured_Raw": auc_struct_raw,
                        "Delta_Raw_vs_Legacy_Raw": auc_struct_raw - auc_legacy_raw if np.isfinite(auc_struct_raw) and np.isfinite(auc_legacy_raw) else np.nan,
                        "AUC_Legacy_Full": auc_legacy_full,
                        "AUC_Structured_Fixed": auc_struct_fixed,
                        "Delta_Fixed_vs_Legacy_Full": auc_struct_fixed - auc_legacy_full if np.isfinite(auc_struct_fixed) and np.isfinite(auc_legacy_full) else np.nan,
                        "AUC_Structured_Full": auc_struct_full,
                        "Delta_Full_vs_Legacy_Full": auc_struct_full - auc_legacy_full if np.isfinite(auc_struct_full) and np.isfinite(auc_legacy_full) else np.nan,
                        "Angle_Raw_vs_Legacy_Raw_Deg": float(angle_raw[idx]) if idx < len(angle_raw) else np.nan,
                        "Angle_Fixed_vs_Legacy_Full_Deg": float(angle_fixed[idx]) if idx < len(angle_fixed) else np.nan,
                        "Angle_Full_vs_Legacy_Full_Deg": float(angle_full[idx]) if idx < len(angle_full) else np.nan,
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    mimic_df = pd.DataFrame(mimic_rows)
    summary_path = os.path.join(out_dir, "stage_isolation_summary.csv")
    mimic_path = os.path.join(out_dir, "mimic_per_class_stage_deltas.csv")
    summary_df.to_csv(summary_path, index=False)
    mimic_df.to_csv(mimic_path, index=False)
    print(f"[Saved] {summary_path}")
    print(f"[Saved] {mimic_path}")
    return summary_df, mimic_df


def _build_site_expert_branch_config(config: CAPA5Config, branch_name: str, save_dir: str) -> CAPA5Config:
    cfg = copy.deepcopy(config)
    cfg.DEBUG = False
    cfg.VERBOSE = False
    cfg.PRINT_SUMMARY = False
    cfg.CACHE_MODE = "off"
    cfg.SAVE_DIR = save_dir

    if branch_name == "legacy_safe":
        cfg.ENABLE_STRUCTURED_PROMPT_BANK = False
        cfg.PROMPT_BANK_PROFILE = "v3"
        cfg.PROMPT_BUCKET_PRIORS = dict(PROMPT_BANK_PROFILE_PRIORS.get(cfg.PROMPT_BANK_PROFILE, DEFAULT_PROMPT_BUCKET_PRIORS))
        cfg.PROMPT_POOLING_MODE = "mean"
        cfg.ENABLE_PROMPT_CORESET = False
        cfg.PROMPT_LEGACY_MIX = 0.0
        cfg.PROMPT_CLASS_MIX_PROFILE = "none"
        return cfg

    if branch_name == "structured_visual":
        cfg.ENABLE_STRUCTURED_PROMPT_BANK = True
        cfg.PROMPT_BANK_PROFILE = "visual"
        cfg.PROMPT_BUCKET_PRIORS = dict(PROMPT_BANK_PROFILE_PRIORS.get(cfg.PROMPT_BANK_PROFILE, DEFAULT_PROMPT_BUCKET_PRIORS))
        cfg.PROMPT_POOLING_MODE = "bucketed"
        cfg.ENABLE_PROMPT_CORESET = True
        cfg.PROMPT_BUCKET_KEEP = 1
        cfg.PROMPT_CORESET_SIZE = max(int(cfg.M), int(cfg.PROMPT_CORESET_SIZE))
        cfg.PROMPT_LEGACY_MIX = 0.30
        cfg.PROMPT_CLASS_MIX_PROFILE = "none"
        return cfg

    if branch_name == "report_surface":
        cfg.ENABLE_STRUCTURED_PROMPT_BANK = True
        cfg.PROMPT_BANK_PROFILE = "report"
        cfg.PROMPT_BUCKET_PRIORS = dict(PROMPT_BANK_PROFILE_PRIORS.get(cfg.PROMPT_BANK_PROFILE, DEFAULT_PROMPT_BUCKET_PRIORS))
        cfg.PROMPT_POOLING_MODE = "bucketed"
        cfg.ENABLE_PROMPT_CORESET = True
        cfg.PROMPT_BUCKET_KEEP = 1
        cfg.PROMPT_CORESET_SIZE = max(int(cfg.M), int(cfg.PROMPT_CORESET_SIZE))
        cfg.PROMPT_LEGACY_MIX = 0.18
        cfg.PROMPT_CLASS_MIX_PROFILE = "none"
        return cfg

    if branch_name == "mimic_report_hybrid":
        cfg.ENABLE_STRUCTURED_PROMPT_BANK = True
        cfg.PROMPT_BANK_PROFILE = "report"
        cfg.PROMPT_BUCKET_PRIORS = dict(PROMPT_BANK_PROFILE_PRIORS.get(cfg.PROMPT_BANK_PROFILE, DEFAULT_PROMPT_BUCKET_PRIORS))
        cfg.PROMPT_POOLING_MODE = "bucketed"
        cfg.ENABLE_PROMPT_CORESET = True
        cfg.PROMPT_BUCKET_KEEP = 1
        cfg.PROMPT_CORESET_SIZE = max(int(cfg.M), int(cfg.PROMPT_CORESET_SIZE))
        cfg.PROMPT_LEGACY_MIX = 0.25
        cfg.PROMPT_CLASS_MIX_PROFILE = "mimic_hybrid"
        return cfg

    raise ValueError(f"Unsupported site expert branch: {branch_name}")


def _prepare_site_expert_runner(cfg: CAPA5Config) -> Tuple[CAPA5NotebookRunner, List[Dict[str, object]]]:
    runner = CAPA5NotebookRunner(cfg)
    runner._build_prototypes()
    if runner.support_counts is None:
        runner.support_counts = torch.ones(len(runner.config.ORDERED_CLASS_NAMES), device=runner.device) * runner.config.N_MIN_SUPPORT_FOR_ACTIVE
    if runner.image_centroids is None:
        runner.image_centroids = runner.t_raw_pooled.clone()
    if runner.current_R is None:
        runner.current_R = torch.eye(runner.t_raw_pooled.shape[1], device=runner.device)
    runner.run_pipeline()
    rows = runner.run_manuscript_validation(scoring_mode=cfg.SCORING_MODE, sim_source=cfg.SIM_SOURCE)
    return runner, rows


def _site_expert_row_from_payload(
    payload: Dict[str, object],
    *,
    dataset_name: str,
    strategy: str,
    branch_name: str,
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    stats_base = dict(payload.get("stats_base", {}))
    stats_capa = dict(payload.get("stats_capa", {}))
    row = {
        "Dataset": dataset_name,
        "Strategy": strategy,
        "Branch": branch_name,
        "AUC_Baseline_ZeroShot_Macro": float(stats_base.get("Macro-AUC", np.nan)),
        "AUC_CAPA_Aligned_Macro": float(stats_capa.get("Macro-AUC", np.nan)),
        "Delta_AUC_CAPA_minus_Baseline_Macro": float(stats_capa.get("Macro-AUC", np.nan)) - float(stats_base.get("Macro-AUC", np.nan)),
        "ECE_Baseline": float(stats_base.get("ECE", np.nan)),
        "ECE_CAPA": float(stats_capa.get("ECE", np.nan)),
        "Top1_Median_CAPA": float(stats_capa.get("Top1_Median", np.nan)),
    }
    if extra:
        row.update(extra)
    return row


def _select_branch_outputs(
    base_runner: CAPA5NotebookRunner,
    primary: Dict[str, object],
    secondary: Dict[str, object],
    *,
    primary_name: str,
    secondary_name: str,
    conf_margin: float,
    low_conf_thr: float,
    fallback_name: str,
) -> Dict[str, object]:
    y_primary = np.asarray(primary["y"])
    y_secondary = np.asarray(secondary["y"])
    if y_primary.shape != y_secondary.shape or np.any(y_primary != y_secondary):
        raise RuntimeError("Branch outputs use mismatched sample order or labels; cannot route sample-wise.")

    conf_primary = np.asarray(primary["confidence"], dtype=np.float64).reshape(-1)
    conf_secondary = np.asarray(secondary["confidence"], dtype=np.float64).reshape(-1)
    n = min(len(conf_primary), len(conf_secondary))
    conf_primary = conf_primary[:n]
    conf_secondary = conf_secondary[:n]
    y_eval = y_primary[:n]

    choose_primary = conf_primary >= (conf_secondary + float(conf_margin))
    low_conf = np.maximum(conf_primary, conf_secondary) < float(low_conf_thr)
    if str(fallback_name) == secondary_name:
        choose_primary[low_conf] = False
    else:
        choose_primary[low_conf] = True

    is_multi = bool(primary.get("is_multi", False))
    rank_primary = np.asarray(primary["probs_capa_rank"])
    cal_primary = np.asarray(primary["probs_capa_cal"])
    rank_secondary = np.asarray(secondary["probs_capa_rank"])
    cal_secondary = np.asarray(secondary["probs_capa_cal"])
    if is_multi:
        rank_selected = np.where(choose_primary[:, None], rank_primary[:n], rank_secondary[:n])
        cal_selected = np.where(choose_primary[:, None], cal_primary[:n], cal_secondary[:n])
    else:
        rank_selected = np.where(choose_primary, rank_primary[:n], rank_secondary[:n])
        cal_selected = np.where(choose_primary, cal_primary[:n], cal_secondary[:n])

    stats_sel = base_runner._compute_metrics_from_prob_arrays(
        y_eval,
        rank_selected,
        cal_selected,
        is_multi=is_multi,
    )
    stats_base = dict(primary.get("stats_base", secondary.get("stats_base", {})))
    return {
        "y": y_eval,
        "is_multi": is_multi,
        "stats_base": stats_base,
        "stats_capa": stats_sel,
        "primary_usage_rate": float(np.mean(choose_primary.astype(np.float32))),
        "fallback_rate": float(np.mean(low_conf.astype(np.float32))),
        "primary_conf_mean": float(np.mean(conf_primary)),
        "secondary_conf_mean": float(np.mean(conf_secondary)),
    }


def run_site_expert_branch_analysis(config: CAPA5Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = os.path.join(config.SAVE_DIR, "site_expert")
    os.makedirs(out_dir, exist_ok=True)

    branch_names = [
        "legacy_safe",
        "structured_visual",
        "report_surface",
        "mimic_report_hybrid",
    ]
    branch_runners: Dict[str, CAPA5NotebookRunner] = {}
    branch_rows: Dict[str, List[Dict[str, object]]] = {}
    for branch_name in branch_names:
        cfg_branch = _build_site_expert_branch_config(
            config,
            branch_name=branch_name,
            save_dir=os.path.join(out_dir, branch_name),
        )
        runner_branch, rows_branch = _prepare_site_expert_runner(cfg_branch)
        branch_runners[branch_name] = runner_branch
        branch_rows[branch_name] = rows_branch

    branch_summary_rows: List[Dict[str, object]] = []
    for branch_name, rows in branch_rows.items():
        for row in rows:
            branch_summary_rows.append(
                {
                    "Dataset": str(row.get("Dataset", "")),
                    "Strategy": "branch",
                    "Branch": branch_name,
                    "AUC_Baseline_ZeroShot_Macro": float(row.get("AUC_Baseline_ZeroShot_Macro", np.nan)),
                    "AUC_CAPA_Aligned_Macro": float(row.get("AUC_CAPA_Aligned_Macro", np.nan)),
                    "Delta_AUC_CAPA_minus_Baseline_Macro": float(row.get("Delta_AUC_CAPA_minus_Baseline_Macro", np.nan)),
                    "ECE_CAPA": float(row.get("ECE_Post", np.nan)),
                }
            )

    dynamic_rows: List[Dict[str, object]] = []
    low_conf_thr = float(getattr(config, "SITE_EXPERT_UNKNOWN_LOW_CONF", 0.60))
    conf_margin = float(getattr(config, "SITE_EXPERT_UNKNOWN_CONF_MARGIN", 0.0))
    fixed_map = {
        "CheXpert": "structured_visual",
        "MIMIC": "mimic_report_hybrid",
        "COVID": "structured_visual",
        "RSNA": "structured_visual",
    }

    for d_name, path in config.TEST_DATA_PATHS.items():
        if not os.path.exists(path):
            continue
        legacy_payload = branch_runners["legacy_safe"]._predict_dataset_branch_outputs(d_name, path, scoring_mode=config.SCORING_MODE)
        visual_payload = branch_runners["structured_visual"]._predict_dataset_branch_outputs(d_name, path, scoring_mode=config.SCORING_MODE)
        report_payload = branch_runners["report_surface"]._predict_dataset_branch_outputs(d_name, path, scoring_mode=config.SCORING_MODE)
        mimic_payload = branch_runners["mimic_report_hybrid"]._predict_dataset_branch_outputs(d_name, path, scoring_mode=config.SCORING_MODE)

        fixed_branch = fixed_map.get(d_name, "structured_visual")
        fixed_payload = {
            "legacy_safe": legacy_payload,
            "structured_visual": visual_payload,
            "report_surface": report_payload,
            "mimic_report_hybrid": mimic_payload,
        }[fixed_branch]
        dynamic_rows.append(
            _site_expert_row_from_payload(
                fixed_payload,
                dataset_name=d_name,
                strategy="site_fixed",
                branch_name=fixed_branch,
            )
        )

        unknown_lv = _select_branch_outputs(
            branch_runners["legacy_safe"],
            visual_payload,
            legacy_payload,
            primary_name="structured_visual",
            secondary_name="legacy_safe",
            conf_margin=conf_margin,
            low_conf_thr=low_conf_thr,
            fallback_name="legacy_safe",
        )
        dynamic_rows.append(
            _site_expert_row_from_payload(
                unknown_lv,
                dataset_name=d_name,
                strategy="site_unknown_legacy_visual",
                branch_name="structured_visual_vs_legacy_safe",
                extra={
                    "Primary_Usage_Rate": float(unknown_lv["primary_usage_rate"]),
                    "LowConf_Fallback_Rate": float(unknown_lv["fallback_rate"]),
                    "Primary_Conf_Mean": float(unknown_lv["primary_conf_mean"]),
                    "Secondary_Conf_Mean": float(unknown_lv["secondary_conf_mean"]),
                },
            )
        )

        semantic_vr = _select_branch_outputs(
            branch_runners["legacy_safe"],
            visual_payload,
            report_payload,
            primary_name="structured_visual",
            secondary_name="report_surface",
            conf_margin=conf_margin,
            low_conf_thr=low_conf_thr,
            fallback_name="report_surface",
        )
        dynamic_rows.append(
            _site_expert_row_from_payload(
                semantic_vr,
                dataset_name=d_name,
                strategy="site_unknown_visual_report",
                branch_name="structured_visual_vs_report_surface",
                extra={
                    "Primary_Usage_Rate": float(semantic_vr["primary_usage_rate"]),
                    "LowConf_Fallback_Rate": float(semantic_vr["fallback_rate"]),
                    "Primary_Conf_Mean": float(semantic_vr["primary_conf_mean"]),
                    "Secondary_Conf_Mean": float(semantic_vr["secondary_conf_mean"]),
                },
            )
        )

    df_branch = pd.DataFrame(branch_summary_rows)
    df_dynamic = pd.DataFrame(dynamic_rows)
    branch_path = os.path.join(out_dir, "site_expert_branch_summary.csv")
    dynamic_path = os.path.join(out_dir, "site_expert_strategy_summary.csv")
    df_branch.to_csv(branch_path, index=False)
    df_dynamic.to_csv(dynamic_path, index=False)
    print(f"[SiteExpert] saved branch summary -> {branch_path}")
    print(f"[SiteExpert] saved strategy summary -> {dynamic_path}")
    return df_branch, df_dynamic


def run_go_ml_risk_stratified_analysis(config: CAPA5Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = os.path.join(config.SAVE_DIR, "go_ml_risk_analysis")
    os.makedirs(out_dir, exist_ok=True)

    base_cfg = copy.deepcopy(config)
    base_cfg.DEBUG = False
    base_cfg.VERBOSE = False
    base_cfg.PRINT_SUMMARY = False
    base_cfg.CACHE_MODE = "off"
    base_cfg.GO_ML_ROBUST_MODE = "none"
    base_cfg.SAVE_DIR = os.path.join(out_dir, "baseline_residual")

    robust_cfg = copy.deepcopy(config)
    robust_cfg.DEBUG = False
    robust_cfg.VERBOSE = False
    robust_cfg.PRINT_SUMMARY = False
    robust_cfg.CACHE_MODE = "off"
    robust_cfg.GO_ML_ROBUST_MODE = "huber"
    robust_cfg.GO_ML_HUBER_SCOPE = "conditional"
    robust_cfg.SAVE_DIR = os.path.join(out_dir, "conditional_huber")

    base_runner, _ = _prepare_site_expert_runner(base_cfg)
    robust_runner, _ = _prepare_site_expert_runner(robust_cfg)

    rows: List[Dict[str, object]] = []
    thresh_rows: List[Dict[str, object]] = []

    analysis_items: List[Tuple[str, str, int]] = [(d_name, path, 2) for d_name, path in config.TEST_DATA_PATHS.items()]
    analysis_items.append(("AdaptTrain", config.TRAIN_DATA_PATH, 0))

    for d_name, path, split_override in analysis_items:
        if not os.path.exists(path):
            continue
        payload_base = base_runner._predict_dataset_branch_outputs(
            d_name,
            path,
            scoring_mode=config.SCORING_MODE,
            split_override=split_override,
        )
        payload_robust = robust_runner._predict_dataset_branch_outputs(
            d_name,
            path,
            scoring_mode=config.SCORING_MODE,
            split_override=split_override,
        )
        is_multi = bool(payload_base.get("is_multi", False))

        z_raw, y_eval, _ = base_runner._load_data(path, split_override=split_override)
        z_proc = base_runner._apply_preprocessing(z_raw, base_runner.zI_mean)
        g_cpu = torch.Generator()
        g_cpu.manual_seed(base_runner.config.RANDOM_SEED)
        perm = torch.randperm(len(z_proc), generator=g_cpu)
        z_proc = z_proc[perm]
        if isinstance(y_eval, np.ndarray):
            y_eval = y_eval[perm.cpu().numpy()]
        y_use = payload_base["y"]
        n = min(int(z_proc.shape[0]), int(np.asarray(y_use).shape[0]))
        z_proc = z_proc[:n]
        if isinstance(y_use, np.ndarray):
            y_use = y_use[:n]

        t_base_eval = base_runner._l2_norm(torch.matmul(base_runner._get_alignment_text_base(), base_runner.R_frozen.T))
        risk_df = base_runner._compute_eval_go_ml_risk_features(z_proc, y_use, t_base_eval)
        risk_df = risk_df.iloc[:n].copy()
        risk_df["trigger_candidate"] = risk_df.apply(
            lambda row: robust_runner._should_use_conditional_huber(
                active_count=int(row.get("active_count", 0)),
                resid_meta={
                    "n_other": max(0, int(row.get("active_count", 0)) - 1),
                    "cond_reg": float(row.get("cond_reg_max", np.nan)),
                    "resid_norm_ratio": float(row.get("resid_ratio_max", np.nan)),
                    "max_other_sim": float(row.get("max_other_sim", np.nan)),
                },
                step=None,
            ),
            axis=1,
        )

        base_rank = payload_base["probs_capa_rank"]
        robust_rank = payload_robust["probs_capa_rank"]

        overall_base = base_runner._macro_auc_subset(y_use, base_rank, is_multi=is_multi)
        overall_robust = base_runner._macro_auc_subset(y_use, robust_rank, is_multi=is_multi)
        rows.append(
            {
                "Dataset": d_name,
                "Stratum_Group": "overall",
                "Stratum_Name": "all",
                "N_Samples": int(n),
                "Trigger_Rate": float(risk_df["trigger_candidate"].mean()) if len(risk_df) > 0 else np.nan,
                "Base_AUC_Macro": overall_base,
                "ConditionalHuber_AUC_Macro": overall_robust,
                "Delta_CondHuber_minus_Base": overall_robust - overall_base if np.isfinite(overall_base) and np.isfinite(overall_robust) else np.nan,
            }
        )

        if is_multi:
            card_masks = {
                "single": (risk_df["active_count"] == 1).to_numpy(),
                "dual": (risk_df["active_count"] == 2).to_numpy(),
                "triple_plus": (risk_df["active_count"] >= 3).to_numpy(),
            }
            for name, mask in card_masks.items():
                if int(np.sum(mask)) <= 8:
                    continue
                auc_b = base_runner._macro_auc_subset(y_use, base_rank, is_multi=True, sample_mask=mask)
                auc_r = base_runner._macro_auc_subset(y_use, robust_rank, is_multi=True, sample_mask=mask)
                rows.append(
                    {
                        "Dataset": d_name,
                        "Stratum_Group": "label_cardinality",
                        "Stratum_Name": name,
                        "N_Samples": int(np.sum(mask)),
                        "Trigger_Rate": float(risk_df.loc[mask, "trigger_candidate"].mean()) if int(np.sum(mask)) > 0 else np.nan,
                        "Base_AUC_Macro": auc_b,
                        "ConditionalHuber_AUC_Macro": auc_r,
                        "Delta_CondHuber_minus_Base": auc_r - auc_b if np.isfinite(auc_b) and np.isfinite(auc_r) else np.nan,
                    }
                )

            cond_vals = risk_df.loc[(risk_df["active_count"] >= 2) & np.isfinite(risk_df["cond_reg_max"]), "cond_reg_max"].to_numpy()
            if cond_vals.size > 0:
                cond_thr = float(np.median(cond_vals))
                thresh_rows.append({"Dataset": d_name, "Metric": "cond_reg_max", "Threshold": cond_thr})
                low_mask = ((risk_df["active_count"] >= 2) & (risk_df["cond_reg_max"] < cond_thr)).to_numpy()
                high_mask = ((risk_df["active_count"] >= 2) & (risk_df["cond_reg_max"] >= cond_thr)).to_numpy()
                for name, mask in [("low", low_mask), ("high", high_mask)]:
                    if int(np.sum(mask)) <= 8:
                        continue
                    auc_b = base_runner._macro_auc_subset(y_use, base_rank, is_multi=True, sample_mask=mask)
                    auc_r = base_runner._macro_auc_subset(y_use, robust_rank, is_multi=True, sample_mask=mask)
                    rows.append(
                        {
                            "Dataset": d_name,
                            "Stratum_Group": "condition_number",
                            "Stratum_Name": name,
                            "N_Samples": int(np.sum(mask)),
                            "Trigger_Rate": float(risk_df.loc[mask, "trigger_candidate"].mean()),
                            "Base_AUC_Macro": auc_b,
                            "ConditionalHuber_AUC_Macro": auc_r,
                            "Delta_CondHuber_minus_Base": auc_r - auc_b if np.isfinite(auc_b) and np.isfinite(auc_r) else np.nan,
                        }
                    )

            ratio_vals = risk_df.loc[(risk_df["active_count"] >= 2) & np.isfinite(risk_df["resid_ratio_max"]), "resid_ratio_max"].to_numpy()
            if ratio_vals.size > 0:
                ratio_thr = float(np.median(ratio_vals))
                thresh_rows.append({"Dataset": d_name, "Metric": "resid_ratio_max", "Threshold": ratio_thr})
                low_mask = ((risk_df["active_count"] >= 2) & (risk_df["resid_ratio_max"] < ratio_thr)).to_numpy()
                high_mask = ((risk_df["active_count"] >= 2) & (risk_df["resid_ratio_max"] >= ratio_thr)).to_numpy()
                for name, mask in [("low", low_mask), ("high", high_mask)]:
                    if int(np.sum(mask)) <= 8:
                        continue
                    auc_b = base_runner._macro_auc_subset(y_use, base_rank, is_multi=True, sample_mask=mask)
                    auc_r = base_runner._macro_auc_subset(y_use, robust_rank, is_multi=True, sample_mask=mask)
                    rows.append(
                        {
                            "Dataset": d_name,
                            "Stratum_Group": "residual_ratio",
                            "Stratum_Name": name,
                            "N_Samples": int(np.sum(mask)),
                            "Trigger_Rate": float(risk_df.loc[mask, "trigger_candidate"].mean()),
                            "Base_AUC_Macro": auc_b,
                            "ConditionalHuber_AUC_Macro": auc_r,
                            "Delta_CondHuber_minus_Base": auc_r - auc_b if np.isfinite(auc_b) and np.isfinite(auc_r) else np.nan,
                        }
                    )

            support = base_runner.support_counts.detach().cpu().numpy().astype(np.float64)
            support_thr = float(np.median(support)) if support.size > 0 else np.nan
            thresh_rows.append({"Dataset": d_name, "Metric": "class_support", "Threshold": support_thr})
            low_cls = [i for i, v in enumerate(support) if np.isfinite(v) and v < support_thr]
            high_cls = [i for i, v in enumerate(support) if np.isfinite(v) and v >= support_thr]
            for name, cls_idx in [("low", low_cls), ("high", high_cls)]:
                if len(cls_idx) <= 0:
                    continue
                auc_b = base_runner._macro_auc_subset(y_use, base_rank, is_multi=True, class_indices=cls_idx)
                auc_r = base_runner._macro_auc_subset(y_use, robust_rank, is_multi=True, class_indices=cls_idx)
                rows.append(
                    {
                        "Dataset": d_name,
                        "Stratum_Group": "class_frequency",
                        "Stratum_Name": name,
                        "N_Samples": int(n),
                        "N_Classes": int(len(cls_idx)),
                        "Trigger_Rate": np.nan,
                        "Base_AUC_Macro": auc_b,
                        "ConditionalHuber_AUC_Macro": auc_r,
                        "Delta_CondHuber_minus_Base": auc_r - auc_b if np.isfinite(auc_b) and np.isfinite(auc_r) else np.nan,
                    }
                )

    df_rows = pd.DataFrame(rows)
    df_thresh = pd.DataFrame(thresh_rows)
    rows_path = os.path.join(out_dir, "go_ml_risk_strata.csv")
    thresh_path = os.path.join(out_dir, "go_ml_risk_thresholds.csv")
    df_rows.to_csv(rows_path, index=False)
    df_thresh.to_csv(thresh_path, index=False)
    print(f"[GO-ML-Risk] saved strata -> {rows_path}")
    print(f"[GO-ML-Risk] saved thresholds -> {thresh_path}")
    return df_rows, df_thresh


def _clone_config_for_eval_mode(base_config: CAPA5Config, mode: str, save_dir: str) -> CAPA5Config:
    cfg = copy.deepcopy(base_config)
    cfg.EVAL_MODE = str(mode)
    cfg.SAVE_DIR = str(save_dir)
    cfg.PRINT_SUMMARY = False
    return cfg


def _rows_to_markdown(rows: List[Dict[str, object]], columns: List[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [header, sep]
    for row in rows:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                if np.isnan(val):
                    vals.append("nan")
                else:
                    vals.append(f"{val:.6f}")
            else:
                vals.append(str(val).replace("|", "/"))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join(body)


def run_eval_mode_comparison(
    config: CAPA5Config,
    *,
    datasets: Optional[List[str]] = None,
    split_override: int = 2,
    scoring_mode: Optional[str] = None,
    include_preprocessed_baseline: bool = False,
) -> pd.DataFrame:
    suite_dir = os.path.join(config.SAVE_DIR, "eval_mode_comparison")
    os.makedirs(suite_dir, exist_ok=True)

    all_rows: List[Dict[str, object]] = []
    mode_order = EVAL_MODE_ORDER if include_preprocessed_baseline else MAIN_EVAL_MODE_ORDER
    for mode in mode_order:
        mode_dir = os.path.join(suite_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)
        cfg_mode = _clone_config_for_eval_mode(config, mode, mode_dir)
        runner = CAPA5NotebookRunner(cfg_mode)
        df_mode = runner.run_eval_mode_report(
            datasets=datasets,
            split_override=split_override,
            scoring_mode=scoring_mode,
        )
        all_rows.extend(df_mode.to_dict(orient="records"))

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(suite_dir, "eval_mode_comparison.csv")
    json_path = os.path.join(suite_dir, "eval_mode_comparison.json")
    md_path = os.path.join(suite_dir, "eval_mode_comparison.md")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=True, default=_json_default)
    md_cols = [
        "mode",
        "dataset",
        "split",
        "macro_auc",
        "micro_auc",
        "ece",
        "run_dir",
        "prototype_source",
        "dual_track_used",
        "notes",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_rows_to_markdown(all_rows, md_cols))
        f.write("\n")
    print(f"[Saved] {csv_path}")
    print(f"[Saved] {json_path}")
    print(f"[Saved] {md_path}")
    return df


def run_eval_mode_reset_validation(
    config: CAPA5Config,
    *,
    datasets: Optional[List[str]] = None,
    split_override: int = 2,
    scoring_mode: Optional[str] = None,
) -> pd.DataFrame:
    target_dataset = datasets[0] if datasets else next(iter(config.TEST_DATA_PATHS.keys()))
    out_dir = os.path.join(config.SAVE_DIR, "eval_mode_comparison", "reset_validation")
    os.makedirs(out_dir, exist_ok=True)

    full_cfg = _clone_config_for_eval_mode(config, "full_capa", os.path.join(out_dir, "full_capa"))
    CAPA5NotebookRunner(full_cfg).run_eval_mode_report(
        datasets=[target_dataset],
        split_override=split_override,
        scoring_mode=scoring_mode,
    )

    raw_after_cfg = _clone_config_for_eval_mode(config, "raw_baseline", os.path.join(out_dir, "raw_after_full_capa"))
    df_after = CAPA5NotebookRunner(raw_after_cfg).run_eval_mode_report(
        datasets=[target_dataset],
        split_override=split_override,
        scoring_mode=scoring_mode,
    )

    raw_fresh_cfg = _clone_config_for_eval_mode(config, "raw_baseline", os.path.join(out_dir, "raw_fresh"))
    df_fresh = CAPA5NotebookRunner(raw_fresh_cfg).run_eval_mode_report(
        datasets=[target_dataset],
        split_override=split_override,
        scoring_mode=scoring_mode,
    )

    row_after = df_after.iloc[0].to_dict() if len(df_after) else {}
    row_fresh = df_fresh.iloc[0].to_dict() if len(df_fresh) else {}
    compare_cols = ["macro_auc", "micro_auc", "ece", "acc", "brier"]
    result_row: Dict[str, object] = {
        "dataset": target_dataset,
        "same_process_mode_order": "full_capa -> raw_baseline",
        "raw_after_run_dir": row_after.get("run_dir", ""),
        "raw_fresh_run_dir": row_fresh.get("run_dir", ""),
    }
    for col in compare_cols:
        a_val = float(row_after.get(col, np.nan))
        b_val = float(row_fresh.get(col, np.nan))
        result_row[f"{col}_after_full_capa"] = a_val
        result_row[f"{col}_fresh"] = b_val
        result_row[f"{col}_delta"] = a_val - b_val if np.isfinite(a_val) and np.isfinite(b_val) else np.nan

    df = pd.DataFrame([result_row])
    csv_path = os.path.join(out_dir, "reset_validation.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Saved] {csv_path}")
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CAPAv1-GT mainline runner")
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
        "--structured-prompt-bank",
        type=str,
        default="off",
        choices=["on", "off"],
        help="Enable/disable structured radiology-style prompt bank construction.",
    )
    parser.add_argument(
        "--prompt-bank-profile",
        type=str,
        default="v3",
        choices=["v1", "v2", "v3"],
        help="Structured prompt bank profile: v1 exploratory, v2 conservative, or v3 hybrid legacy+structured.",
    )
    parser.add_argument(
        "--prompt-pooling-mode",
        type=str,
        default="mean",
        choices=["mean", "bucketed"],
        help="Prompt prototype pooling rule: simple mean or bucket-aware weighted pooling.",
    )
    parser.add_argument(
        "--prompt-legacy-mix",
        type=float,
        default=0.0,
        help="When structured prompt bank is enabled, blend structured prototypes into the legacy prompt prototype with this weight.",
    )
    parser.add_argument(
        "--prompt-class-mix-profile",
        type=str,
        default="none",
        choices=["none", "cxr_conservative"],
        help="Optional class-specific override profile for structured prompt mix weights.",
    )
    parser.add_argument(
        "--prompt-coreset",
        type=str,
        default="off",
        choices=["on", "off"],
        help="Enable/disable prompt coreset selection from a larger prompt pool.",
    )
    parser.add_argument(
        "--prompt-coreset-size",
        type=int,
        default=5,
        help="Number of prompts kept per class when prompt coreset is enabled.",
    )
    parser.add_argument(
        "--prompt-max-candidates",
        type=int,
        default=24,
        help="Maximum prompt candidates per class before coreset selection.",
    )
    parser.add_argument(
        "--prompt-bucket-keep",
        type=int,
        default=2,
        help="When structured prompt bank is enabled, keep up to this many diverse prompts per bucket during coreset selection.",
    )
    parser.add_argument(
        "--prompt-score-temp",
        type=float,
        default=0.12,
        help="Softmax temperature for within-bucket prompt weighting.",
    )
    parser.add_argument(
        "--prompt-bucket-score-temp",
        type=float,
        default=0.18,
        help="Softmax temperature for bucket-level weighting.",
    )
    parser.add_argument(
        "--early-text-prompt-support",
        type=str,
        default="on",
        choices=["on", "off"],
        help="Use support-selected report prompts as early text-side semantic support.",
    )
    parser.add_argument(
        "--early-text-prompt-top-k",
        type=int,
        default=12,
        help="Number of support-selected report prompts per class for early text prototype construction.",
    )
    parser.add_argument(
        "--early-text-prompt-selection",
        type=str,
        default="margin",
        choices=["margin", "own", "mean_margin", "soft_margin"],
        help="Support-side prompt selection score.",
    )
    parser.add_argument(
        "--early-text-prompt-entry-mode",
        type=str,
        default="full",
        choices=["full", "proto_only", "mean_only"],
        help="Whether early prompts affect both zT_mean and prototypes, prototypes only, or text mean only.",
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
        "--guarded-slerp",
        type=str,
        default="off",
        choices=["on", "off"],
        help="Enable/disable Variant-B style guarded slerp candidate search.",
    )
    parser.add_argument(
        "--guarded-slerp-lambda-max",
        type=float,
        default=0.10,
        help="Maximum slerp step toward image centroids for guarded candidates.",
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
        "--go-stage2",
        type=str,
        default="off",
        choices=["on", "off"],
        help="Enable/disable Guardian Stage-2 cumulative drift detector.",
    )
    parser.add_argument(
        "--go-stage2-delta",
        type=float,
        default=0.01,
        help="Stage-2 reference margin before cumulative drift is counted.",
    )
    parser.add_argument(
        "--go-stage2-lambda",
        type=float,
        default=0.08,
        help="Stage-2 alarm threshold on the cumulative drift statistic.",
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
        default="residual",
        choices=["original", "residual", "adaptive"],
        help="Signal vector for GO multi-label update: original embedding, residual direction, or adaptive mix. Default: residual.",
    )
    parser.add_argument(
        "--go-ml-confounders",
        type=str,
        default="full",
        choices=["full", "topm", "sim_weighted"],
        help="Which co-occurring labels to project out when computing residuals.",
    )
    parser.add_argument(
        "--go-ml-topm",
        type=int,
        default=1,
        help="When --go-ml-confounders=topm, keep the top-m most similar confounders.",
    )
    parser.add_argument(
        "--go-ml-sim-temp",
        type=float,
        default=0.20,
        help="Temperature for similarity-weighted confounder residualization.",
    )
    parser.add_argument(
        "--go-ml-adapt-min-ratio",
        type=float,
        default=0.15,
        help="Minimum residual-norm ratio before adaptive residual mixing activates strongly.",
    )
    parser.add_argument(
        "--go-ml-robust",
        type=str,
        default="none",
        choices=["none", "huber"],
        help="Robust residual centroid update rule.",
    )
    parser.add_argument(
        "--go-ml-huber-delta",
        type=float,
        default=0.20,
        help="Cosine-distance Huber threshold for robust residual centroid updates.",
    )
    parser.add_argument(
        "--go-ml-huber-scope",
        type=str,
        default="always",
        choices=["always", "conditional", "warmup"],
        help="Apply Huber always, only on risky samples, or only during warm-up.",
    )
    parser.add_argument(
        "--go-ml-huber-min-active",
        type=int,
        default=3,
        help="Conditional Huber trigger: minimum active-label count.",
    )
    parser.add_argument(
        "--go-ml-huber-min-cond",
        type=float,
        default=25.0,
        help="Conditional Huber trigger: minimum regularized Gram condition number.",
    )
    parser.add_argument(
        "--go-ml-huber-min-ratio",
        type=float,
        default=0.90,
        help="Conditional Huber trigger: minimum residual/original norm ratio.",
    )
    parser.add_argument(
        "--go-ml-huber-min-sim",
        type=float,
        default=0.35,
        help="Conditional Huber trigger: minimum max similarity to another co-occurring prototype.",
    )
    parser.add_argument(
        "--go-ml-huber-warmup-steps",
        type=int,
        default=50,
        help="If huber scope is warmup, only apply within this many steps.",
    )
    parser.add_argument(
        "--go-ml-risk-analysis",
        type=str,
        default="off",
        choices=["on", "off"],
        help="Run baseline-residual vs conditional-Huber risk-stratified analysis.",
    )
    parser.add_argument(
        "--dualtrack-abstain",
        type=str,
        default="off",
        choices=["on", "off"],
        help="Enable/disable low-confidence dual-track abstain-to-raw fallback.",
    )
    parser.add_argument(
        "--dualtrack-abstain-conf",
        type=float,
        default=0.60,
        help="Confidence threshold below which dual-track falls back to the raw head.",
    )
    parser.add_argument(
        "--compare-per-dataset-capa",
        action="store_true",
        help="Run experimental shared-CAPA vs per-dataset-CAPA comparison.",
    )
    parser.add_argument(
        "--prompt-stage-isolation",
        type=str,
        default="off",
        choices=["on", "off"],
        help="Run prompt stage isolation diagnostics: legacy/raw -> structured/raw -> structured+fixed -> structured+full.",
    )
    parser.add_argument(
        "--site-expert-analysis",
        type=str,
        default="off",
        choices=["on", "off"],
        help="Run independent branch site-expert analysis with fixed-site and confidence-gated routing.",
    )
    parser.add_argument(
        "--site-expert-low-conf",
        type=float,
        default=0.60,
        help="If both competing branches have confidence below this threshold, use the designated fallback branch.",
    )
    parser.add_argument(
        "--site-expert-conf-margin",
        type=float,
        default=0.0,
        help="Required confidence margin before the primary branch overrides the fallback branch.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Optional output directory. If omitted, use config default SAVE_DIR.",
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="full_capa",
        choices=list(EVAL_MODE_ORDER),
        help="Explicit evaluation mode for the main evaluation path.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Optional comma-separated subset of datasets to evaluate, e.g. MIMIC or CheXpert,MIMIC.",
    )
    parser.add_argument(
        "--compare-eval-modes",
        action="store_true",
        help="Run raw_baseline and full_capa sequentially with isolated runners.",
    )
    parser.add_argument(
        "--include-preprocessed-baseline",
        action="store_true",
        help="Also include the optional preprocessed_baseline diagnostic row in --compare-eval-modes output.",
    )
    parser.add_argument(
        "--validate-eval-reset",
        action="store_true",
        help="Run a same-process full_capa -> raw_baseline reset smoke test.",
    )
    args = parser.parse_args()
    base_cfg = CAPA5Config(
        LABEL_SPACE=str(args.label_space),
        PARAM_PROFILE=str(args.param_profile),
        SOURCE_LABEL_ORDER_PROFILE=str(args.source_label_order_profile),
    )
    guarded_alphas_override = _parse_float_csv(args.guarded_alphas)
    config = CAPA5Config(
        PARAM_PROFILE=str(args.param_profile),
        LABEL_SPACE=str(args.label_space),
        SOURCE_LABEL_ORDER_PROFILE=str(args.source_label_order_profile),
        EVAL_MODE=str(args.eval_mode),
        DEBUG=bool(args.debug),
        VERBOSE=bool(args.debug),
        PRINT_SUMMARY=False,
        INIT_TEMPERATURE=(
            float(args.init_temperature) if args.init_temperature is not None else float(base_cfg.INIT_TEMPERATURE)
        ),
        INIT_SCALE_FACTOR=(
            float(args.init_scale) if args.init_scale is not None else float(base_cfg.INIT_SCALE_FACTOR)
        ),
        MIN_CLASSES_FOR_ADAPTATION=(
            max(1, int(args.min_classes_adapt))
            if args.min_classes_adapt is not None
            else int(base_cfg.MIN_CLASSES_FOR_ADAPTATION)
        ),
        ENABLE_STRUCTURED_PROMPT_BANK=(str(args.structured_prompt_bank).lower() == "on"),
        PROMPT_BANK_PROFILE=str(args.prompt_bank_profile),
        PROMPT_POOLING_MODE=str(args.prompt_pooling_mode),
        PROMPT_LEGACY_MIX=min(1.0, max(0.0, float(args.prompt_legacy_mix))),
        PROMPT_CLASS_MIX_PROFILE=str(args.prompt_class_mix_profile),
        ENABLE_PROMPT_CORESET=(str(args.prompt_coreset).lower() == "on"),
        PROMPT_CORESET_SIZE=max(1, int(args.prompt_coreset_size)),
        PROMPT_BUCKET_KEEP=max(1, int(args.prompt_bucket_keep)),
        PROMPT_RESOURCE_MAX_CANDIDATES=max(1, int(args.prompt_max_candidates)),
        PROMPT_SCORE_TEMP=max(1e-3, float(args.prompt_score_temp)),
        PROMPT_BUCKET_SCORE_TEMP=max(1e-3, float(args.prompt_bucket_score_temp)),
        ENABLE_EARLY_TEXT_PROMPT_SUPPORT=(str(args.early_text_prompt_support).lower() == "on"),
        EARLY_TEXT_PROMPT_TOP_K=max(1, int(args.early_text_prompt_top_k)),
        EARLY_TEXT_PROMPT_SELECTION_MODE=str(args.early_text_prompt_selection),
        EARLY_TEXT_PROMPT_ENTRY_MODE=str(args.early_text_prompt_entry_mode),
        SITE_EXPERT_UNKNOWN_LOW_CONF=min(0.99, max(0.0, float(args.site_expert_low_conf))),
        SITE_EXPERT_UNKNOWN_CONF_MARGIN=max(0.0, float(args.site_expert_conf_margin)),
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
        ENABLE_CAPAV1_GUARDED_SLERP=(str(args.guarded_slerp).lower() == "on"),
        CAPAV1_GUARDED_SLERP_LAMBDA_MAX=max(0.0, float(args.guarded_slerp_lambda_max)),
        CAPAV1_GUARDED_ALPHAS=(
            guarded_alphas_override
            if guarded_alphas_override is not None
            else list(base_cfg.CAPAV1_GUARDED_ALPHAS)
        ),
        ENABLE_GO_GUARDIAN=(str(args.go_guardian).lower() == "on"),
        ENABLE_GO_GUARDIAN_STAGE2=(str(args.go_stage2).lower() == "on"),
        GO_PSI_WINDOW=max(16, int(args.go_psi_window)),
        GO_PSI_BINS=max(5, int(args.go_psi_bins)),
        GO_PSI_THR=max(0.0, float(args.go_psi_thr)),
        GO_TAU_RESUME=max(0.0, float(args.go_tau_resume)),
        GO_RESUME_WINDOWS=max(1, int(args.go_resume_windows)),
        GO_WARMUP_STEPS=max(0, int(args.go_warmup_steps)),
        GO_BASELINE_COLLECT_STEPS=max(0, int(args.go_baseline_collect_steps)),
        GO_DRY_RUN=(str(args.go_dry_run).lower() == "on"),
        GO_STAGE2_DELTA=max(0.0, float(args.go_stage2_delta)),
        GO_STAGE2_LAMBDA=max(1e-6, float(args.go_stage2_lambda)),
        ENABLE_GO_MULTILABEL_PROJECTION=(str(args.go_ml_proj).lower() == "on"),
        GO_ML_TAU_BASE=max(1e-8, float(args.go_ml_tau)),
        GO_ML_COND_TARGET=max(1.01, float(args.go_ml_cond_target)),
        GO_ML_USE_RESIDUAL_NORM_WEIGHT=(str(args.go_ml_residual_norm).lower() == "on"),
        GO_ML_SIGNAL_MODE=str(args.go_ml_signal),
        GO_ML_SIGNAL_USE_ORIGINAL=(str(args.go_ml_signal).lower() == "original"),
        GO_ML_CONFOUNDER_MODE=str(args.go_ml_confounders),
        GO_ML_TOPM=max(1, int(args.go_ml_topm)),
        GO_ML_SIM_WEIGHT_TEMP=max(1e-3, float(args.go_ml_sim_temp)),
        GO_ML_ADAPTIVE_MIN_RESID_RATIO=min(1.0, max(0.0, float(args.go_ml_adapt_min_ratio))),
        GO_ML_ROBUST_MODE=str(args.go_ml_robust),
        GO_ML_HUBER_DELTA=max(1e-4, float(args.go_ml_huber_delta)),
        GO_ML_HUBER_SCOPE=str(args.go_ml_huber_scope),
        GO_ML_HUBER_COND_MIN_ACTIVE=max(2, int(args.go_ml_huber_min_active)),
        GO_ML_HUBER_COND_MIN_COND=max(1.0, float(args.go_ml_huber_min_cond)),
        GO_ML_HUBER_COND_MIN_RESID_RATIO=min(1.0, max(0.0, float(args.go_ml_huber_min_ratio))),
        GO_ML_HUBER_COND_MIN_OTHER_SIM=min(1.0, max(0.0, float(args.go_ml_huber_min_sim))),
        GO_ML_HUBER_WARMUP_STEPS=max(0, int(args.go_ml_huber_warmup_steps)),
        CAPAV1_DUALTRACK_ENABLE_ABSTAIN=(str(args.dualtrack_abstain).lower() == "on"),
        CAPAV1_DUALTRACK_ABSTAIN_CONF=min(0.99, max(0.0, float(args.dualtrack_abstain_conf))),
        ENTRY_SCRIPT=ENTRY_SCRIPT_PATH,
        SAVE_DIR=(str(args.save_dir) if args.save_dir else base_cfg.SAVE_DIR),
    )
    selected_datasets = [item.strip() for item in str(args.datasets).split(",") if item.strip()] or None
    legacy_mode_requested = bool(
        args.compare_softmax
        or args.compare_per_dataset_capa
    )

    if args.compare_eval_modes:
        run_eval_mode_comparison(
            config,
            datasets=selected_datasets,
            split_override=2,
            scoring_mode=config.SCORING_MODE,
            include_preprocessed_baseline=bool(args.include_preprocessed_baseline),
        )
    elif not legacy_mode_requested:
        runner = CAPA5NotebookRunner(config)
        runner.run_eval_mode_report(
            datasets=selected_datasets,
            split_override=2,
            scoring_mode=config.SCORING_MODE,
        )
    else:
        runner = CAPA5NotebookRunner(config)

        if config.VERBOSE:
            print("[Checks] Quick sanity checks")
            for path in [config.CALIB_DATA_PATH, config.TRAIN_DATA_PATH] + list(config.TEST_DATA_PATHS.values()):
                if not os.path.exists(path):
                    print(f"  [WARN] Missing path: {path}")
                    continue
                z, y, is_multi = runner._load_data(path, is_calibration=('calib' in path.lower()))
                D = z.shape[1]
                print(f"  {os.path.basename(path)} -> N={len(z)}, D={D}, labels_present={y is not None}, multi={is_multi}")

        runner._build_prototypes()
        if config.VERBOSE:
            print("  prototypes raw/processed shapes:", runner.t_raw_pooled_raw.shape, runner.t_raw_pooled.shape)
            assert runner.t_raw_pooled.shape == runner.t_raw_pooled_raw.shape

        if runner.support_counts is None:
            runner.support_counts = torch.ones(len(runner.config.ORDERED_CLASS_NAMES), device=runner.device) * runner.config.N_MIN_SUPPORT_FOR_ACTIVE
        if runner.image_centroids is None:
            runner.image_centroids = runner.t_raw_pooled.clone()
        if runner.current_R is None:
            runner.current_R = torch.eye(runner.t_raw_pooled.shape[1], device=runner.device)

        if config.VERBOSE and runner.W_zca is not None:
            WT = runner.W_zca.cpu().numpy()
            I_approx = WT @ np.linalg.pinv(WT)
            print("  ZCA approx identity max_err:", np.max(np.abs(I_approx - np.eye(I_approx.shape[0]))))

        d = runner.t_raw_pooled.shape[1]
        I = torch.eye(d, device=runner.device)
        if config.VERBOSE:
            Rtest = runner._solve_procrustes()
            ortho_err = torch.norm(Rtest @ Rtest.T - I).item()
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

    if args.validate_eval_reset:
        run_eval_mode_reset_validation(
            config,
            datasets=selected_datasets,
            split_override=2,
            scoring_mode=config.SCORING_MODE,
        )
    if str(args.prompt_stage_isolation).lower() == "on":
        run_prompt_stage_isolation_analysis(config)
    if str(args.site_expert_analysis).lower() == "on":
        run_site_expert_branch_analysis(config)
    if str(args.go_ml_risk_analysis).lower() == "on":
        run_go_ml_risk_stratified_analysis(config)

