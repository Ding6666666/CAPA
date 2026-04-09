from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch

@dataclass
class CAPA5Config:
    RANDOM_SEED: int = 42
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME: str = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    LOCAL_MODEL_PATH: str = r"D:\Project\ML\CAPA\model"
    DATA_ROOT: str = r"D:\Project\ML\CAPA\data"
    CALIB_DATA_PATH: str = rf"{DATA_ROOT}\CHEXPERT_MIMIC.pkl"
    TRAIN_DATA_PATH: str = rf"{DATA_ROOT}\data_train.pkl"
    # Post-hoc temperature scaling should be fit on a held-out calibration set (NOT test sets)
    TAU_CALIB_DATA_PATH: str = rf"{DATA_ROOT}\CHEXPERT_MIMIC.pkl"
    TAU_CALIB_FRAC: float = 0.2
    TAU_CALIB_MAX: int = 5000
    TAU_CALIB_MIN: int = 2000

    # === Logging ===
    VERBOSE: bool = False
    PRINT_SUMMARY: bool = True
    DEBUG: bool = False
    SAVE_DIR: str = r"D:\Project\ML\CAPA\results" 
    AUC_BOOTSTRAP_ROUNDS: int = 1000
    
    TEST_DATA_PATHS: Dict[str, str] = field(default_factory=lambda: {
        "CheXpert": rf"D:\Project\ML\CAPA\data\raw_data\CheXpert-v1.0-small",
        "MIMIC":    rf"D:\Project\ML\CAPA\data\MIMIC_200x5.pkl",
        "COVID":    rf"D:\Project\ML\CAPA\data\raw_data\COVID19-Radiography-Database",
        "RSNA":     rf"D:\Project\ML\CAPA\data\raw_data\rsna",
    })

    ORDERED_CLASS_NAMES: List[str] = field(default_factory=lambda: [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion", 
        "Pneumonia", "Pneumothorax", "Fracture", "Lung Lesion", "Lung Opacity", 
        "Pleural Other", "Enlargement of the Cardiac Silhouette", "Pneumoperitoneum", "Support Devices"
    ])

    # === Budget & Active Learning ===
    CONF_HIGH: float = 0.85  
    CONF_MED: float = 0.70   
    
    MULTI_LABEL_RIDGE: float = 0.1  
    PROB_THRESHOLD: float = 0.5     
    KAPPA_EMA: float = 25         
    
    # === Curriculum Learning & Warm-up ===
    WARMUP_BATCHES: int = 50        
    CURRICULUM_THRESH_START: float = 0.90 
    HPQ_CONF_LO: float = 0.65   
    
    SCS_THRESH: float = 0.72    
    M: int = 5
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

    ENABLE_GO_MULTILABEL_PROJECTION: bool = True
    GO_ML_TAU_BASE: float = 1e-2
    GO_ML_COND_TARGET: float = 1e3
    GO_ML_USE_RESIDUAL_NORM_WEIGHT: bool = True
    GO_ML_SIGNAL_USE_ORIGINAL: bool = True

    # === Procrustes & Gating ===
    KAPPA0: float = 0.0
    GAMMA_S: float = 1.0
    
    # [关键参数] 每个类别必须达到的最小样本数
    N_MIN_HPQ_FOR_ACTIVE: int = 8 
    
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
    PROCRUSTES_WEIGHT_CAP_MULT: float = 2.0
    ENABLE_HARD_NEG_PROCRUSTES: bool = False
    HARD_NEG_BETA: float = 0.15
    HARD_NEG_TOPK: int = 3
    HARD_NEG_TEMP: float = 0.07
    MIN_CLASSES_FOR_ADAPTATION: int = 6
    ENABLE_OT_PROTOTYPE_MIXING: bool = True
    OT_SINKHORN_EPS: float = 0.08
    OT_SINKHORN_ITERS: int = 100
    OT_ACTIVE_ONLY: bool = True
    OT_MIN_ACTIVE_CLASSES: int = 2
    # alpha in [0,1]: 0 => pure OT-mixed text; 1 => original text.
    OT_IDENTITY_BLEND: float = 0.08
    
    INIT_TEMPERATURE: float = 0.590625
    INIT_SCALE_FACTOR: float = 5.0
    TAU_PRIOR: float = 0.0
    SCORING_MODE: str = "mixed"  # "mixed" (default) or "softmax"
    SIM_SOURCE: str = "gate"  # "gate" or "dataset"
    TRAIN_BATCH_SIZE: int = 128
    # CAPA+Cache (simple global train-cache) switch and hyper-params.
    ENABLE_CAPA_CACHE: bool = True
    # Locked default from fixed-geometry cache grid best config:
    # topk=48, alpha=0.3, temp=0.05
    CACHE_TOPK: int = 48
    CACHE_TEMP: float = 0.05
    CACHE_ALPHA: float = 0.3
    CACHE_CHUNK: int = 512
    CACHE_SELF_MATCH_COS: float = 0.999999
    ENABLE_CAPA_BASELINE_SOFT_FUSION: bool = True
    CAPA_BASELINE_FUSION_LAMBDA: float = 1.0

    MEDICAL_SYNONYM_MAP: Dict[str, List[str]] = field(default_factory=lambda: {
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
        "Support Devices": ["support devices", "medical tubes", "lines", "pacemaker"]
    })

    TEMPLATES_PI: List[str] = field(default_factory=lambda: [
        "This chest X-ray shows {finding}.",
        "There is evidence of {finding}.",
        "Findings consistent with {finding}.",
        "The image demonstrates {finding}.",
        "Signs of {finding} are present."
    ])
    BINARY_POSITIVE_CLASS_MAP: Dict[str, List[str]] = field(default_factory=lambda: {
        "default": ["Pneumonia"],
        "COVID": ["Pneumonia"],
        "RSNA": ["Pneumonia"],
    })
