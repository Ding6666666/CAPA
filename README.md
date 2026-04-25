# CAPAv1-GT

## What is this?

This is the standalone engineering implementation of the current `CAPAv1-GT` mainline for 5-label chest X-ray adaptation with BioMedCLIP.

The default mainline is:

- `chexpert5` label space
- CheXpert-5 target-aware train/calibration data materialized from `data_train.pkl`
- `GT support -> order-free multi-label residualization -> residual-centroid EMA -> guarded Procrustes`
- `legacy-safe` prompt bank
- `cache=off`

## How do I run it?

Show CLI help:

```bash
python run_experiment.py --help
```

Run the default mainline:

```bash
python run_experiment.py
```

Run the default main comparison:

```bash
python run_experiment.py --compare-eval-modes
```

Write outputs to a custom directory:

```bash
python run_experiment.py --save-dir ./results
```

## What data/directories does it depend on?

By default, the code looks for these directories under the repository root:

- `model/`
- `data/`
- `results/`

Expected data layout:

```text
CAPA/
|- data/
|  |- CHEXPERT_MIMIC.pkl
|  |- data_train.pkl
|  |- data_train_chexpert5_target_positive.pkl                         # auto-created if missing
|  |- data_train_chexpert5_target_positive_image_calibration.pkl        # auto-created if missing
|  |- data_train_chexpert5_target_positive_summary.json                 # auto-created if missing
|  |- cheXpert_200x5.pkl
|  |- MIMIC_200x5.pkl
|  `- raw_data/
|     |- CheXpert-v1.0-small/
|     |- COVID19-Radiography-Database/
|     `- rsna/
|- model/
`- results/
```

If your local paths are different, update the defaults in `capa_experiment/mainline.py` or override them through `CAPA5Config`.

For the default `chexpert5` mainline, `data_train.pkl` remains the immutable source pool.
The code automatically creates a target-aware view that keeps rows with at least one of:
`Atelectasis`, `Cardiomegaly`, `Consolidation`, `Edema`, `Pleural Effusion`.
The calibration file is the image-only subset of that same target-aware view.

## What are the results?

Default main comparison modes:

- `raw_baseline`: raw zero-shot diagnostic reference
- `full_capa`: complete CAPA pipeline

The former processed-space baseline is now available only as the optional diagnostic mode
`preprocessed_baseline`; include it with `--include-preprocessed-baseline`.

Current mainline results relative to the frozen BioMedCLIP zero-shot baseline:

| Dataset | BioMedCLIP Baseline | CAPAv1-GT | Delta |
| --- | ---: | ---: | ---: |
| CheXpert | `0.5778` | `0.6441` | `+0.0663` |
| MIMIC | `0.5717` | `0.6339` | `+0.0621` |
| COVID | `0.7580` | `0.7226` | `-0.0354` |
| RSNA | `0.6124` | `0.5594` | `-0.0530` |

These are macro-AUC values from:
`results/mainline_target_aware_check/capav1_gt/eval_mode_comparison/eval_mode_comparison.csv`.
This mainline is now aligned to the CheXpert-5/MIMIC-5 protocol; COVID/RSNA remain external binary stress checks.
