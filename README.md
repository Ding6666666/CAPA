# CAPAv1-GT

Standalone engineering implementation of the current `CAPAv1-GT` experiment pipeline for chest X-ray label adaptation with BioMedCLIP.

This repository is the cleaned and self-contained `code/` package extracted from the larger CAPA workspace. It does **not** depend on the repository-root `experiment_3line.py`.

## What This Repository Contains

This project keeps the current GT-based mainline only:

- internal label space defaults to `chexpert5`
- source label interpretation defaults to `chexpert5_reordered_200x5`
- adaptation follows:
  `GT support -> multi-label residualization -> centroid EMA -> guarded Procrustes`
- cache is available only as an evaluation-time expert with `off|gated`
- OT is intentionally removed from this engineering entrypoint

In method terms, this code should be read as a `CAPAv1-GT` / `GT-oracle` line:

- GT replaces the pseudo-label evidence-generation stage that previously relied on `SCS + HPQ/LQ`
- the downstream geometry and multi-label machinery are retained, including:
  - multi-label residual projection
  - centroid EMA updates
  - Guardian-style monitoring
  - guarded Procrustes selection
  - evaluation and reporting

## Repository Layout

```text
code/
|- run_experiment.py
`- capa_experiment/
   |- cli.py
   |- config.py
   |- constants.py
   |- mainline.py
   |- runner.py
   |- runtime.py
   |- __init__.py
   `- __main__.py
```

## Main Files

- [run_experiment.py](./run_experiment.py): top-level CLI entrypoint
- [capa_experiment/mainline.py](./capa_experiment/mainline.py): source-of-truth implementation for the current experiment
- [capa_experiment/config.py](./capa_experiment/config.py): public config export
- [capa_experiment/constants.py](./capa_experiment/constants.py): label spaces, synonym maps, binary target maps, source-order helpers
- [capa_experiment/cli.py](./capa_experiment/cli.py): CLI wrapper around the mainline
- [capa_experiment/runner.py](./capa_experiment/runner.py): public runner export

## Environment

This codebase is designed to run in the existing `capa` environment used during development.

Core Python dependencies include:

- `torch`
- `open_clip`
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `tqdm`

## Data and Directory Layout

By default, paths resolve relative to the repository root:

- `model/`
- `data/`
- `results/`

Expected data layout:

```text
code/
|- data/
|  |- CHEXPERT_MIMIC.pkl
|  |- data_train.pkl
|  |- cheXpert_200x5.pkl
|  |- MIMIC_200x5.pkl
|  `- raw_data/
|     |- CheXpert-v1.0-small/
|     |- COVID19-Radiography-Database/
|     `- rsna/
|- model/
`- results/
```

If your local layout differs, override paths through `CAPA5Config` or by editing the defaults in [capa_experiment/mainline.py](./capa_experiment/mainline.py).

## Default Experimental Semantics

- label space: `chexpert5`
- source order profile: `chexpert5_reordered_200x5`
- minimum active-class support: `8`
- cache mode: `off`
- scoring mode: `mixed`
- sim source: `gate`

The default setup is intentionally conservative and aligned with the current validated GT mainline.

## Running

Show CLI help:

```bash
python run_experiment.py --help
```

Run the default GT mainline:

```bash
python run_experiment.py --label-space chexpert5 --source-label-order-profile chexpert5_reordered_200x5
```

Run with gated cache enabled:

```bash
python run_experiment.py \
  --label-space chexpert5 \
  --source-label-order-profile chexpert5_reordered_200x5 \
  --cache-mode gated
```

Write outputs to a custom directory:

```bash
python run_experiment.py --save-dir ./results
```

## Outputs

Typical outputs include:

- `final_manuscript_table.csv`
- `four_dataset_three_way.csv`
- `capa_state.pkl`
- per-dataset curve figures
- intermediate evaluation tables

Results are written under the configured `SAVE_DIR`, typically in a `capav1_gt/` subdirectory.

## Validation Status

This repository version was checked against the current GT mainline and validated through:

- CLI help check
- Python syntax compilation
- standalone end-to-end evaluation from the `code/` entrypoint

The current validated standalone run uses:

- label space: `chexpert5`
- source order profile: `chexpert5_reordered_200x5`
- cache mode: `off`

The main comparison is the frozen BioMedCLIP zero-shot baseline versus the `CAPAv1-GT` aligned model:

| Dataset | BioMedCLIP Baseline | CAPAv1-GT | Delta |
| --- | ---: | ---: | ---: |
| CheXpert | `0.5950` | `0.6183` | `+0.0233` |
| MIMIC | `0.5921` | `0.5945` | `+0.0024` |
| COVID | `0.7907` | `0.8166` | `+0.0259` |
| RSNA | `0.6811` | `0.7764` | `+0.0953` |

These numbers come from the validated standalone evaluation run of this repository, not from the larger workspace wrapper.

## Notes

- This repository exposes one supported experiment path only: the cleaned `CAPAv1-GT` mainline.
- Older historical branches and exploratory variants are intentionally not included here.
- OT is not included in this engineering package.
- Cache is available only as an optional evaluation-time expert and is `off` by default.
- The goal of this repository is a clear, reproducible implementation of the current experiment, not a full archive of every earlier ablation.
