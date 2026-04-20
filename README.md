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
- standalone smoke run from the `code/` entrypoint

The validated standalone smoke result matched the current mainline behavior on:

- CheXpert: `+0.0233`
- MIMIC: `+0.0024`
- COVID: `+0.0259`
- RSNA: `+0.0953`

## Notes

- This repository intentionally exposes a single cleaned mainline rather than multiple historical branches.
- OT is not part of this engineering package.
- Cache is treated as an evaluation-time optional expert, not as an always-on enhancement.
- The code is optimized for reproducibility and clarity over preserving older exploratory branches.
