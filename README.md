# CAPA Experiment Code

`code/` is now a self-contained `CAPAv1-GT` experiment package.
The mainline implementation is vendored into [capa_experiment/mainline.py](./capa_experiment/mainline.py),
so this folder no longer depends on the repository-root `experiment_3line.py`.

## Directory Layout

```text
code/
|- run_experiment.py
`- capa_experiment/
   |- cli.py
   |- constants.py
   |- config.py
   |- mainline.py
   |- runner.py
   `- runtime.py
```

## Usage

From the project root:

```bash
python code/run_experiment.py --help
```

## Current Semantics

- Default label space: `chexpert5`
- Default source label interpretation: `chexpert5_reordered_200x5`
- Mainline adaptation: `GT support -> multi-label residualization -> centroid EMA -> guarded Procrustes`
- Cache: evaluation-only `off|gated`
- OT: removed from this engineering entrypoint
- Default paths resolve relative to the package root: `data/`, `model/`, `results/`

## Notes

- `mainline.py` is the vendored source-of-truth implementation for the current GT-only experiment.
- `constants.py` and `config.py` re-export the mainline definitions so external imports stay stable.
- `runner.py` and `cli.py` expose the same mainline runner and entrypoint from inside the package.
