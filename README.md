# CAPAv1-GT

## What is this?

This is the standalone engineering implementation of the current `CAPAv1-GT` mainline for 5-label chest X-ray adaptation with BioMedCLIP.

The default mainline is:

- `chexpert5` label space
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

If your local paths are different, update the defaults in `capa_experiment/mainline.py` or override them through `CAPA5Config`.

## What are the results?

Current mainline results relative to the frozen BioMedCLIP zero-shot baseline:

| Dataset | BioMedCLIP Baseline | CAPAv1-GT | Delta |
| --- | ---: | ---: | ---: |
| CheXpert | `0.5950` | `0.6197` | `+0.0247` |
| MIMIC | `0.5921` | `0.5977` | `+0.0056` |
| COVID | `0.7907` | `0.8158` | `+0.0252` |
| RSNA | `0.6811` | `0.7749` | `+0.0938` |
