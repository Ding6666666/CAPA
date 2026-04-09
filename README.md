# CAPA Experiment Code

`experimen.py` and the structured package under `code/` now track `main.py` as the source of truth.
Use this layout when you want the clearer module structure without changing the experiment behavior from `main.py`.

## Directory Layout

```text
code/
|- run_experiment.py
`- capa_experiment/
   |- cli.py
   |- config.py
   |- runner.py
   |- runtime.py
   `- mixins/
      |- core.py
      |- guardian.py
      |- scoring.py
      |- adaptation.py
      |- evaluation.py
      `- reporting.py
```

## Usage

From the project root:

```bash
python code/run_experiment.py --help
```

This entrypoint mirrors `main.py`'s CLI and runtime behavior.
