# SPEL Model

Recurrent predictive-coding model code.

```text
environments/    Task rules, stimuli, and episode datasets
models/          Networks, losses, training, and evaluation definition
analysis/        Preprocessing, selectivity, and decoding
visualizations/  Plotting scripts and notebooks
configs/         Experiment configurations
artifacts/       Local data and outputs
run.py           Training and evaluation CLI
```

## QuickStart

```bash
uv venv --python 3.10 --seed .venv
.venv/bin/pip install -e .
```

Use `.venv/bin/python` for project commands. Run notebooks with this environment's Jupyter kernel after the install.
