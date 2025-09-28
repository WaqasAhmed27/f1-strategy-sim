# F1 Strategy Sim

Modular F1 Race Prediction System for the 2025 season. Pulls FastF1 session data, engineers features, trains models, and predicts finishing order, lap times, and intervals. Roadmap includes strategy simulations and dashboards.

## Quickstart

1. Create a virtual environment and install:

```bash
pip install -e .[dev]
```

2. Run CLI help:

```bash
f1sim --help
```

3. Predict a race (placeholder):

```bash
f1sim predict --season 2025 --round 1
```

## Structure

- `src/f1sim/`: package code
- `configs/`: YAML configs
- `data/`: raw and processed data (gitignored)
- `models/`: trained artifacts (gitignored)
