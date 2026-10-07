# How to run

## Environment

- **Python:** 3.11.2. Versions are pinned in `requirements.txt`: joblib 1.5.3, matplotlib 3.11.1, numpy 2.4.6, pandas 3.0.3, scikit-learn 1.9.0.
- **Install:** `pip install -r requirements.txt`.
- **No lightgbm:** pip's certificate check fails on our network, so HistGradientBoosting from scikit-learn is used instead.

## Data (not in git)

Place the competition folders at the repository root, exactly as `data/README.md` shows:

```
General Data/   Training Data/   Test Data/   Submission Templates/
```

## Commands

Run all commands from the repository root (the folder that contains `task1/` and `task2a/`):

```
python task2a/run_all.py           # Task 2A: about 30 s
python task1/run_all.py            # Task 1: about 4 min (includes the two validation runs)
python task1/run_all.py --no-eval  # Task 1, final fit only (skips the validation runs; faster)
python task2b/run_all.py           # Task 2B, all phases including exact sensitivity
python task2b/integration_check.py # saved-model inference and all three submissions
```

| Pipeline | Steps |
|---|---|
| `task2a/run_all.py` | prepare_labels → prepare_features → train_model → make_submission → forecast_report |
| `task1/run_all.py` | prepare_labels → prepare_features → train_model → make_submission → compare_submission |
| `task2b/run_all.py` | smoke → capacity → policy → greedy → exhaustive checks → exact optimiser → window checks/audit → sensitivity → write-up → submission |

- Both pipelines stop at the first failing step.
- `task2a/run_all.py --full` also re-runs the Task 2A analysis, the Phase 3 backtest and the scoring of the final config.
- **Reproducibility:** each pipeline was run twice from scratch, with byte-identical submissions both times.

## Where things appear

| | Task 2A | Task 1 |
|---|---|---|
| Submission | `outputs/submission_task2a.csv` (60 rows) | `outputs/submission_task1.csv` (5,014 rows) |
| Models (gitignored) | `task2a/models/task2a_model.joblib` | `task1/service_model.joblib`, `task1/late_model.joblib` |
| Model summary (tracked) | `task2a/models/task2a_model_summary.json` | `task1/reports/metrics.json` |
| Intermediate data (gitignored) | `task2a/data/` | `task1/task1_training_labels.csv`, `task1/train_features.csv`, `task1/test_features.csv` |
| Reports | `task2a/reports/` | `task1/reports/` |

- **Inference only** (loads saved models): `task2a/predict.py` has `predict_task2a(test_inputs_df)`. The combined notebook cell is in NOTEBOOK_SNIPPETS.md.
- **Experiment scripts:** `task2a/backtest.py`, `task2a/experiments.py` and `task1/phase7_features.py` / `phase8_models.py` regenerate the reports.
- **Don't re-run the confirmation:** `task2a/experiments.py --confirm` was evaluated once and refuses to run again.
- **Task 2B:** see TASK2B_README.md for outputs, exact proof scope and local-only audit files. The current regression used Python 3.11.0; retraining can cause small numerical differences across environments despite pinned package versions.
- **Zip layout:** the brief wants the model files next to the notebook. Copy the three `.joblib` files after running both pipelines, or point the notebook at their repo paths.
