# Handover: Task 2A and Task 1 changes

For the teammates building the shared Datathon deliverables (final notebook, preprocessing
document, architecture diagram, AI disclosure, video, zip). Everything here is on branch
`datathon-task2a-task1-fixes`.

## Results summary

**Task 2A, the weekly depot demand forecast:**
- **Models:**
  - Fresh: 0.6 × a daily Poisson GLM + 0.4 × a global HistGradientBoosting model.
  - Style: an outlet × weekday GLM. Each Style outlet orders once a week on a fixed day, and skips the order when that day is closed.
  - Tech: a calendar-shaped 52-week level.
  - Chilled: a depot × month chilled share times the Fresh forecast.
- **Accuracy:** across four rolling 10-week backtests, total weekly error (WAPE) is **3.14%** (3.47% on 2025 weeks 14–23). That compares with 8.7% for a last-8-weeks average and 3.38% for the first GLM. A confirmation period, evaluated once, gave 3.47% against 5.3–5.7% for the baselines.
- **Forecast:** the closure weeks are handled. New Year build-up lifts W15 by +40–60%; New Year closures cut W16 by about 33%; the Vesak Friday closure cuts Peliyagoda Style and Tech in W18.

**Task 1, service time and lateness:**
- **The fix:** five features were being silently dropped, and the lateness probabilities were uncalibrated.
- **Final models:** 30 planned-information features. HistGradientBoosting predicts service time; a sigmoid-calibrated HistGradientBoosting classifier predicts lateness, with probabilities clipped to [0.001, 0.999].
- **Accuracy:** tested on two time-based periods, one season-matched to the real test:
  - service-time error (MAE) fell from 5.15 / 5.06 to **3.92 / 3.87** minutes;
  - late-prediction error (log loss) fell from 0.232 / 0.314 to **0.140 / 0.163**;
  - ranking accuracy (AUC) rose from 0.93 / 0.92 to **0.972 / 0.976**.
- **Submission:** the mean predicted late rate is 0.213. February is 0.13 and monsoon March 0.25, matching the seasonal history.

## Package contents

| File | Use it for |
|---|---|
| [HOW_TO_RUN.md](HOW_TO_RUN.md) | commands, data locations, outputs, versions |
| [NOTEBOOK_SNIPPETS.md](NOTEBOOK_SNIPPETS.md) | the final inference cell (tested: reproduces both submissions exactly) |
| [inference_cell.py](inference_cell.py), [smoke_test_output.txt](smoke_test_output.txt) | the same cell as a script, and its verified output |
| [PREPROCESSING_SECTIONS.md](PREPROCESSING_SECTIONS.md) | ready-to-paste text for the preprocessing document |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Mermaid pipeline diagrams and the proposed deployment |
| [AI_DISCLOSURE_NOTES.md](AI_DISCLOSURE_NOTES.md) | input for the AI tool disclosure |
| [VIDEO_TALKING_POINTS.md](VIDEO_TALKING_POINTS.md) | key numbers and plots for the demo video |

**More detail:**
- `WORKLOG.md` at the repo root: the full log.
- `task1/CHANGES_FOR_REVIEW.md`: the Task 1 change report.
- `task2a/reports/` and `task1/reports/`: metrics, tables and plots.

> **Before merging or pulling this branch:** it untracks the dataset folders, so `git pull`
> deletes them from your working folder. Back them up first (see `data/README.md`).
