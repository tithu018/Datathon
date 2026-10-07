# Datathon work log

Feeds the AI tool disclosure and the preprocessing document. One entry per phase.

---

## Phase 0: Setup and baseline (2026-10-07)

**Done**
- `git fetch`; confirmed `main` == `origin/main` == `eaf7297`. Created branch
  `datathon-task2a-task1-fixes` from it.
- **Repo found to be PUBLIC** (`https://github.com/tithu018/Datathon`,
  unauthenticated GitHub API returned HTTP 200, `"visibility": "public"`).
  The competition datasets had been committed in `eaf7297` and are on
  `origin/main`, against the booklet's data-sharing terms. The team leader
  is making the repo private. No history rewrite or force-push will be done;
  nothing is pushed until the repo is confirmed private.
- Untracked (`git rm --cached`) all raw dataset CSVs (`General Data/`,
  `Training Data/`, `Test Data/`, `Submission Templates/`) and the three generated Task 1 CSVs
  (`task1/task1_training_labels.csv`, `task1/train_features.csv`,
  `task1/test_features.csv`). Files stay on disk locally.
- `.gitignore`: ignores the four dataset folders (including `Submission Templates/`)
  and `*.csv` by default, with exceptions for `outputs/`, `task1/reports/` and
  `task2a/reports/`; also ignores `task2a/data/` and notebook checkpoints.
- Added `data/README.md` (where datasets must be placed locally).
- Added `requirements.txt` (only imported packages, at installed versions).
- Added `task1/baseline_eval.py` and `task1/reports/baseline_metrics.json`.
- Copied the original `outputs/submission_task1.csv` to
  `task1/reports/submission_task1_original.csv`.

**WARNING for teammates (also goes into `task1/CHANGES_FOR_REVIEW.md`)**
Merging this branch removes the dataset CSVs from tracking. When a teammate
pulls the merge, git **deletes those files from their working folder**. Keep a
copy of the datasets outside the repo and copy them back after pulling.

**Environment**
- Interpreter: system Python **3.11.2** at
  `C:\Users\Arnikan\AppData\Local\Programs\Python\Python311\python.exe`.
  (The shell's default `python` is an unrelated, empty `Lab7` virtualenv; do not use it.)
- Installed and pinned in `requirements.txt` (exact versions): joblib 1.5.3,
  numpy 2.4.6, pandas 3.0.3, scikit-learn 1.9.0.
- **lightgbm dropped.** `pip install` fails with `SSL: CERTIFICATE_VERIFY_FAILED,
  self signed certificate in certificate chain` (HTTPS interception on this
  network). User decision: do not bypass certificate verification; use
  scikit-learn's `HistGradientBoostingRegressor`/`Classifier` instead (supports
  Poisson loss and native categorical features). `docs/DATATHON_PLAN.md` updated
  in Phase 0 step 4, Phase 4 items 1–2, Phase 8 item 1 and the cut list.

**Task 1 baseline (original pipeline, unchanged)**

| Split | n_test | late rate | Service MAE | Service RMSE | ROC-AUC | Log loss | Brier | share p=0 | share p=1 |
|---|---|---|---|---|---|---|---|---|---|
| Random 80/20 | 18,379 | 0.196 | 5.520 | 8.950 | 0.9278 | 0.290 | 0.0916 | 12.1% | 0.11% |
| Time-based (order_date ≥ 2026-01-03) | 5,173 | 0.131 | 5.152 | 8.319 | 0.9315 | 0.232 | 0.0727 | 14.2% | 0% |

- Labels: 91,894 rows; 0 negative or missing service times; late rate 0.1958.
- Only 12 features survive (brand, district, depot and vehicle columns are
  silently dropped by the `_order`/`_leg` suffix bug), as expected.
- **Reproducibility finding:** re-running the original pipeline here gives
  service predictions identical to the committed submission (max diff 6e-14),
  but late probabilities differ by up to 0.227 (mean abs diff 0.052; mean p
  0.302 here vs 0.251 committed; share of exact 0s 10.6% vs 14.2%). The
  committed `outputs/submission_task1.csv` was restored so Phase 0 changes no
  outputs.

**Late-probability non-reproducibility: investigation (for Phase 6 and CHANGES_FOR_REVIEW.md)**
- *Seeds are not the cause.* The late model sets `random_state=42`. Two runs
  here are bit-identical, and `n_jobs=1` equals `n_jobs=-1`. Changing the seed
  (0, 1, 7, None) moves predictions by the same ~0.052 mean, so the gap is a
  systematic shift, not seed noise.
- *Inputs are not the cause.* Service predictions reproduce exactly, so the
  training rows, their order and the 80/20 split are identical. Training on all
  rows, `min_samples_leaf` 1 or 5, or `n_estimators` 100/200 do not reproduce it.
- *Main cause: scikit-learn 1.9 changed RandomForest bootstrap with weights.*
  In 1.9 (`sklearn/ensemble/_forest.py`), `class_weight="balanced"` weights are
  used to **draw** the bootstrap sample (`rng.choice(..., p=weights)`), and the
  trees are fitted on counts only. Up to 1.8, the bootstrap was uniform and the
  class weights were **multiplied** into each tree's sample weights. The
  regressor has no weights, so its path is unchanged; that is why only the late
  model differs. Emulating the pre-1.9 behaviour cuts the mean abs diff from
  0.052 to 0.014 (corr 0.9975 vs 0.9880). The remaining gap is most likely other
  library version differences; it cannot be pinned down without the teammate's
  `pip list`.
- *Consequence:* the original code has no pinned versions, so its late
  probabilities depend on the installed scikit-learn. `baseline_metrics.json`
  reflects the original code under scikit-learn 1.9.0, not the teammate's exact
  model. Fix in Phase 6/8: pin versions (done in `requirements.txt`) and revisit
  `class_weight="balanced"` (planned anyway in Phase 8).
- Open question for the teammate: which pandas / numpy / scikit-learn versions
  produced the committed submission?

**Decisions**
- Untrack data on this branch only (user decision); no history rewrite.
- Use the system Python 3.11 instead of the `Lab7` venv (user approved).
- Drop lightgbm; use scikit-learn HistGradientBoosting (user decision).
- Untrack `Submission Templates/` too (user decision).
- Commit `docs/DATATHON_PLAN.md` (with the lightgbm edits) in the Phase 0 commit.
- Time-based test splits by `order_date`; a few deferred orders near the
  cutoff were dispatched after it (minor, noted).

**Authorship**
- Agent (Claude Code) wrote: `.gitignore` changes, `data/README.md`,
  `requirements.txt`, `task1/baseline_eval.py`, this entry.
- Agent also ran the late-probability reproducibility experiments (scratch
  scripts, not committed) and edited `docs/DATATHON_PLAN.md` to replace LightGBM.
- User decided: untracking data and templates, requirements scope, path
  convention, no-push rule, dropping lightgbm, system Python.

**Addendum (recorded at Phase 0 approval)**
- `task1/reports/baseline_metrics.json` reflects the original Task 1 code run
  under **scikit-learn 1.9.0**. The late-probability version issue disappears
  once Task 1 is fixed and retrained (Phase 6+), because every new model is
  trained in the pinned environment.
- Add `matplotlib` to `requirements.txt` when plots are first created (Phase 2).
- Phase 0 committed as `f9aeb89`. Not pushed (repo still public).

---

## Phase 1: Task 2A labels (2026-10-07)

**Done**
- `task2a/prepare_labels.py`: loads `Training Data/deliveries_train.csv` and
  `Test Data/task1_test_inputs.csv`, keeps every order (attempted, deferred,
  not_run), assigns weeks by `order_date` through `calendar.csv`
  (`iso_year`, `iso_week`), and builds:
  - `task2a/data/daily_volume.csv`: complete daily grid, calendar date x depot x
    brand, zero-filled, 4,914 rows (819 days x 6 series, 2024-01-01 to 2026-03-29,
    complete ISO weeks). Columns: total_volume, chilled_volume, n_orders,
    is_operating.
  - `task2a/data/weekly_volume.csv`: depot x brand x iso_year x iso_week, 702 rows
    (117 weeks x 6 series), with operating_days per week.
  Both are gitignored (`task2a/data/`).

**Key numbers (all asserted against the user's independent reference; all pass)**
- Orders: 92,307 train + 5,014 task1 = 97,321; 0 overlapping `delivery_id`.
- order_date 2024-01-01 to 2026-03-28; every date found in calendar.csv; 0 orders
  on `is_operating = 0` days.
- dispatch_status: attempted 95,275, deferred 1,633, not_run 413
  (task1 file: 4,924 / 90 / 0).
- Volume (m3): Fresh 170,358.844 (ambient 107,905.754, chilled 62,453.090),
  Style 26,044.960, Tech 5,963.394. Chilled only in Fresh.
- 117 weeks per series (2024-W01 to 2026-W13). Mean weekly total: Kandy Fresh
  500.26, Style 80.94, Tech 21.19; Peliyagoda Fresh 955.80, Style 141.67,
  Tech 29.78.
- Peliyagoda Fresh 2025-W15 = 1,351.0 (6 operating days), 2025-W16 = 623.5
  (4 operating days).
- daily sum = weekly sum = raw sum (202,367.198 m3 total; 62,453.090 chilled).
- Extra check: brand and depot of every order agree with `outlets.csv`.

**Caveat (for the preprocessing document)**
- The task1 file contains only dispatched orders (no `not_run`). In the training
  file, not_run orders are 0.44% of volume (0.45% of orders), so the last 6
  history weeks (2026-W08 to W13) may undercount demand by about that share.

**Decisions**
- Daily grid covers whole ISO weeks (it runs to Sunday 2026-03-29, a
  non-operating day with zero volume), so weekly sums are over complete weeks.

**Authorship**
- Agent wrote `task2a/prepare_labels.py` and this entry. Reference numbers were
  computed independently by the user.
