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

**Phase 1 decisions (recorded at Phase 1 approval)**
- Daily grid extended to Sunday 2026-03-29 so the last history week is complete: kept.
- **Known limitation:** the task1 file has no not_run orders, so 2026-W08..W13
  may undercount demand by about 0.44% of volume. Documented only, not
  scaled: it is an order of magnitude below the ~5% model error.
- Phase 1 committed as `f28c936`. Not pushed (repo still public).

---

## Phase 2: Task 2A features and analysis (2026-10-07)

**Done**
- `task2a/prepare_features.py`: calendar-level features for every date in
  calendar.csv, crossed with the 6 series; history rows carry the Phase 1
  labels, forecast rows (2026-03-30 to 2026-06-07) have none. Outputs
  `task2a/data/calendar_features.csv` (910 rows) and
  `task2a/data/daily_features.csv` (5,334 rows: 4,914 history + 420 forecast).
  Features: dow, is_operating, is_payday, days_to_payday, days_since_payday
  (capped at 14), festival_ramp, festival_ramp_sq, festival_name (festival the
  ramp leads up to, "none" when ramp = 0), is_holiday, monsoon, month,
  trend_years (since 2024-01-01), days_since_last_operating_day,
  is_first_day_after_closure (after 2+ consecutive closed days),
  days_to_next_closure, days_to_next_holiday_closure (non-Sunday closures,
  capped at 14; agent addition, see decisions), n_outlets (per depot x brand).
- `task2a/analysis.py`: plots and numbers; console output saved to
  `task2a/reports/phase2_analysis_output.txt`, numbers to
  `task2a/reports/phase2_analysis.json`.
- Added `matplotlib==3.11.1` to `requirements.txt`.

**Checks (all pass)**
- Payday rule reproduces all 59 calendar paydays.
- festival_ramp > 0 only in the 9 days before a festival; ramp = 1 on festival days.
- No missing feature values in history or the forecast window.
- Forecast window = exactly 2026-W14..W23 (matches task2a_test_inputs.csv),
  7 days per series-week, contiguous with history.
- Operating days 2026: W16 = 4, W18 = 5, W22 = 6 (others 6).

**Key numbers** (method: volume / centered 57-day mean of clean operating days;
payday, ramp and catch-up effects are also weekday-adjusted)
- Weekday (Fresh): Mon 0.95, Tue 0.92, Wed 0.93, Thu 1.02, Fri 1.07, Sat 1.11;
  Saturday +19.8% vs Wednesday (plan said ~17%).
- Style weekday schedule is **depot-specific**: Kandy delivers Mon/Wed/Thu/Sat,
  Peliyagoda Mon/Wed/Thu/Fri (Tue zero for both). Tech: strong weekday pattern
  (Mon 0.25, Fri 1.80 pooled).
- Payday rule: the 25th and the last day of the month; a Sunday payday moves
  back to Saturday (9 moved). Christmas paydays (2024-12-25, 2025-12-25) fall
  on closed days. Effect: Fresh x1.12 on payday and also x1.12 on the next two
  days (3-day window); Style x1.08; Tech x1.33.
- Festival ramp (Fresh): rises roughly linearly to x1.31-1.32 at ramp 0.8-0.9
  (x1.24 on operating festival days). Style up to x1.47; Tech noisy.
- Festival dates: thai_pongal, new_year and christmas are almost fixed;
  vesak (05-23, 05-12, 05-01), poson (06-21, 06-10, 05-30), esala (08-19,
  08-08) and deepavali (10-31, 10-20) move about 11 days earlier each year.
  2026: new_year Mon 04-13 (closed, plus Tue 04-14 closed), vesak Fri 05-01
  (closed; in 2024/2025 vesak was operating and May 1 was a separate closure),
  poson Sat 05-30 (operating, also a payday).
- New Year (Fresh, both depots): last 3 days before x1.63 (2024) and x1.67
  (2025), peak x1.82 and x1.76; days +1..+3 after x0.91-0.92 (no rebound).
- Catch-up after a 2+ day closure: Fresh x0.96 (only 2 events, both after New
  Year), Style x0.99, Tech x0.49 (4 series-days each; too few to model).
  After a 1-day mid-week closure (May Day, Christmas): Fresh x1.12, Style x1.07,
  Tech x1.06.
- Chilled share (Fresh): seasonal, 0.355-0.36 in Oct-Jan, rising to
  0.377-0.382 in May-Jun; Kandy 0.363, Peliyagoda 0.368 overall.
- Style seasonality: spikes are in **fixed ISO weeks**, W10 (first full week
  of March; 2024, 2025, 2026, no festival), W32 (first full week of August, in
  both years even though esala moved from W34 to W32), W15 (New Year build-up),
  W51 (Christmas); deepavali spikes follow the festival (W44 2024, W42-43 2025).
  Vesak/Poson/Thai Pongal show no Style effect (festival-week ratio 0.92-1.04).
  So Style peaks are mostly calendar-driven, with New Year, Deepavali and
  Christmas effects.
- Year-on-year growth (Jan-Mar, volume per operating day): Fresh +5.1%
  (2025 vs 2024), +6.2% (2026 vs 2025); Style +1.3%, +1.7%; Tech -9.1%, +27.6%.

**Decisions / caveats**
- Added `days_to_next_holiday_closure` alongside `days_to_next_closure`, because
  the latter is almost fully determined by the weekday (Saturday -> 1).
- Thai Pongal ratios are understated: their pre-festival baseline window
  overlaps the Christmas / year-end period.
- Report files contain aggregates derived from the datasets; push only after
  the repo is confirmed private.

**Authorship**
- Agent wrote `task2a/prepare_features.py`, `task2a/analysis.py` and this entry.
  User specified the extra features and analyses and the forecast-window checks.

**Phase 2 decisions (recorded at Phase 2 approval)**
- Keep `days_to_next_holiday_closure`; drop `days_to_next_closure` (collinear with weekday).
- Style: depot x weekday structure, plus `style_peak_week` (ISO weeks 10, 32, 51).
  Festival effects vary by festival (ramp x festival_name).
- Payday: separate d0/d1/d2 flags; test calendar-day vs operating-day counting.
- Tech: no level shift; the backtest chooses a smoothing window (Phase 4).
- Committing report aggregates locally is fine; push only after the repo is
  private and the user says "PUSH" (`git push -u origin datathon-task2a-task1-fixes`).
- Phase 2 committed as `f60c759`. Not pushed (repo still public).

---

## Phase 3: Backtest framework, baselines and GLM (2026-10-07)

**Done**
- `task2a/prepare_features.py`: dropped `days_to_next_closure`; added
  `payday_cal_d0..d2` (0/1/2 calendar days after the latest payday),
  `payday_op_d0..d2` (payday itself if operating, then the 1st/2nd operating
  day after it), `after_midweek_closure` (first operating day after a single
  closed non-Sunday day), `style_peak_week` (ISO weeks 10, 32, 51). Checked
  that the 2026-W22 payday tail (May 30, moved from Sun May 31) spills into
  W23: calendar counting gives d2 = Mon Jun 1; operating counting gives
  d1 = Jun 1, d2 = Jun 2.
- `task2a/models.py`: `SeriesGLM` (scikit-learn `PoissonRegressor`, log link,
  alpha = 1e-4) per depot x brand series and target, fitted on operating days.
  Features: weekday dummies, payday d0/d1/d2, ramp and ramp^2 per festival
  (7 festivals), after_midweek_closure, 1/2/3 days before a holiday closure,
  month dummies, trend in years, style_peak_week (Style only). Closed days and
  structural-zero weekdays (< 1% of the series' mean daily volume in training:
  Style Tue, Kandy Style Fri, Peliyagoda Style Sat) are forced to 0. Daily
  predictions summed to ISO weeks. Also weekly baselines: last-8-weeks mean,
  same week last year, 13- and 26-week means, EWM (half-life 8 weeks).
- `task2a/backtest.py`: 4 rolling test periods (train on all weeks before the
  period, forecast 10 weeks): 2025-W14..W23 (headline), 2026-W04..W13,
  2025-W30..W39 (Esala, Style W32), 2025-W40..W49 (Deepavali). Outputs in
  `task2a/reports/`: backtest_results.csv (every weekly prediction),
  backtest_metrics.csv, backtest_glm_multipliers.csv, backtest_headline.png,
  backtest_output.txt.

**Key numbers (WAPE of weekly volume, mean of the 4 periods; headline in brackets)**
- Total, overall: GLM (payday cal) 3.4% [3.8%]; last-8 mean 8.7% [15.0%];
  same week last year 9.7% [12.9%]; 13/26-week means and EWM 8.2% [14.1-14.3%].
- Chilled (Fresh), overall: GLM 2.5% [2.3%]; last-8 mean 7.8% [15.3%].
- Per brand, total: Fresh GLM 2.1% vs last-8 7.4%; Style GLM 5.6% vs 12.3%;
  **Tech GLM 31.7% vs last-8 31.8%, 26-week mean 28.7%** (GLM adds nothing for Tech).
- Payday counting: calendar days better (total 3.38% vs 3.52%; Fresh 2.05% vs
  2.21%; chilled 2.50% vs 2.58%), better in 3 of 4 periods. **Chosen: calendar.**
- Plan prototype check: last-8 mean headline = 15.0% (matches the plan);
  GLM 3.8% vs the prototype's 5.9%.
- Peliyagoda Fresh multipliers (GLM on all history vs Phase 2): weekday vs Wed
  Mon 1.08/1.02, Tue 1.07/0.99, Thu 1.17/1.10, Fri 1.16/1.15, Sat 1.25/1.20;
  payday d0/d1/d2 1.146/1.146/1.142 vs 1.12/1.12/1.12; New Year ramp 1.33 at
  0.4, 1.62 at 0.8, 1.69 at 0.9 (Phase 2 festival-aligned New Year x1.63-1.82;
  the pooled all-festival ramp is only x1.32, so festival-specific ramps
  matter); trend +5.7%/yr; after_midweek_closure 1.00 (Phase 2 pooled 1.12);
  1 day before a holiday closure x1.04.
- Headline weekly errors, Peliyagoda Fresh: -3.1% to +3.4% every week
  (W15 1,321 vs 1,351; W16 619 vs 624). Peliyagoda Style: -14% to +4%, mostly
  under-forecast (W18, the May Day week, -14%).

**Observations for Phase 4**
- Tech: GLM overreacts (e.g. Kandy Tech 2025-W18 predicted 2.6 vs actual 12.7);
  smoothed means are better; choose a window in Phase 4.
- Style: under-forecasts most headline weeks (level), and Kandy Style
  over-forecasts the closure weeks W16/W18.
- Peliyagoda Fresh weekday multipliers are a bit higher than the pooled Phase 2
  values (different adjustment set and depot); same shape.

**Authorship**
- Agent wrote `task2a/models.py`, `task2a/backtest.py`, the feature changes and
  this entry. User specified the GLM feature set, test periods, payday test and
  report contents.

**Phase 3 decisions (recorded at Phase 3 approval)**
- Drop the 2+ day closure flag; keep `after_midweek_closure` only if an ablation
  helps on the mean of the 4 periods (2026-W18: Sat May 2 follows the Vesak closure).
- Tech: compare (a) smoothed mean, (b) GLM + smoothing blend, (c) calendar-shaped level.
- Style: test outlet x weekday and recency weighting / shorter window.
- Phase 3 committed as `16c843e`. Not pushed (repo still public).

---

## Phase 4: Task 2A model improvements and chilled share (2026-10-07)

**Selection discipline**
- Selection uses only the mean over the 4 Phase 3 periods (headline shown
  separately). A 5th period, 2025-W02..W11, is a confirmation period: evaluated
  once, at the end, on the recommended config and the baselines
  (`experiments.py --confirm` refuses a second run without `--force`).
- Every results table includes the mean signed bias per series.

**Done**
- `prepare_features.py`: dropped `is_first_day_after_closure` (analysis.py derives
  it locally; Phase 2 numbers unchanged). New `task2a/data/style_outlet_days.csv`:
  one row per Style outlet per date of its weekday. Asserted: every Style outlet
  orders on exactly one weekday, has an order on every operating day of that
  weekday, and none on closed days (2,890 orders, 25 outlets). A closed day
  skips that week's order; it is not moved.
- `models.py`: SeriesGLM gained `include_midweek`, `halflife_weeks` (recency
  sample weights) and `window_weeks`; new `StyleOutletGLM` (Poisson GLM on
  order volume per outlet-date, outlet dummies instead of weekday dummies);
  `GlobalHGB` (HistGradientBoostingRegressor, loss="poisson", depot/brand/festival
  as native categoricals, min_samples_leaf 40, l2 1.0, no early stopping, seed 42);
  Tech smoothers and the calendar-shaped level (smoothed volume per unit of
  weekday capacity x the forecast week's capacity); chilled share table
  (depot x month).
- `forecast.py`: `Context` + `Forecaster` turn a config (spec per brand + chilled
  method) into weekly forecasts; enforce total >= 0, 0 <= chilled <= total,
  chilled = 0 for Style/Tech. Shared by the backtest and the Phase 5 final fit.
- `evaluation.py`: periods, contexts, scoring (WAPE, MAE, signed bias).
- `experiments.py`: all experiments, improvement table, config, confirmation.
- Outputs in `task2a/reports/`: phase4_output.txt, phase4_improvement.csv,
  phase4_candidates.csv, phase4_config.json, phase4_confirmation.csv,
  phase4_confirmation_output.txt.
- Checked: the Phase 3 backtest still reproduces byte-for-byte.

**Key numbers (total WAPE, mean of 4 periods / headline)**
- after_midweek_closure ablation: with 3.377% / 3.817%, without 3.362% / 3.767%
  -> **dropped** (consequence: no lift for Sat 2026-05-02 after the Vesak closure).
- HGB grid (8 combos: depth 3/4, lr 0.03/0.06, 300/600 iter): best depth 4,
  lr 0.06, 600 iter: 3.56% / 3.87% alone (Fresh 2.13%, Style 7.03%, Tech 29.5%).
  The best combo is at the edge of the grid; not extended (one modest grid).
- GLM+HGB blend, weight on HGB chosen per brand: Fresh 0.4 (2.04% -> 1.97%),
  Style 0.2 (5.64% -> 5.58%), Tech 0.7 (31.6% -> 28.8%).
- Style: outlet x weekday 5.10% / 5.17% (GLM 5.64% / 7.34%); half-life 52w
  5.099% vs 5.102% (practical tie; the rule picked 52w); half-life 26w 5.23%;
  52-week window much worse (outlet 9.0%, GLM 13.4%, it loses last year's
  festivals); blending HGB into the outlet model: best weight 0.
- Tech: (a) best smoothed mean = 26-week mean 28.75%; (b) GLM + smoothed
  (w_smooth 0.8) 28.43%; (c) calendar-shaped, 52-week mean 28.22% (headline
  33.6%) -> chosen. All within 0.5 pp; Tech is mostly noise.
- Chilled: share (depot x month) x final Fresh total 2.24% / 2.30% vs GLM on
  chilled directly 2.50% / 2.27% -> **share** chosen.
- Final config: 3.14% / 3.47% (Phase 3 GLM 3.38% / 3.82%); per period:
  headline 3.47%, 2026-W04..W13 2.77%, esala 3.64%, deepavali 2.69%. Per series
  WAPE / bias: Kandy Fresh 1.93% / -0.4%, Kandy Style 7.08% / -0.6%, Kandy Tech
  27.5% / -1.3%, Peliyagoda Fresh 2.00% / -0.5%, Peliyagoda Style 3.97% / -1.2%,
  Peliyagoda Tech 28.8% / -7.4%.

**Confirmation period 2025-W02..W11 (evaluated once)**
| model | total WAPE | bias | Fresh | Style | Tech | chilled |
|---|---|---|---|---|---|---|
| recommended config | 3.47% | -1.6% | 1.79% | 8.47% | 32.1% | 2.48% |
| Phase 3 GLM (reference) | 3.48% | -2.2% | 1.98% | 7.07% | 33.2% | 3.67% |
| last-8-weeks mean | 5.73% | -0.4% | 4.55% | 8.18% | 31.3% | 3.99% |
| same week last year | 5.26% | -3.0% | 4.52% | 5.27% | 29.1% | 4.72% |
- Fresh and chilled confirm the gains; Style does not (Kandy Style bias -10.2%).
  This period trains on only ~1 year of history (2024), unlike every selection
  period and the final fit.

**Not done**
- Optional P50/P90 intervals (Phase 4 item 5): skipped for time.

**Authorship**
- Agent wrote `models.py` (extensions), `forecast.py`, `evaluation.py`,
  `experiments.py`, the feature changes and this entry. User set the selection
  discipline, the confirmation period, the candidate list and the HGB constraints.

**Phase 4 decisions (user, before commit)**
- Final config accepted with one change: Style uses the plain outlet x weekday GLM
  (no recency weighting); it ties the 52w half-life on the selection periods
  (5.102% vs 5.099%) and is simpler. The rule's pick is kept in
  `phase4_config.json` under `selected_by_rule`.
- **The confirmation period (2025-W02..W11) was evaluated once and was not used
  for any selection.** No post-hoc blending (e.g. with "same week last year")
  after seeing it, and it was not re-run after the Style change.
- Tech: option (c) kept; Peliyagoda Tech bias (-7.4%, about 2 m3/week) accepted.
- HGB grid left as is.
- Re-scored the edited config on the 4 selection periods only
  (`experiments.py --score-config`): total WAPE 3.14% mean / 3.47% headline
  (Fresh 1.97%, Style 5.10%, Tech 28.22%, chilled 2.24%), within the user's
  3.17% threshold -> Phase 4 approved.
- Phase 4 committed as `63c5b33`. Not pushed (repo still public).

---

## Phase 5: Task 2A final fit and submission (2026-10-07)

**Done**
- `forecast.py` refactored into `fit_spec`/`predict_spec` and `fit_config`/
  `predict_config`; the backtest `Forecaster` and the final fit run the same
  functions. Verified: `experiments.py --score-config` output is byte-identical
  before and after the refactor (3.14% / 3.47%). Removed the unused
  `calendar_level_forecast` from `models.py`.
- `experiments.py`: selection mode no longer overwrites a config carrying
  `user_decisions` (writes `phase4_config_rule.json` instead).
- `train_model.py`: fits the approved config (`phase4_config.json`) on all
  history 2024-01-01..2026-03-29 (4,914 daily rows, 2,925 Style outlet-days);
  saves `task2a/models/task2a_model.joblib` (gitignored) and
  `task2a_model_summary.json` (fitted multipliers, Tech levels and weekday
  weights, chilled shares, versions). Seeds: HGB random_state 42 (the only
  stochastic component; no early stopping), numpy seed 42.
- `predict.py`: `predict_task2a(test_inputs_df)` loads the saved bundle only (no
  retraining), reads the forecast-window calendar features from task2a/data, and
  returns row_id + the 2 prediction columns in input order.
- `make_submission.py`: fills the template, rounds to 3 decimals, writes
  `outputs/submission_task2a.csv`. Asserts: 60 rows; row_id values and order
  identical to the template; no missing or negative values; chilled 0 for
  Style/Tech; 0 <= chilled <= total; chilled > 0 for Fresh.
- `forecast_report.py`: sanity table (`task2a/reports/final_forecast_table.csv`,
  console in `final_forecast_output.txt`) and `final_forecast.png`.
- `run_all.py`: runs labels -> features -> final fit -> submission -> report
  (`--full` adds analysis, Phase 3 backtest and the Phase 4 config score).
- `GlobalHGB` stores its structural-zero weekdays as a sorted tuple instead of a
  set: with a set, the pickled model differed between processes (string hash
  randomization) although predictions were identical.

**Reproducibility**
- Deleted `task2a/data/` and `task2a/models/`, ran `run_all.py` from scratch twice:
  `submission_task2a.csv`, `task2a_model.joblib`, the model summary and the
  sanity table are byte-identical (cmp), and `pandas.DataFrame.equals` passes.

**Key numbers (forecast, 10 weeks)**
- Totals over W14..W23: Fresh 15,863 m3 (chilled 5,970, 37.6%), Style 2,220,
  Tech 516.
- Fitted: Fresh blend GLM 0.6 / HGB 0.4 (HGB 600 iterations); Tech level per
  weekday-capacity unit Kandy 3.76, Peliyagoda 5.32; chilled share Apr-Jun
  0.371-0.382.
- 10 weeks flagged (>25% from the last-8 mean), all explained by the calendar:
  - W15 up (New Year build-up, ramp to 0.8): Fresh +40%/+42%, Style +49%/+61%;
    within +4% to +6% of 2025-W15 for Fresh and Peliyagoda Style, -8% Kandy Style.
  - W16 down (Mon 13 + Tue 14 Apr closed; 4 operating days): Fresh -32%/-33%,
    Kandy Style -37% (2 of 9 outlets skip), Peliyagoda Tech -34%. Peliyagoda
    Style -20% (not flagged). Fresh within +6% of 2025-W16.
  - W18 down (Fri 1 May closed for Vesak): Peliyagoda Style -31% (5 of 16 outlets
    skip their Friday order), Peliyagoda Tech -45% (Friday is a main Tech day).
- Not flagged but worth noting: W22 Fresh +25% vs last-8 (two paydays May 25 and
  May 30 with their tails, Poson ramp and Poson on Sat 30 May, open), +24-25% vs
  2025's Poson week, which had no paydays. Kandy Style W18 +8% and Kandy Tech W18
  -10%: Kandy Style has no Friday outlets and Kandy Tech only one Friday outlet,
  so Vesak's Friday closure barely affects them.

**Authorship**
- Agent wrote `train_model.py`, `predict.py`, `make_submission.py`,
  `forecast_report.py`, `run_all.py`, the `forecast.py` refactor and this entry.
  User specified the checks, the reproducibility test and the sanity table.

**Phase 5 decisions (user, before commit)**
- W22 check (in-sample only, final Fresh models, no backtest periods;
  `task2a/insample_payday_ramp_check.py`, output in
  `reports/insample_payday_ramp_check.txt`): actual / fitted on Fresh operating
  days. Payday window (d0-d2) overlapping ramp > 0: mean daily ratio 0.9993
  (sum ratio 1.0020; only 14 series-days: deepavali, esala, poson); payday only
  0.9993; ramp only 1.0000; neither 0.9999. History weeks with 2 paydays (7):
  weekly ratios 0.990-1.013 (mean 1.003). Weeks with a payday and ramp > 0 (12):
  0.984-1.016 (mean 1.002). The multiplicative payday x ramp combination is not
  biased in sample -> **W22 forecast kept** (rule: keep if 0.95-1.05).
- Track `task2a/models/task2a_model_summary.json`; the `.joblib` stays ignored.
