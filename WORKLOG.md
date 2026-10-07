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
- Phase 5 committed as `1c66e4b`. Not pushed (repo still public).

---

## Phase 6: Task 1 fixes, part 1: dropped-column bug and time-based validation (2026-10-07)

**Label audit** (`task1/label_audit.py`, read-only; output `task1/reports/label_audit.txt`)
- No bug found; labels unchanged.
- Times are parsed to minutes before every comparison (prepare_labels: int h*60+m;
  prepare_features: pd.to_datetime %H:%M). All 8 time columns are zero-padded
  HH:MM with 0 unparseable values; an independent re-derivation reproduces every
  service_minutes and late label.
- Midnight: no crossings (0 arrivals before departure, 0 leave-before-arrival,
  0 planned arrival before planned departure, 0 windows closing before opening).
  5,029 legs have times before 03:00 (departures from 02:00); all same-day.
  Latest leave_outlet_time 23:33.
- Early arrivals: 4,023 of 91,894 (4.38%; Fresh 4.4%, Style 5.1%, Tech 1.7%),
  median wait 14 min (max 132); the label removes the wait (mean service 19.00
  min vs 19.88 if the wait were counted). 0 departures before the window opens.
- Late rate (arrival strictly after close): 19.58% overall; Fresh 20.35%,
  Style 5.49%, Tech 2.46%; 279 arrivals exactly at close count as on time.
- Order and route-leg copies of brand, district, depot, vehicle_id,
  vehicle_type, vehicle_temp and planned arrival agree 100% (train and test).
- Service-time tail: p99 80 min, max 428; 45 stops > 180 min, 42 of them
  Style/Tech (p99 168 and 195 min). Kept.

**Bug fix**
- `task1/prepare_features.py`: the order and leg files both carry brand, district,
  depot, vehicle_type, vehicle_temp and planned_arrival_time, so the merge renamed
  them `_order`/`_leg` and the candidate-list filter silently dropped them. Now
  asserts the two copies are equal and restores the plain name. Explicit
  `FEATURES` list in the new `task1/features.py` (17 features); prepare_features
  and train_model assert every listed column exists. `train_features.csv` now also
  carries `delivery_id` and `order_date` (not features) for the time-based split.
- `time_to_minutes` now asserts no unparseable times (errors="coerce" could have
  hidden NaNs).
- `task1/baseline_eval.py` selects the original 12 features explicitly, so it still
  reproduces after the change: re-run, `baseline_metrics.json` byte-identical.

**Training and evaluation**
- New `task1/evaluation.py` (random split as the original, time-based split
  order_date >= 2026-01-03, metrics). `task1/train_model.py` trains the same Random
  Forests (300 trees, min_samples_leaf 2, balanced class weight for lateness,
  random_state 42) on both splits; only the features changed. Writes
  `task1/reports/metrics_phase6.json`. It still saves the random-split models as
  before; `outputs/submission_task1.csv` is not regenerated (Phase 9).

| split | metric | baseline (12 feat.) | Phase 6 (17 feat.) |
|---|---|---|---|
| random | service MAE / RMSE | 5.520 / 8.950 | 5.306 / 8.725 |
| random | late AUC / log loss / Brier | 0.9278 / 0.2900 / 0.0916 | 0.9319 / 0.2845 / 0.0913 |
| random | share p = 0 | 12.1% | 1.4% |
| time-based | service MAE / RMSE | 5.152 / 8.319 | 4.950 / 8.032 |
| time-based | late AUC / log loss / Brier | 0.9315 / 0.2320 / 0.0727 | 0.9387 / 0.2315 / 0.0726 |
| time-based | share p = 0 | 14.2% | 4.3% |

**Authorship**
- Agent wrote `task1/label_audit.py`, `task1/features.py`, `task1/evaluation.py`,
  the changes to `prepare_features.py`, `train_model.py` and `baseline_eval.py`, and
  this entry. User specified the audit items, the same-models rule and the metrics.

**Phase 6 decisions (recorded at Phase 6 approval)**
- Phase 9: retrain the final Task 1 models on ALL training data (after validation).
- **Push decision:** the team chose to push the branch to the public origin
  (`https://github.com/tithu018/Datathon`), overriding the earlier "no push while
  public" rule. From Phase 6 on, the branch `datathon-task2a-task1-fixes` is pushed
  after each approved commit. Never main, never force-push, no other remotes.
  Before each push, `git ls-files` is checked for dataset CSVs and .joblib files.
- Phase 6 committed as `44dc070` and pushed: `git push -u origin datathon-task2a-task1-fixes`
  created the remote branch (new branch, tracking set up). Pre-push check: no dataset
  CSVs, no Task 1 generated CSVs and no .joblib/.pkl tracked or added in any branch
  commit; the datasets remain only in main's history (eaf7297), already on origin.

---

## Phase 7: Task 1 fixes, part 2: new features (2026-10-07)

**Done**
- `task1/feature_groups.py`: candidate groups from planned information only
  (outlets.csv, service_allowance.csv, calendar.csv, traffic_speed.csv and the
  PLANNED leg columns). Route context is computed from the route-leg tables
  (planned departure, planned travel, planned arrival).
- `task1/features.py`: BASE_FEATURES (Phase 6), FEATURE_GROUPS, EXTRA_CANDIDATES,
  KEPT_GROUPS and FEATURES; `assert_no_leakage` rejects actual_*, arrival_time,
  leave_outlet_time and the derived label columns (asserted for all candidates,
  in prepare_features, train_model and the experiment).
- `task1/models.py`: the original Random Forests built for a feature list (shared
  by train_model.py and phase7_features.py). `train_model.py` now writes
  `reports/metrics.json` (latest run); `metrics_phase6.json` stays as the Phase 6 record.
- `task1/phase7_features.py`: cumulative groups, same Random Forests, keep rule on
  the time-based test: lower service MAE or late log loss by >= 0.5% (relative)
  and neither worse by > 1%; rejected groups are not carried forward.
  Outputs `reports/phase7_feature_groups.csv`, `reports/metrics_phase7.json`,
  `reports/phase7_output.txt`.

**Finding: the specified route-context feature is identically 0.** In the route
legs, the planned dwell (next leg's planned departure minus this planned arrival)
equals the service allowance at every stop that has a next leg (66,696 of 66,696),
so "cumulative allowance of earlier stops minus planned dwell" is 0 everywhere. It
is kept in the route_context group as specified (a constant; no effect). The plan
does NOT schedule the wait when a vehicle arrives before the window opens; the sum
of those unscheduled waits at earlier stops (`cum_unplanned_wait_before`, planned
information) was measured as a labelled extra and NOT applied (user decision).

**Results (time-based test; cumulative)**
| step | group | kept | MAE | log loss | AUC | Brier | share p=0 |
|---|---|---|---|---|---|---|---|
| 0 | Phase 6 base | - | 4.950 | 0.2315 | 0.9387 | 0.0726 | 4.3% |
| 1 | outlet (dock, parking, mall window) | yes | 4.795 (-3.2%) | 0.2313 | 0.9406 | 0.0727 | 2.8% |
| 2 | service_allowance | no | -0.1% | +0.6% | | | |
| 3 | planned_slack | no | -1.0% | +1.1% | | | |
| 4 | route_context | yes | 4.734 (-1.3%) | 0.2269 (-1.9%) | 0.9443 | 0.0710 | 5.5% |
| 5 | calendar (payday, ramp, holiday) | yes | 4.148 (-12.4%) | 0.2185 (-3.7%) | 0.9507 | 0.0678 | 14.0% |
| 6 | traffic (speed_index) | yes | 4.136 (-0.3%) | 0.2110 (-3.4%) | 0.9499 | 0.0657 | 17.5% |
| extra | cum_unplanned_wait_before | not applied | +0.2% | -0.1% | | | |
- Final: 30 features. Time-based MAE 4.136 / RMSE 6.373 / AUC 0.9499 / log loss
  0.2110 / Brier 0.0657 (baseline 5.152 / 8.319 / 0.9315 / 0.2320 / 0.0727).
  Random split MAE 4.482, AUC 0.9415, log loss 0.2661.
- Calendar check: service time rises with the festival ramp (17.2 -> 31.8 min at
  ramp > 0.5, while order units rise only 47 -> 59), on paydays (18.6 -> 24.1) and
  holidays (36.3 min): busy days slow receiving. Known at planning time.
- `train_model.py` with KEPT_GROUPS reproduces the Phase 7 final metrics exactly.
- Share of p = 0 rose to 17.5%: calibration is Phase 8.

**Late rate (for Phase 8 calibration)**
- Strongly seasonal, following the monsoon flag: monsoon = 1 27.4%, monsoon = 0
  12.8%. By month: Jan-Feb 11-14%, Mar-Jun 26-29%, Jul-Sep 11-15%, Oct-Nov 24-30%,
  Dec 13-14%; the same in 2024 and 2025.
- The time-based test (2026-01-03..02-14, all monsoon = 0) has 13.1% vs 20.0% in its
  training part; Jan-Feb was 12.3% (2024), 12.5% (2025), 13.3% (2026), so the gap is
  seasonal, not drift. The real Task 1 test (2026-02-16..03-28) spans late Feb
  (monsoon 0) and March (monsoon 1), so its late rate should be higher than the
  validation period: calibrating on the time-based test alone would bias low.

**road_conditions.csv (reported, not used; user decision)**
- district, date, disruption_index (100 = clear, min 40); 10,920 rows = 12 districts x
  910 dates (2024-01-01..2026-06-28). Covers all 432 test district-dates; 66% of
  them are below 100 (57% overall).
- Strong relation in training: late rate 59% at index <= 60, 36% at 61-80, 17% at
  81-99, 14% at 100; service time 22.6 / 19.9 / 18.7 / 18.7 min.

**Authorship**
- Agent wrote `task1/feature_groups.py`, `task1/models.py`, `task1/phase7_features.py`,
  the `features.py`/`prepare_features.py`/`train_model.py` changes and this entry.
  User specified the groups, their order, the keep rule's intent, the leakage
  assert and the reports.

**Phase 7 decisions and re-run (user, before commit)**
- **Road conditions added** as the final group (`disruption_index`, district x
  dispatch date). Justification: predictions are made on the dispatch morning,
  before the delivery starts (booklet wording), when road advisories are known; the
  organizers supplied the file covering the test dates.
- **Dropped** the constant `cum_allowance_minus_dwell_before`. Finding: the planned
  dwell equals the service allowance at every stop that has a next leg (66,696 of
  66,696), so the feature was identically 0. The unscheduled-waits extra
  (`cum_unplanned_wait_before`) is **not applied** (re-run: MAE +0.14%, log loss +0.69%).
- **Keep rule thresholds** (user-confirmed): a group is kept if it lowers time-based
  service MAE or late log loss by >= 0.5% (relative) and neither gets worse by > 1%;
  rejected groups are not carried forward.
- Re-run (the route-context group without the constant column changes the
  classifier's feature sampling, so all steps were re-measured; same decisions):
  | step | group | kept | MAE | log loss | AUC | Brier |
  |---|---|---|---|---|---|---|
  | 0 | Phase 6 base | - | 4.950 | 0.2315 | 0.9387 | 0.0726 |
  | 1 | outlet | yes | 4.795 (-3.2%) | 0.2313 | 0.9406 | 0.0727 |
  | 2 | service_allowance | no | -0.1% | +0.6% | | |
  | 3 | planned_slack | no | -1.0% | +1.1% | | |
  | 4 | route_context | yes | 4.733 (-1.3%) | 0.2270 (-1.9%) | 0.9444 | 0.0709 |
  | 5 | calendar | yes | 4.146 (-12.4%) | 0.2175 (-4.2%) | 0.9503 | 0.0675 |
  | 6 | traffic | yes | 4.138 (-0.2%) | 0.2105 (-3.2%) | 0.9497 | 0.0655 |
  | 7 | road_conditions | yes | 4.134 (-0.1%) | **0.1739 (-17.4%)** | **0.9687** | 0.0535 |
- Final Phase 7 set: 30 features (KEPT_GROUPS outlet, route_context, calendar,
  traffic, road_conditions). Time-based: MAE 4.134, RMSE 6.380, AUC 0.9687, log loss
  0.1739, Brier 0.0535, share p = 0 18.3%. Random split: MAE 4.456, AUC 0.9700, log
  loss 0.1986. `train_model.py` reproduces these exactly.
- make_submission.py must select FEATURES (36 candidate columns in test_features.csv):
  fixed in Phase 9.
- Phase 7 committed as `c425c96` and pushed. Note: staging `task1/` as a whole also
  committed the (then unrun) Phase 8 script `task1/phase8_models.py` and the HGB
  builders in `task1/models.py`; they do not affect any Phase 7 result. Not rewritten
  (already pushed); Phase 8 results are committed with Phase 8.

---

## Phase 8: Task 1 models and calibration (2026-10-07)

**Setup** (`task1/phase8_models.py`, 30 Phase 7 features; selection on the mean of two tests)
- A: order_date >= 2026-01-03 (train 86,721 / test 5,173; late 13.1%; monsoon share 0).
- B: 2025-02-16..03-28, season-matched to the real test (train 49,055 / test 4,856;
  late 21.4%; monsoon share 0.68).
- Calibration: CalibratedClassifierCV(cv=StratifiedKFold(5, shuffle=True, seed 42))
  on the training portion, so every fold spans all seasons. HGB: max_leaf_nodes 31,
  min_samples_leaf 40, l2 1.0, no early stopping, random_state 42.

**Service time (MAE / RMSE, mean of A and B)**
- RF (current) 4.091 / 6.278. Best: HGB squared_error lr 0.05, 600 iter 3.896 / 6.062
  (-4.8% MAE; A 3.921, B 3.871). HGB absolute_error best 3.917 / 6.354.
  All 8 HGB settings beat the RF on MAE.

**Lateness (mean of A and B; raw -> clipped [0.01, 0.99])**
| model | log loss | Brier | AUC | mean pred A / B (actual 0.131 / 0.214) |
|---|---|---|---|---|
| RF balanced (current) | 0.1894 -> 0.1926 | 0.0585 | 0.9692 | 0.183 / 0.266 |
| RF no class weight | 0.1668 -> 0.1707 | 0.0515 | 0.9703 | 0.134 / 0.208 |
| HGB raw (lr 0.05, 600) | 0.1530 -> 0.1581 | 0.0482 | 0.9734 | 0.131 / 0.204 |
| HGB + isotonic | 0.1508 -> 0.1562 | 0.0476 | 0.9738 | 0.131 / 0.205 |
| HGB + sigmoid | 0.1513 -> 0.1562 | 0.0475 | 0.9739 | 0.131 / 0.205 |
- RF balanced over-predicts (mean 0.183 vs 0.131 in A); its reliability shows e.g.
  bin 0.3-0.4 -> actual 0.12. Share of exact p = 0: RF balanced 16.7%, RF no weight
  22.5%, isotonic 48.8%, HGB raw and sigmoid 0%.
- Calibrated HGB tracks the late rate in both seasons (0.131 vs 0.131; 0.205 vs 0.214).
- Outputs: task1/reports/phase8_service.csv, phase8_late.csv, phase8_reliability.csv,
  calibration.png, metrics_phase8.json, phase8_output.txt.

**Authorship**
- Agent wrote `task1/phase8_models.py`, the HGB builders in `task1/models.py` and this
  entry. User specified the test periods, candidates, calibration method and clipping.

**Phase 8 decisions (user, before commit)**
- Final config: service time = HistGradientBoostingRegressor (squared_error, lr 0.05,
  600 iter, max_leaf_nodes 31, min_samples_leaf 40, l2 1.0, seed 42); lateness = the
  same HGB classifier + sigmoid calibration, CalibratedClassifierCV(cv=StratifiedKFold(5,
  shuffle=True, random_state=42)). Grid left as is.
- **Clip [0.001, 0.999]** instead of [0.01, 0.99] (fixed choice, not tuned): sigmoid
  produces no exact 0/1, so a tight clip only costs log loss; the looser clip still
  guarantees finite log loss. `task1/phase8_clip.py` -> `reports/metrics_phase8_clip.json`:
  | period | raw | clip 0.01 | clip 0.001 (final) | Brier | AUC | mean pred / actual |
  |---|---|---|---|---|---|---|
  | A | 0.1396 | 0.1447 | **0.1399** | 0.0435 | 0.9720 | 0.1315 / 0.1311 |
  | B | 0.1630 | 0.1677 | **0.1633** | 0.0516 | 0.9757 | 0.2055 / 0.2136 |
  Mean log loss 0.1516 (0.01 clip: 0.1562).
- Phase 8 committed as `7518d0b` and pushed.

---

## Phase 9: Task 1 final fit, submission and change report (2026-10-07)

**Done**
- `task1/models.py`: `make_final_models` (HGB regressor; HGB classifier + sigmoid,
  CalibratedClassifierCV over StratifiedKFold(5, shuffle, seed 42)), `predict_late`
  (clip [0.001, 0.999]).
- `task1/train_model.py`: validates the original pipeline (12 features, Random
  Forests) and the final config on tests A and B, then fits the final models on
  ALL 91,894 training rows; saves the .joblib files (gitignored) and
  `reports/metrics.json`. `--no-eval` skips validation.
- `task1/make_submission.py`: selects FEATURES explicitly (test_features.csv also
  holds rejected candidate columns); asserts 5,014 rows, delivery_id values and order
  identical to the template, no missing values, service >= 0, probabilities within
  [0.001, 0.999].
- `task1/compare_submission.py` -> `reports/submission_comparison.txt`.
- `task1/run_all.py`: labels -> features -> validation + final fit -> submission ->
  comparison.
- `task1/CHANGES_FOR_REVIEW.md` for the Task 1 owner (issues, fixes with commits,
  metrics, feature groups, sklearn 1.9 note, re-run steps, merge warning, open questions).

**Validation (original pipeline -> final)**
| period | MAE | RMSE | AUC | log loss | Brier | p = 0 |
|---|---|---|---|---|---|---|
| A 2026-01-03..02-14 | 5.152 -> 3.921 | 8.319 -> 6.124 | 0.9315 -> 0.9720 | 0.2320 -> 0.1399 | 0.0727 -> 0.0435 | 14.2% -> 0% |
| B 2025-02-16..03-28 | 5.058 -> 3.871 | 7.414 -> 6.000 | 0.9234 -> 0.9757 | 0.3139 -> 0.1633 | 0.1011 -> 0.0516 | 8.9% -> 0% |
- In-sample mean late probability after the final fit 0.1965 vs actual 0.1958.

**New submission vs original**
- Mean predicted late rate 0.2126 (original 0.2506); February (monsoon 0) 0.132,
  March (monsoon 1) 0.253, matching the seasonal history and period B (21.4%).
  Exact zeros 0% (original 14.2%). Mean service 18.34 min (original 20.30).
- Correlation with the original: late probability Pearson 0.81 / Spearman 0.86;
  service Pearson 0.93 / Spearman 0.86.
- Largest late changes: Fresh on heavily disrupted days (disruption_index 46-59),
  original 0.01-0.11 -> new 0.93-0.99 (training late rate at index <= 60: 59%).
  Largest service changes: large Style/Tech orders on non-festival, non-payday days,
  -38 to -71 min (original max 180 min, new max 138).

**Reproducibility**
- Deleted the generated task1 CSVs, .joblib files and the submission, ran
  `task1/run_all.py` twice from scratch: `outputs/submission_task1.csv`,
  `reports/metrics.json` and the comparison report are byte-identical, and
  `DataFrame.equals` passes.

**Authorship**
- Agent wrote the Phase 9 code, `CHANGES_FOR_REVIEW.md` and this entry. User set the
  final configuration, the clip, the checks and the contents of the change report.
- Phase 9 committed as `0f34b85` and pushed.

---

## Phase 10 (reduced scope): handover package (2026-10-07)

**Scope (user decision):** the final notebook, full documents and zip are built by the
teammates; this phase produces a handover package in `docs/handover/` instead.

**Done**
- `task1/CHANGES_FOR_REVIEW.md`: "this commit" replaced by the Phase 9 hash `0f34b85`.
- `docs/handover/`: README.md (index + results summary), HOW_TO_RUN.md,
  NOTEBOOK_SNIPPETS.md (final inference cell), inference_cell.py (the same code),
  smoke_test_output.txt, PREPROCESSING_SECTIONS.md, ARCHITECTURE.md (Mermaid +
  deployment), AI_DISCLOSURE_NOTES.md, VIDEO_TALKING_POINTS.md.
- Smoke test: `inference_cell.py` (+ comparison) loads only the saved models and
  reproduces `outputs/submission_task1.csv` and `outputs/submission_task2a.csv`
  exactly (`DataFrame.equals` True for both, reading the CSVs with round-trip float
  parsing; the default parser differs by 1e-14).
- Note for the notebook: task1/ and task2a/ both have a `models.py`; the cell loads
  Task 1's feature list by file path and puts only task2a/ on sys.path.

**Authorship**
- Agent wrote the handover documents and the inference cell; user defined the
  package contents and scope.

---

## 2B Phase 0: Task 2B setup and checker smoke test (2026-10-07, branch `ananth`)

**Done**
- Branch `ananth` created from `main` (`2dc0e44`).
- Datasets restored locally from history (`git archive eaf7297 ... | tar -x`), not
  staged; a second copy of `General Data/` and `Test Data/` under `data/`, because
  `check_allocation.py` (organisers' script, unchanged) only searches `data/`.
  Documented in `data/README.md`. `git status` stays clean (both copies gitignored).
- `.venv` from `requirements.txt` (pandas 3.0.3, numpy 2.4.6, scikit-learn 1.9.0;
  Python 3.11.0 on this machine). Task 2B adds no dependency.
- `task2b/common.py`: paths, loaders (orders, fleet joined to vehicles.csv, district
  travel, service allowance), budgets (270 Fresh / 480 daytime / 2 trips) and
  `trip_time()` with the checker's formula.
- `task2b/smoke_check.py` -> `task2b/reports/phase0_smoke_output.txt`;
  `task2b/run_all.py`; `.gitignore` adds `task2b/data/`.
- `docs/TASK2B_PLAN.md`: data facts, agreed priority policy, phases 0-7.

**Checks (all pass)**
- 85 orders, unique `order_ref`, template rows identical and in the same order,
  all S1 and Peliyagoda, no missing sizes, every district and (brand, dock_type) has
  a reference row; 38 fleet vehicles join vehicles.csv, 28 available / 10 in workshop,
  all Peliyagoda.
- `trip_time()` reproduces both booklet examples (Gampaha 101 min, Colombo 112 min,
  213 of 270) and equals `check_allocation.trip_time()` for every district x brand.
- All-deferred submission: checker PASSED. Chilled S1-007 on ambient VEH008: checker
  FAILED with "non-refrigerated", so the checker really reads our data.

**Notes**
- pandas 3 uses a strict string dtype: write `trip_id` into string columns as "1"/"2".

**Authorship**
- Agent wrote the Phase 0 code, plan and this entry; user chose the priority policy
  (lexicographic, skip-twice first) and optimiser-as-submission.

## 2B Phase 1: capacity analysis (2026-10-07, branch ananth)

- Reviewed Phase 0 against booklet pp.18-21 and reran all smoke checks successfully; official checker unchanged.
- Added analysis.py: demand/capacity by class, district time/stop bounds, per-vehicle chilled packing/time relaxation, and individual feasibility.
- 181.629 m3 chilled demand; all four reefers have 172.400 m3 theoretical two-trip capacity. Relaxed bound 169.086 m3 => at least 12.543 m3 chilled shortage. Bounds allow order reuse and are explicitly not feasible plans.
- Found additional unavoidable deferral: Style S1-078, 40.660 m3 > largest available 38 m3. Corrected earlier ambient-all-fit and truck-only shortage claims.
- Checks: analysis assertions and Phase 0 passed; no new dependency, no dataset files staged.
- Authorship: Codex wrote analysis and documentation; user authorised phase commits and requested independent agent reviews.
