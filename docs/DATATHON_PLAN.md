# Datathon implementation plan: Task 2A and Task 1 fixes

This is the single plan for the coding agent. It replaces `TASK2A_PLAN.md`.
Deadline: **Friday, October 9, 2026, 11:59 PM (Asia/Colombo)**. Target: code finished by the evening of October 8.

---

## A. Rules for the agent (read first, follow always)

### Approval protocol

1. **One phase at a time.** Only start a phase when I write `Start Phase N`.
2. **Before coding, say what you'll do.** Restate the phase goal and list the files you plan to create or change.
3. **When a phase is done, stop and report** using the template in section B. Leave the changes **uncommitted**.
4. **Commit only when I write `APPROVED Phase N`.**
   - Commit message format: `[Phase N] <summary>`.
   - Use the prefix `task1:` or `task2a:` in the summary.
   - After committing, stop. Do not start the next phase.
5. **If I ask for changes,** make them, re-run the relevant checks, report again, and wait.
6. **If a decision isn't covered by this plan, ask me.** Don't guess.
   Examples: whether to use a feature with unclear availability, or whether to drop a model.

### Never do these

- Push, merge, rebase or force-push, or touch `main`. I will push manually.
- Commit datasets, `.joblib` model files or other large generated files.
- Use pre-trained models, AutoML, low-code tools, or API-based modelling. These are against the competition rules.
- Use Task 1 **actual** times as model features. That includes:
  - `actual_depart_time`
  - `actual_travel_duration_min`
  - `arrival_time`
  - `leave_outlet_time`
  
  They exist only in training data and are used for labels only.
- Delete or overwrite the teammate's work without a record. Task 1 changes happen on this branch, so git keeps the original.

### Always do these

- Run scripts from `Datathon/`, using relative, Windows-safe paths (`pathlib`).
- Set random seeds everywhere, and pin new dependencies in the requirements file.
- Append a short entry to `Datathon/WORKLOG.md` at the end of every phase. Include:
  - what was done
  - the key numbers
  - the decisions made
  - which parts the agent wrote and which I changed
  
  This log feeds the AI tool disclosure and the preprocessing document.

---

## B. End-of-phase report template

```
## Phase N report: <title>
Files created/changed: ...
Commands run: ...
Results: <key tables/metrics, short>
Checks: <assertions run, pass/fail>
Plots saved: <paths>
Decisions I need from you: ...
Suggested commit message: [Phase N] ...
Status: WAITING FOR APPROVAL (nothing committed)
```

---

## C. Context

### Repo

Task 1 is a 4-script pipeline in `Datathon/task1/`:
- `prepare_labels.py`
- `prepare_features.py`
- `train_model.py`
- `make_submission.py`

It was committed as `eaf7297`, and its output is `Datathon/outputs/submission_task1.csv` (5,014 rows). Model `.joblib` files are gitignored. There is no final notebook yet.

### Task 2A

Forecast weekly order volume for each depot × brand × ISO week:
- **Depots:** Kandy and Peliyagoda.
- **Brands:** Fresh, Style and Tech.
- **Weeks:** ISO weeks 14–23 of 2026, which is 60 rows.

Output: `submission_task2a.csv` with the columns `row_id, pred_total_volume_m3, pred_chilled_volume_m3`. Chilled must be 0 for Style and Tech. Keep the template's `row_id`s and row order unchanged.

**History:**
- `deliveries_train.csv`: 2024-01-01 to 2026-02-14.
- `task1_test_inputs.csv`: 2026-02-16 to 2026-03-28.

These are combined as training history. `calendar.csv` runs to 2026-06-28.

**Findings from initial analysis:**
- **Daily Fresh volume follows multipliers:**
  - weekday: Saturday is about 17% above Wednesday
  - payday: about 12% higher
  - `festival_ramp`: up to about 30% higher
  - non-operating days: 0
- **Style** has a fixed weekday schedule: Thursday peaks and Tuesday is 0. **Tech** is small and lumpy.
- **Fresh grows** about 4–5% a year.
- **Chilled share** is about 0.38 from April to June and about 0.355 from October to February.
- **Festivals in the forecast window:**
  - Week 15: a full New Year build-up.
  - Week 16: only 4 operating days (13–14 April closed).
  - Week 18: Vesak on Friday 1 May, which is closed.
  - Week 22: Poson on Saturday 30 May (operating), plus a payday.
  
  In 2025, New Year also fell on a Monday. Peliyagoda Fresh went from 1,351 m³ in the week before to 623 m³ in the festival week.
- **A prototype already works well.** A daily Poisson GLM, summed to weeks, had an error (WAPE) of **5.9%**. The last-8-weeks mean had 15.0%. Both were measured on 2025 weeks 14–23.

### Task 1 issues to fix

1. Brand, district, depot and vehicle features are silently dropped. The merge renames them with `_order`/`_leg` suffixes, so they no longer match the feature list.
2. Outlet, service allowance, calendar, traffic and road-condition data aren't used.
3. Validation is a random 80/20 split, but the test orders come from a later period. The scores aren't saved.
4. The late-probability Random Forest (`class_weight="balanced"`) produces uncalibrated probabilities, including exact 0.0 values.
5. There's no final notebook.

---

## D. Phases

### Phase 0: Setup and baseline (no modelling changes)

1. Make sure the working tree is clean and `main` is up to date. Create the branch `datathon-task2a-task1-fixes` from it.
2. Commit this plan to `Datathon/docs/DATATHON_PLAN.md`. This is the only commit allowed before approval, and it's covered by the Phase 0 approval.
3. Check `.gitignore`: the raw CSVs and `.joblib` files must be excluded. List anything tracked that shouldn't be.
4. Check the environment:
   - the Python version
   - the installed packages against the requirements file
   - lightgbm is **not used**: pip cannot verify certificates on this network and bypassing verification is not allowed. Use scikit-learn's `HistGradientBoostingRegressor`/`HistGradientBoostingClassifier` instead.
5. **Task 1 baseline.** Run the original Task 1 pipeline unchanged and record:
   - its random-split scores
   - its scores on a **time-based test period**: the last 6 weeks of training orders (`order_date` ≥ 2026-01-03), trained on everything before
   
   Metrics:
   - service time: MAE and RMSE
   - lateness: ROC-AUC, log loss, Brier score, and the share of predictions that are exactly 0 or 1
   
   Save these to `Datathon/task1/reports/baseline_metrics.json`. Every Task 1 change is compared against this baseline.
6. Copy the current `outputs/submission_task1.csv` to `Datathon/task1/reports/submission_task1_original.csv`, for comparison later. This file is small, so it's fine to commit.

### Phase 1: Task 2A labels (`task2a/prepare_labels.py`)

1. Load both order files and assert that no `delivery_id` appears in both.
2. Keep **every** order, including `deferred` and `not_run`.
3. Assign weeks by `order_date` (never `dispatch_date`), using `iso_year` and `iso_week` from `calendar.csv`. Assert that every order matches a calendar date.
4. Build a complete daily grid of date × depot × brand (zero where there were no orders), with `total_volume` and `chilled_volume`.
5. Build a weekly table: depot × brand × `iso_year` × `iso_week`.
6. Assert:
   - daily totals sum to weekly totals, which sum to the raw total
   - chilled volume exists only for Fresh
   - there are no orders on non-operating days
   - each series has 117 complete weeks
7. Print the dispatch-status breakdown and document the caveat: the Task 1 file has no `not_run` orders, so its 6 weeks may undercount by about 0.45%.
8. Save the outputs to `task2a/data/`, gitignored if large.

### Phase 2: Task 2A features and analysis (`task2a/prepare_features.py`)

1. **Daily features**, for history and for the forecast window up to 2026-06-07:
   - day of week
   - `is_operating`
   - `is_payday`
   - days to and from the nearest payday
   - `festival_ramp` and its square
   - festival identity (which festival the ramp leads up to)
   - `is_holiday`
   - `monsoon`
   - month
   - trend in years since 2024-01-01
   - outlet count per depot × brand from `outlets.csv`
2. Assert that the forecast-window features have no missing values and that the window covers exactly ISO weeks 14–23 of 2026.
3. **Plots**, saved to `task2a/reports/`:
   - the weekly series for each depot × brand, with festival weeks marked
   - the multipliers by weekday, payday and festival ramp
   - the chilled share over time

### Phase 3: Backtest framework, baselines and GLM (`task2a/backtest.py`)

1. **Rolling tests with a 10-week horizon.** For each test period, train on everything before it and forecast the next 10 weeks.
   - Headline test: 2025 weeks 14–23.
   - Others: 2026 weeks 4–13, plus 2 more periods during 2025.
   - Skip early 2024, because there isn't enough history.
2. **Metrics:** WAPE and MAE for total and chilled volume, overall and for each brand and series.
3. **Models:**
   - **Baselines:** the last-8-weeks mean, and the same week last year.
   - **Daily Poisson GLM per series:** fit on operating days only, force 0 on closed days, and sum to weeks. Report the GLM's weekday, payday and festival multipliers.
4. Save `task2a/reports/backtest_results.csv`, plus a plot of actual against predicted for the headline test.

### Phase 4: Task 2A model improvements and chilled share

Add each item below on its own and measure it against the same test periods. Keep it only if it helps on average and doesn't badly hurt the headline test.

1. **Global HistGradientBoosting** (`HistGradientBoostingRegressor`, scikit-learn) on daily data: `loss="poisson"`, depot and brand as categorical features (`categorical_features`), shallow and regularized.
2. **A GLM + HistGradientBoosting blend.** Choose the weights on the test periods.
3. **Style:** an outlet × weekday variant.
4. **Tech:** a blend of the model and a smoothed recent mean.
5. **Chilled volume:** a share model by month or season, multiplied by the predicted total. Compare it with forecasting chilled volume directly.
6. Produce an **improvement table**: one row per addition, showing how each changes the error against the baseline.
7. **Optional, if time allows:** a median and a 90th-percentile interval from bootstrapped test-period residuals.

End of phase: recommend a final configuration and wait for my decision.

### Phase 5: Task 2A final fit and submission

1. **`task2a/train_model.py`:** fit the chosen configuration on all history up to 2026-03-28, and save the `.joblib` files and the share table.
2. **`task2a/predict.py`:** write `predict_task2a(test_inputs_df) -> DataFrame`. It loads the saved models, builds the features and returns the predictions.
3. **`task2a/make_submission.py`:** fill the template and write `outputs/submission_task2a.csv`. Assert:
   - 60 rows
   - `row_id`s and order identical to the template
   - no missing values, no negatives
   - chilled is 0 for Style and Tech
   - chilled is never larger than total
4. **Plot** the forecast against history, and explain weeks 15, 16, 18 and 22.

### Phase 6: Task 1 fixes, part 1 (bug fix and time-based validation)

1. Fix the dropped brand, district, depot and vehicle columns in `task1/prepare_features.py`. Make the feature list explicit, and assert that every listed feature column exists, so this can't fail silently again.
2. Change `task1/train_model.py` to report **both** the random-split and the time-based test scores, using the Phase 0 setup. Save them to `task1/reports/metrics.json`.
3. Re-run the pipeline and show the before and after scores. Don't change the models or features in any other way yet.

### Phase 7: Task 1 fixes, part 2 (new features)

These features must be known at planning time. Add them in groups, and measure each group on the time-based test:

1. **Outlet features** from `outlets.csv`: `dock_type`, `parking_constraint`, `mall_window`.
2. **Service allowance:** `service_allowance_min` from `service_allowance.csv`, by brand × dock type.
3. **Planned slack:**
   - the window close time minus the planned arrival time
   - the planned arrival time minus the window open time
   - the planned dwell time, which is the next leg's planned departure minus this stop's planned arrival, where a next leg exists
4. **Route context:**
   - the position of the stop in the route (`seq`)
   - the number of stops in the route
   - the cumulative planned travel time up to this stop
   - **the cumulative service allowance of earlier stops minus the planned dwell time.** This estimates how much delay builds up before this stop.
5. **Calendar:** payday, `festival_ramp`, holiday.
6. **Traffic:** `speed_index` from `traffic_speed.csv`, by district × planned arrival hour × monsoon.
7. **Road conditions:** check whether `road_conditions.csv` covers the test dates. **Ask me before using it**, because whether a planner would know this in advance is debatable.

Report the score change for each group, and keep only the groups that help.

### Phase 8: Task 1 fixes, part 3 (models and calibration)

1. Compare the current Random Forest with `HistGradientBoosting` (regressor and classifier), for both targets, on the time-based test.
2. **Lateness:** try a model without balanced class weights, and calibrate it with `CalibratedClassifierCV` (isotonic or sigmoid). Report:
   - log loss
   - Brier score
   - ROC-AUC
   - a calibration plot
3. Clip the final late probabilities to [0.01, 0.99].
4. Recommend a final configuration and wait for my decision.

### Phase 9: Regenerate Task 1 and write the change report

1. Retrain the chosen Task 1 models on all training data, and regenerate `outputs/submission_task1.csv` with the template's row order and the existing checks.
2. Compare the new submission with `submission_task1_original.csv`: the distributions, the correlation, and the largest changes.
3. Write `Datathon/task1/CHANGES_FOR_REVIEW.md` for the teammate. Cover:
   - each issue that was found
   - each fix and the commit where it was made
   - the baseline and new metrics side by side
   - features added or rejected, with the reason for each
   - how to re-run the pipeline
   - any open questions

### Phase 10: Integration and documentation drafts

1. **Final notebook.** Create `TeamName_FinalNotebook.ipynb` (I'll give you the team name). It should contain:
   - label construction, preprocessing, training and evaluation for both tasks, importing from or summarizing the scripts and showing the saved metrics
   - a **last cell** that loads the saved models and runs inference for Task 1 and Task 2A, clearly printing sample inputs and predictions
2. Run the notebook from top to bottom.
3. **Draft documents** in `Datathon/docs/`:
   - The preprocessing document sections: data preparation, label construction, cleaning, feature engineering, and the reasoning for each.
   - Mermaid source for the architecture diagram. For both tasks it should show models, preprocessing and the proposed deployment approach.
   - An AI tool disclosure draft, built from `WORKLOG.md`.
4. **Pre-submission checklist:**
   - all three submission CSVs are present and pass their validations
   - the model files are present next to the notebook
   - `TeamName_Datathon.zip` has the right folder layout

---

## E. Suggested schedule

| When | Phases |
|---|---|
| Oct 7, evening | 0, 1, 2 |
| Oct 8 | 3, 4, 5, then 6, 7, 8 |
| Oct 9 | 9 and 10, then the video, the zip, and submitting well before 11:59 PM |

If time gets short, cut in this order:
1. the optional intervals in Phase 4
2. Phase 7's traffic and road-condition groups
3. extra HistGradientBoosting tuning in Phase 8

**Never cut** Phase 6, the label checks, or the final notebook.
