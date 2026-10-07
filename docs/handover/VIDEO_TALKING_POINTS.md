# Video talking points (Task 2A and the Task 1 changes)

## Task 2B extension

- Show `outputs/submission_task2b.csv`: 75 served orders, all ten protected orders,
  58 outlets covered; nine chilled deferrals retain ambient delivery.
- Explain two bottlenecks: limited reefer volume/trips/morning time, and one
  indivisible 40.660 m³ Style order versus a maximum available 38 m³ vehicle.
- Show the exact optimiser versus greedy: 132.836 versus 125.859 m³ chilled.
- Explain the measured fairness choice: volume-first reaches 143.772 m³ but skips
  the protected Puttalam chilled order; protecting it costs 10.936 m³ overall.
- Show the repair opportunity: one 33.4 m³ workshop reefer increases delivery by
  35.996 m³ to 168.832 m³. Mention independent optimality verification.
- State limits: exact chilled search with fixed ambient packing; window/fuel
  diagnostics are approximations, with arrival lateness separate from unloading.

1. **Task 2A labels follow the brief exactly.**
   - **Labels:** every order counts (including deferred and never-dispatched ones), assigned to the week the store requested it. That gives 97,321 orders across 6 depot × brand series of 117 weeks.
   - **Show:** `task2a/reports/weekly_series.png`.
2. **The calendar drives demand.**
   - **Fresh:** +20% on Saturdays against Wednesdays, and +12% on payday and the two days after.
   - **Festivals:** New Year build-up lifts Fresh by up to about 70%.
   - **Style:** peaks in fixed weeks (W10, W32, W51), and each Style outlet orders once a week on a fixed day, so a closed day means a skipped order.
   - **Show:** `task2a/reports/festival_aligned_fresh.png`, `multipliers.png`.
3. **Task 2A model and its accuracy.**
   - **Model:** a GLM + boosting blend for Fresh, an outlet model for Style, a capacity-shaped level for Tech, and a chilled share.
   - **Accuracy:** 3.14% weekly error across four rolling backtests, against 8.7% for a last-8-weeks average. 3.47% on a confirmation period that was evaluated once.
   - **Show:** the improvement table in `task2a/reports/phase4_improvement.csv`, and `backtest_headline.png`.
4. **The Task 2A forecast handles closures.**
   - **W15:** +40–60% (New Year build-up).
   - **W16:** −33% (two closed days).
   - **W18:** down for Peliyagoda Style and Tech (Vesak on a Friday).
   - **W22:** +25% for Fresh (two paydays plus Poson; checked in-sample).
   - **Show:** `task2a/reports/final_forecast.png`.
5. **Task 1: a silent bug, then features known at planning time.**
   - **Bug:** five features were dropped by a merge; they're restored, and checked by assert.
   - **Features:** feature groups were added only if they helped on a time-based test. The calendar cut service-time error by 12%; road disruption cut late log loss by 17%. Road disruption assumes dispatch-morning prediction.
   - **Show:** `task1/reports/phase7_feature_groups.csv`.
6. **Task 1: calibration for the season.**
   - **Problem:** lateness is about 27% in monsoon months against about 13% otherwise, and the real test includes monsoon March. So we validate on a season-matched period.
   - **Results:**
     - the calibrated boosting model has log loss 0.140 / 0.163 (against 0.232 / 0.314 originally) and AUC 0.972 / 0.976;
     - it predicts 0.13 for February and 0.25 for March, matching history;
     - no exact-zero probabilities, against 14% originally.
   - **Show:** `task1/reports/calibration.png`, and the February vs March split in `task1/reports/submission_comparison.txt`.
7. **Engineering.**
   - **Reproducible:** one command per task (`run_all.py`) gives byte-identical submissions from scratch.
   - **Controlled:** the agent worked under a phase-by-phase approval log (`WORKLOG.md`), and the confirmation period was evaluated only once.
