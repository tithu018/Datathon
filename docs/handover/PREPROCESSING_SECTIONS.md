# Preprocessing document: Task 2A and Task 1 sections (ready to paste)

## Task 2A: Forecast depot demand

### Data preparation and label construction
- **Orders:** we combined `deliveries_train.csv` (92,307 orders, 2024-01-01 to 2026-02-14) with `task1_test_inputs.csv` (5,014 orders, 2026-02-16 to 2026-03-28), giving 97,321 orders. No `delivery_id` appears in both.
- **Every order counts once**, including deferred (1,633) and never-dispatched (413) orders, because they still represent demand.
- **Weeks:** each order is assigned to the week the store requested it (`order_date`, never `dispatch_date`), using `iso_year`/`iso_week` from `calendar.csv`. Every order date is in the calendar, and none falls on a non-operating day.
- **Labels:** total `order_volume_m3`, and the chilled part of it, per depot × brand × ISO week. That gives 6 series of 117 complete weeks (2024-W01 to 2026-W13). Chilled volume exists only for Fresh.
- **Checks:** every count, total and weekly mean was checked against independently computed reference numbers. Daily, weekly and raw totals agree exactly.
- **Known limitation:** the Task 1 order file contains no never-dispatched orders, so the last 6 history weeks may understate demand by about 0.44%. That's an order of magnitude below the model error, so it's documented, not corrected.

### Daily grid
- We built a complete daily grid of every calendar date × depot × brand, with zero volume where there were no orders. It extends to the end of the last ISO week, so every week has all 7 days.
- Models are fitted on operating days only. Closed days (Sundays, festivals, public holidays) are forced to 0.

### Feature engineering
All features come from the calendar and outlet tables, so they're known for future weeks.

- **Weekday.** Fresh peaks on Saturday, about 20% above Wednesday.
- **Payday window.** Paydays fall on the 25th and the last day of the month, moved back to Saturday when they land on a Sunday. Fresh demand is about 12% higher on payday and on each of the two days after, so we use one flag per day.
- **Festival build-up.** The festival ramp and its square, interacted with the festival's name, because each festival has a different effect: New Year lifts Fresh by up to about 70%, while Vesak and Poson barely affect Style.
- **Closures.** Days until the next non-Sunday closure (demand is pulled forward), and a flag for the day after a single mid-week closure. The latter was dropped after an ablation showed no gain.
- **Other features:** month, a linear trend (Fresh grows about 5–6% a year), and a Style peak-week flag. Style spikes in fixed ISO weeks 10, 32 and 51, even when festival dates move.
- **Style outlet structure.** Each Style outlet orders exactly once a week, on a fixed weekday (checked for all 2,890 orders). When that day is closed, the order is skipped, not moved. So Style is modelled per outlet and summed for the weeks the outlet's day is open.
- **Tech capacity.** Tech outlets order on two fixed weekdays. The Tech forecast is a level per unit of weekday capacity (smoothed over 52 weeks), multiplied by each week's open-day capacity.

### Validation design
- **Rolling backtests:** four origins, each forecasting 10 weeks ahead and training only on earlier weeks: 2025-W14..23 (the headline test, the same calendar weeks as the forecast), 2026-W04..13, 2025-W30..39 and 2025-W40..49.
- **Metrics:** weekly WAPE and MAE, plus signed bias, for total and chilled volume, overall, per brand and per series. Every addition was kept only if it improved the mean over the four periods.
- **Confirmation period (2025-W02..W11):** held out from all selection and evaluated **once**, on the final configuration and the baselines.

| Configuration | Total WAPE |
|---|---|
| Final | 3.47% |
| First GLM | 3.48% |
| Baselines | 5.3–5.7% |

Fresh and chilled confirmed the selection-period gains; Style didn't. That period trains on only one year of history.

## Task 1: Service time and lateness (changes to the original pipeline)

### Label audit (no change needed)
- **Labels** are built per delivery from its route leg:
  - service time = leaving the outlet − max(arrival, window open), because a vehicle that arrives early waits;
  - late = arrival strictly after window close.
- **Checks:**
  - Times are parsed to minutes before any comparison.
  - Nothing crosses midnight.
  - An independent re-derivation reproduces every label.
- **Early arrivals:** 4.38%, with a median wait of 14 minutes, which is excluded from service time.
- **Late rate:** 20.4% Fresh, 5.5% Style, 2.5% Tech.

### Bug fix
- The order and route-leg files share six columns: brand, district, depot, vehicle type, vehicle temperature and planned arrival.
- The merge renamed them, so five model features were silently dropped.
- They're now asserted identical and restored. An explicit feature list is checked for existence at every step, and a leakage guard forbids every actual-time column.

### Feature groups and why
We added each group in turn with the same models, and kept a group only if it improved service MAE or late log loss on a time-based test by at least 0.5% without hurting the other by more than 1%.

| Group | Decision |
|---|---|
| Outlet attributes (dock type, parking constraint, mall window) | kept |
| Route context (stop position, stops per route, cumulative planned travel, planned route start) | kept |
| Calendar (payday, festival ramp, holiday) | kept: busy days slow receiving, so service time rises from 17 to 32 minutes late in a festival build-up |
| Typical traffic (speed index by district × planned hour × monsoon) | kept |
| Road conditions (district × date disruption index) | kept |
| Service allowance | rejected |
| Planned slack and dwell | rejected |

- **Finding:** the planned dwell equals the service allowance at every stop, so "allowance minus dwell" carries no information.
- **Road-conditions assumption:** predictions are made on the dispatch morning, before deliveries start, when road advisories are known. The organizers supplied the data covering the test dates. This group lowered late log loss by 17%.

### Calibration and seasonality
- **Lateness is strongly seasonal:** about 27% in monsoon months, against about 13% otherwise. The real test window (16 February to 28 March) includes monsoon March.
- **We therefore validate on two periods:**
  - A: 2026-01-03..02-14, non-monsoon, 13.1% late;
  - B: 2025-02-16..03-28, the same calendar window as the test, 21.4% late.

  Model choices were selected on the mean of the two.
- **The original balanced Random Forest over-predicted lateness** (mean 0.18 against an actual 0.13), and 14% of its outputs were exactly 0.
- **The final classifier** is HistGradientBoosting calibrated with a sigmoid, fitted with 5 stratified, shuffled cross-validation folds so every fold spans all seasons. Probabilities are clipped to [0.001, 0.999].

| Period | Mean predicted late rate | Actual |
|---|---|---|
| A | 0.131 | 0.131 |
| B | 0.205 | 0.214 |

Log loss fell from 0.232 / 0.314 to 0.140 / 0.163 (A / B). The final models are refitted on all training data.
