# Architecture: Task 2A and Task 1 pipelines

## Task 2B: capacity and allocation

```mermaid
flowchart LR
  S[Peak-day orders and template] --> V[Strict identifiers and reference checks]
  F[Fleet and vehicle capacities] --> V
  R[Published travel and service allowances] --> V
  V --> C[Capacity analysis and protected-order policy]
  C --> G[Greedy baseline]
  C --> E[Enumerate all chilled trips and vehicle schedules]
  E --> X[Exact branch-and-bound with fixed ambient allocation]
  X --> A[Window diagnostics and repaired-reefer sensitivity]
  X --> P[Allocation CSV and concise policy]
  G --> B[Baseline comparison]
  X --> B
  P --> K[Unchanged official feasibility checker]
  X --> T[Independent exhaustive verification]
  A --> H[Local timelines, fuel and order explanations]
```

This is a local optimisation pipeline with no trained model or external solver
dependency. The exact certificate covers chilled assignments under the chosen
policy, with ambient packing fixed. Windows and one-day fuel are approximate
diagnostics; they do not certify physical return/reload execution or weekly fuel.

## Task 2A: weekly depot demand forecast

```mermaid
flowchart LR
  subgraph Inputs
    D[deliveries_train.csv]
    T1[task1_test_inputs.csv]
    C[calendar.csv]
    O[outlets.csv]
  end
  D & T1 --> L["prepare_labels.py<br/>all orders incl. deferred / not_run<br/>week = ISO week of order_date"]
  C --> L
  L --> G["daily grid: date x depot x brand<br/>weekly labels: total + chilled"]
  C & O --> F["prepare_features.py<br/>weekday, payday d0-d2, festival ramp x name,<br/>closures, month, trend, Style peak weeks"]
  G --> F
  D & T1 --> SO["Style outlet x weekday table<br/>one order per open outlet-day"]
  F --> M1["Fresh: 0.6 x Poisson GLM per series<br/>+ 0.4 x global HistGradientBoosting"]
  SO --> M2["Style: outlet x weekday Poisson GLM"]
  F --> M3["Tech: 52-week level x weekday capacity"]
  F --> M4["Chilled: depot x month share x Fresh total"]
  M1 & M2 & M3 & M4 --> R["sum days to ISO weeks<br/>0 on closed days, 0 <= chilled <= total"]
  R --> S[outputs/submission_task2a.csv]
  F -. selection .-> B["backtest.py / experiments.py<br/>4 rolling 10-week origins + 1 confirmation"]
```

## Task 1: service time and lateness (updated)

```mermaid
flowchart LR
  subgraph Inputs
    TR[deliveries_train.csv + route_legs_train.csv]
    TE[task1_test_inputs.csv + route_legs_test.csv]
    REF[outlets, calendar, traffic_speed,<br/>road_conditions]
  end
  TR --> LB["prepare_labels.py<br/>service = leave - max(arrival, window open)<br/>late = arrival > window close"]
  LB --> FE["prepare_features.py + feature_groups.py<br/>30 planned-information features<br/>explicit list + leakage assert"]
  TE --> FE
  REF --> FE
  FE --> SV["Service: HistGradientBoostingRegressor"]
  FE --> LT["Lateness: HistGradientBoostingClassifier<br/>+ sigmoid calibration (5 shuffled folds)<br/>clip 0.001-0.999"]
  SV & LT --> SUB[outputs/submission_task1.csv]
  FE -. validation .-> V["time-based tests:<br/>A 2026-01-03..02-14, B 2025-02-16..03-28"]
```

## Proposed deployment

- **Task 2A (capacity planning): a weekly batch.**
  - **When:** every Sunday night, after the week's orders close.
  - **What:** refresh labels from the order system, then refit the configuration on all history. This takes seconds; it's GLMs plus one small boosting model.
  - **Output:** a 10-week depot × brand forecast, with chilled volume, published to the dispatcher's capacity screen.
  - **Monitoring:** compare each week's actuals with the forecast (WAPE and bias per series). Re-run the backtest monthly.
  - **Calendar:** has to be maintained ahead of time, because festival dates and closures drive the largest swings.
- **Task 1 (daily delivery planning): dispatch-morning scoring.**
  - **When:** after the plan is built and before vehicles leave. That's when road advisories (`disruption_index`) are known.
  - **What:** score every planned stop for expected service time and late probability. The dispatcher sees high-risk stops before departure.
  - **Retraining:** monthly, from the completed route records, keeping the season-matched validation and the calibration check (mean predicted vs actual late rate by month).
- **Serving:** both models are scikit-learn pipelines saved with joblib, loaded by a small batch job (`predict_task2a`, and the Task 1 inference cell). Versions are pinned in `requirements.txt`.
