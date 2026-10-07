# Task 2B plan: peak-day fleet allocation (scenario S1)

Branch: `ananth`. Deadline: **Friday, October 9, 2026, 11:59 PM (Asia/Colombo)**.
Commits use `[2B Phase N] task2b: <summary>`; one phase per commit, reviewed before the next.

## The task

There is one dispatch day at Peliyagoda, a week before a festival, with more orders than the fleet can carry.
Mark each of the 85 orders `served` or `deferred`, and give each served order a `vehicle_id` and a `trip_id` (1 or 2).

Feasibility rules, which `check_allocation.py` verifies:
- **One brand and one district per trip.**
- **Refrigeration:** chilled orders must go on a reefer.
- **Access:** van_only outlets must get a van.
- **Home depot:** a vehicle only serves its own depot's outlets.
- **Whole orders:** an order is never split across trips.
- **Capacity:** each trip must stay within both the weight cap and the volume cap.
- **Trips per vehicle:** at most 2.
- **Time budgets per vehicle:** Fresh trips get 270 minutes in total, and Style and Tech trips get 480 minutes combined.

Trip time = depot-to-district + inter-stop × (stops − 1) + the sum of the service allowances.
The return leg is not counted.

## What the data shows

- **Orders (85, all Peliyagoda):**
  - Fresh: 75 orders. 26 are chilled, and 3 of those are van_only (Colombo).
  - Style: 5 orders.
  - Tech: 5 orders.
  - 26 outlets place two orders each, one chilled and one ambient.
  - 10 orders have `deferred_yesterday = 1`.
- **Fleet:** 28 vehicles are available and 10 are in the workshop. Five of the 10 in the workshop are reefers.

| Resource (available) | Capacity per trip | Demand | |
|---|---|---|---|
| Reefer trucks: VEH003, VEH006, VEH007 | 79.2 m³ and 15,960 kg; at most 6 trips = 158.4 m³ and 31,920 kg | Chilled orders at normal-access outlets: 175.6 m³, 31,685 kg, in 7 districts | **Bottleneck** |
| Reefer van: VEH036 | 7.0 m³, 1,040 kg | Chilled van_only orders: 6.0 m³, 1,096 kg | Needs 2 trips |
| Ambient trucks (22) | 656 m³ | Fresh ambient 131 m³; Style 77 m³; Tech 18 m³ | Ample |
| Ambient vans: VEH037, VEH038 | 17 m³ | Ambient van_only orders: 2.0 m³ | Ample |

Three limits make the reefer trucks the bottleneck:
- **Volume:** at least ~17 m³ of chilled goods can't be served even with perfect packing. Those deferrals are unavoidable.
- **Trip count:** 7 chilled districts compete for only 6 reefer-truck trips.
- **Time:** depot-to-district travel takes 173 min to Puttalam, 137 to Matara and 127 to Kurunegala. A reefer truck sent to one of these gets one Fresh trip in its 270-minute window.

## Priority policy (agreed)

Allocations are ranked by these goals in strict order. A plan wins on the first goal; later goals only break ties.

1. **Don't skip the same outlet twice.** Serve `deferred_yesterday = 1` orders, and orders with `days_since_last_served ≥ 3`, wherever feasible.
2. **Maximise chilled volume served.** Festival dairy, meat and produce are perishable.
3. **Maximise orders and outlets served.**
4. **Minimise vehicle-minutes.**

Two hard rules go with it:
- Never put an ambient order on a reefer while a chilled order is waiting. Reefer space is the scarce resource, and an extra order adds a stop, so there's no time saving.
- If an outlet's chilled order is deferred, still serve its ambient order.

**Submission:** the optimiser's plan. The greedy plan is reported as the baseline.

## Phases

| Phase | Goal | Main output |
|---|---|---|
| 0 Setup | Data in place, environment, `task2b/` skeleton, checker smoke test | `task2b/reports/phase0_smoke_output.txt` |
| 1 Capacity analysis | Demand vs capacity by vehicle class, weight, volume, trips and time; chilled upper bound; bottleneck named | `task2b/analysis.py`, `reports/capacity_analysis.txt` |
| 2 Priority policy | The policy above, as code (lexicographic order score) | `task2b/policy.py` |
| 3 Greedy allocator | Feasibility core (same formula as the checker); scarce-vehicle-first greedy; checker passes | `task2b/feasibility.py`, `allocate_greedy.py` |
| 4 Exact optimiser | Enumerate feasible reefer trips per district (≤ 2⁹ subsets) and search the vehicle combinations; proven best under the policy; gap vs greedy | `task2b/optimize.py` |
| 5 Checks beyond the checker | Window timeline from 03:30, mall windows, fuel km (informational), skip-twice audit, "+1 reefer" sensitivity | `reports/` |
| 6 Write-up | ~1-page policy: bottleneck, rules, worked example, unavoidable vs chosen deferrals and their cost; per-order reason file | `docs/handover/TASK2B_POLICY.md` |
| 7 Integration | `outputs/submission_task2b.csv`, notebook cells, WORKLOG, AI disclosure, video points | — |

## Setup notes

- **Datasets** aren't tracked (see `data/README.md`). Restore them locally from history without staging:

  ```
  git archive eaf7297 "General Data" "Training Data" "Test Data" "Submission Templates" | tar -x
  mkdir -p data && git archive eaf7297 "General Data" "Test Data" | tar -x -C data
  ```

  The `data/` copy is needed because `check_allocation.py` only searches `data/`.
- **Environment:**

  ```
  python -m venv .venv
  .venv/Scripts/python -m pip install -r requirements.txt
  ```

  Task 2B needs pandas only, with no solver or other new dependency.
- **Run:** `.venv/Scripts/python task2b/run_all.py` from the repo root.
