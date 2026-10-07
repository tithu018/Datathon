# Task 2B: peak-day prioritisation policy

**Result:** 75 of 85 orders served at 58 of 59 outlets; all 10 protected orders served.
Chilled delivered: **132.836 of 181.629 m³**. All 58 individually feasible ambient orders are served.

**What limits service.** Three available reefer trucks carry 79.2 m³ per trip; one reefer van carries 7 m³/1,040 kg.
Two trips each give a theoretical 172.4 m³ ceiling. Weight, whole-order packing, single-district trips and the 270-minute Fresh budget reduce the exact volume-first maximum to **143.772 m³**.
Separately, Style order S1-078 is 40.660 m³ against a largest available vehicle of 38 m³, forcing its deferral.

**Priority.** Rank allocations by protected orders (`deferred_yesterday=1` OR `days_since_last_served≥3`), chilled volume, order count, outlet count, then fewer published minutes.
Both flags count once; histories can differ between orders at one outlet. Use ambient vehicles for ambient orders while chilled demand waits; serve an ambient order even if its chilled pair is deferred. The van may serve normal-access outlets.

**Calculation.** A Gampaha Fresh trip with two rear docks and one street stop takes 37 + 2×9 + 15 + 15 + 16 = **101 minutes**.
A four-street Colombo trip takes 112; together they use 213 of 270 minutes. Return travel is excluded by the published rule.
Enumerating every feasible chilled trip and vehicle trip-pair proves the chilled optimum; the ambient allocation is fixed, so global minimum ambient minutes is not claimed.

| Deferred group | Orders | m³ omitted | Explanation |
|---|---:|---:|---|
| Fresh, Colombo | 3 | 6.013 | Joint reefer constraints and priority |
| Fresh, Kalutara | 2 | 10.957 | Joint reefer constraints and priority |
| Fresh, Kurunegala | 3 | 25.039 | Joint reefer constraints and priority |
| Fresh, Matara | 1 | 6.784 | Joint reefer constraints and priority |
| Style, Kurunegala | 1 | 40.660 | Cannot fit any available vehicle |

**Unavoidable versus chosen.** At least **37.857 m³** chilled must be omitted even when volume alone leads; specific chilled orders remain individually feasible.
Protecting all flagged orders, including Puttalam S1-083, adds **10.936 m³** to that loss: total chilled deferral **48.793 m³** across nine orders.
Volume-first delivers two more chilled orders but skips S1-083. The priority plan improves on greedy by **6.977 m³**. All nine deferred chilled orders retain their ambient delivery; S1-078 alone leaves an outlet uncovered.

**Checks and remedy.** The unchanged official checker passes. A no-return window simulation finds zero late arrivals; four unloads finish after their outlet close, all before 08:00. This diagnostic does not certify physical return/reload timing or weekly fuel compliance.
Repairing VEH004 gives **168.832 m³** chilled and 80 served orders, a **35.996 m³** gain.
Per-order explanations and timelines remain local in `task2b/data/`; the submission contains only the required six columns.
