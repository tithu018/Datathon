"""Phase 6: concise policy and local per-order decisions, derived from outputs."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
from task2b.common import DATA_DIR, REPORT_DIR, load_orders, load_fleet
from task2b.policy import protected
from task2b.feasibility import validate


def main():
    orders = load_orders()
    sub = pd.read_csv(DATA_DIR / "optimized_submission.csv")
    validate(sub, orders, load_fleet())
    summary = json.loads((REPORT_DIR / "optimization_summary.json").read_text(encoding="utf-8"))
    sensitivity = json.loads((REPORT_DIR / "sensitivity_summary.json").read_text(encoding="utf-8"))
    result = summary["optimized"]["full_score"]
    volume_best = summary["volume_first"]["full_score"][1] / 1000
    served_volume = result[1] / 1000
    demand = orders.loc[orders.temp_requirement.eq("chilled"), "order_volume_m3"].sum()
    merged = orders.copy()
    for col in ("decision", "vehicle_id", "trip_id"):
        merged[col] = sub[col]
    merged["protected"] = merged.apply(protected, axis=1)
    merged["classification"] = "served"
    merged["reason"] = "Selected by the protected-order, chilled-volume policy; published rules satisfied."
    impossible = merged.order_ref.eq("S1-078")
    merged.loc[impossible, "classification"] = "individually_unavoidable"
    merged.loc[impossible, "reason"] = "Whole Style order is 40.660 m3; largest available compatible vehicle is 38.000 m3. Splitting is forbidden."
    chilly_deferred = merged.decision.eq("deferred") & merged.temp_requirement.eq("chilled")
    merged.loc[chilly_deferred, "classification"] = "joint_capacity_and_policy"
    merged.loc[chilly_deferred, "reason"] = (
        "Individually feasible, but omitted by the exact protected-first chilled allocation under joint capacity/trip/time limits. "
        "Its ambient pair is served. Individual omitted rows are not claimed inherently unavoidable.")
    merged["deferred_volume_m3"] = merged.order_volume_m3.where(merged.decision.eq("deferred"), 0)
    merged["deferred_weight_kg"] = merged.order_weight_kg.where(merged.decision.eq("deferred"), 0)
    assert len(merged) == 85 and merged.order_ref.is_unique and merged.reason.notna().all()
    merged[["scenario", "order_ref", "outlet_id", "brand", "district", "decision", "vehicle_id", "trip_id",
            "protected", "classification", "deferred_volume_m3", "deferred_weight_kg", "reason"]].to_csv(
                DATA_DIR / "order_decisions.csv", index=False)
    table = []
    for (brand, district), group in merged[merged.decision.eq("deferred")].groupby(["brand", "district"]):
        table.append(f"| {brand}, {district} | {len(group)} | {group.order_volume_m3.sum():.3f} | "
                     + ("Cannot fit any available vehicle" if brand == "Style" else "Joint reefer constraints and priority") + " |")
    output = f"""# Task 2B: peak-day prioritisation policy

**Result:** {result[2]} of 85 orders served at {result[3]} of 59 outlets; all {result[0]} protected orders served.
Chilled delivered: **{served_volume:.3f} of {demand:.3f} m³**. All 58 individually feasible ambient orders are served.

**What limits service.** Three available reefer trucks carry 79.2 m³ per trip; one reefer van carries 7 m³/1,040 kg.
Two trips each give a theoretical 172.4 m³ ceiling. Weight, whole-order packing, single-district trips and the 270-minute Fresh budget reduce the exact volume-first maximum to **{volume_best:.3f} m³**.
Separately, Style order S1-078 is 40.660 m³ against a largest available vehicle of 38 m³, forcing its deferral.

**Priority.** Rank allocations by protected orders (`deferred_yesterday=1` OR `days_since_last_served≥3`), chilled volume, order count, outlet count, then fewer published minutes.
Both flags count once; histories can differ between orders at one outlet. Use ambient vehicles for ambient orders while chilled demand waits; serve an ambient order even if its chilled pair is deferred. The van may serve normal-access outlets.

**Calculation.** A Gampaha Fresh trip with two rear docks and one street stop takes 37 + 2×9 + 15 + 15 + 16 = **101 minutes**.
A four-street Colombo trip takes 112; together they use 213 of 270 minutes. Return travel is excluded by the published rule.
Enumerating every feasible chilled trip and vehicle trip-pair proves the chilled optimum; the ambient allocation is fixed, so global minimum ambient minutes is not claimed.

| Deferred group | Orders | m³ omitted | Explanation |
|---|---:|---:|---|
{chr(10).join(table)}

**Unavoidable versus chosen.** At least **{demand-volume_best:.3f} m³** chilled must be omitted even when volume alone leads; specific chilled orders remain individually feasible.
Protecting all flagged orders, including Puttalam S1-083, adds **{volume_best-served_volume:.3f} m³** to that loss: total chilled deferral **{demand-served_volume:.3f} m³** across nine orders.
Volume-first delivers two more chilled orders but skips S1-083. The priority plan improves on greedy by **{summary['chilled_volume_gain_m3']:.3f} m³**. All nine deferred chilled orders retain their ambient delivery; S1-078 alone leaves an outlet uncovered.

**Checks and remedy.** The unchanged official checker passes. A no-return window simulation finds zero late arrivals; four unloads finish after their outlet close, all before 08:00. This diagnostic does not certify physical return/reload timing or weekly fuel compliance.
Repairing {sensitivity['restored_vehicle']} gives **{sensitivity['full_score'][1]/1000:.3f} m³** chilled and {sensitivity['full_score'][2]} served orders, a **{sensitivity['additional_chilled_m3']:.3f} m³** gain.
Per-order explanations and timelines remain local in `task2b/data/`; the submission contains only the required six columns.
"""
    path = Path("docs/handover/TASK2B_POLICY.md")
    path.write_text(output, encoding="utf-8")
    assert len(output.split()) < 600, "Keep the policy approximately one page"
    print(f"Policy and 85 per-order reasons written; {len(output.split())} words; Phase 6 checks PASSED")


if __name__ == "__main__":
    main()
