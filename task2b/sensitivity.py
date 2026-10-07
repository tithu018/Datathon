"""Phase 5: one repaired workshop reefer, exact solver under changed availability."""
import json
import sys

sys.path.insert(0, ".")
from task2b.common import DATA_DIR, REPORT_DIR, load_orders, load_fleet
from task2b.allocate_greedy import allocate_ambient
from task2b.optimize import solve
from task2b.feasibility import make_submission, validate
from task2b.policy import score


def main():
    orders, fleet = load_orders(), load_fleet()
    base = json.loads((REPORT_DIR / "optimization_summary.json").read_text(encoding="utf-8"))
    workshop = fleet[~fleet.available & fleet.is_reefer & ~fleet.is_van]
    # Restore the largest workshop reefer truck, with identifier breaking ties.
    vehicle = workshop.sort_values(["volume_cap_m3", "weight_cap_kg", "vehicle_id"],
                                   ascending=[False, False, True]).iloc[0]
    changed = fleet.copy()
    changed.loc[changed.vehicle_id.eq(vehicle.vehicle_id), ["status", "available"]] = ["available", True]
    allocation, certificate = solve(orders, changed)
    sub = make_submission(orders, allocate_ambient(orders, changed) | allocation)
    minutes = validate(sub, orders, changed)
    full_score = score(orders.loc[sub.decision.eq("served")].itertuples(), minutes)
    certificate.update({"restored_vehicle": vehicle.vehicle_id,
                        "full_score": list(full_score),
                        "additional_chilled_m3": (full_score[1] - base["optimized"]["full_score"][1]) / 1000,
                        "validation": "Local published-rule validation with availability changed; official base checker must not validate this hypothetical fleet."})
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sub.to_csv(DATA_DIR / "sensitivity_submission.csv", index=False)
    (REPORT_DIR / "sensitivity_summary.json").write_text(json.dumps(certificate, indent=2) + "\n", encoding="utf-8")
    print(f"Restore {vehicle.vehicle_id}: {full_score[1]/1000:.3f} m3 chilled, +{certificate['additional_chilled_m3']:.3f} m3; exact changed-fleet checks PASSED")


if __name__ == "__main__":
    main()
