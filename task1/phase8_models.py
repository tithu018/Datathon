"""Task 1 Phase 8: models and calibration on the final Phase 7 feature set.

Two time-based test periods; selection uses the MEAN over both:
  A  order_date >= 2026-01-03 (train on everything before)
  B  2025-02-16..2025-03-28, season-matched to the real test incl. monsoon March
     (train on everything before 2025-02-16)
Service time: Random Forest (current) vs HistGradientBoostingRegressor
(squared_error, absolute_error), small grid.
Lateness: RF balanced (current), RF without class weights, HGB raw, HGB + isotonic,
HGB + sigmoid. Calibration: CalibratedClassifierCV(cv=5) on the training portion;
the folds are stratified and shuffled (seed 42), so every fold spans all seasons.
Probabilities are reported raw and clipped to [CLIP_LO, CLIP_HI].

Run from the repo root after task1/prepare_features.py. Writes
task1/reports/phase8_service.csv, phase8_late.csv, phase8_reliability.csv,
calibration.png and metrics_phase8.json.
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from evaluation import SEED
from features import FEATURES, assert_no_leakage
from models import make_hgb_classifier, make_hgb_regressor, make_rf_classifier, make_rf_models

REPORTS = Path("task1") / "reports"
CLIP_LO, CLIP_HI = 0.01, 0.99
BINS = np.linspace(0, 1, 11)
SERVICE_GRID = [dict(learning_rate=lr, max_iter=it) for lr in (0.05, 0.1) for it in (300, 600)]
LATE_GRID = [dict(learning_rate=lr, max_iter=it) for lr in (0.05, 0.1) for it in (300, 600)]

t0 = time.time()
assert_no_leakage(FEATURES)
df = pd.read_csv(Path("task1") / "train_features.csv")
od = pd.to_datetime(df["order_date"])
PERIODS = {
    "A (2026-01-03..02-14)": (od < "2026-01-03", od >= "2026-01-03"),
    "B (2025-02-16..03-28)": (od < "2025-02-16", (od >= "2025-02-16") & (od <= "2025-03-28")),
}
X, ys, yl = df[FEATURES], df["service_minutes"], df["late"]
for name, (tr, te) in PERIODS.items():
    print(f"{name}: train {int(tr.sum()):,} rows, test {int(te.sum()):,} rows, test late rate {yl[te].mean():.4f}, "
          f"monsoon share {df.loc[te, 'monsoon'].mean():.2f}")


def log(msg):
    print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)


# ---------------------------------------------------------------- service time
service_rows = []
for name, (tr, te) in PERIODS.items():
    cands = {"RF (current)": make_rf_models(FEATURES)[0]}
    for loss in ("squared_error", "absolute_error"):
        for g in SERVICE_GRID:
            cands[f"HGB {loss} lr={g['learning_rate']} it={g['max_iter']}"] = make_hgb_regressor(FEATURES, loss, **g)
    for cname, model in cands.items():
        model.fit(X[tr], ys[tr])
        p = model.predict(X[te])
        service_rows.append({"period": name, "model": cname, "mae": mean_absolute_error(ys[te], p),
                             "rmse": float(np.sqrt(mean_squared_error(ys[te], p))),
                             "bias": float(np.mean(p - ys[te]))})
    log(f"service models done for {name}")
service = pd.DataFrame(service_rows)
service.to_csv(REPORTS / "phase8_service.csv", index=False)
svc = service.pivot_table(index="model", columns="period", values=["mae", "rmse"])
svc[("mae", "mean")] = svc["mae"].mean(axis=1)
svc[("rmse", "mean")] = svc["rmse"].mean(axis=1)
svc = svc.sort_values(("mae", "mean"))
pd.set_option("display.width", 250)
print("\n=== Service time (minutes), mean over A and B ===")
print(svc.round(3).to_string())

# ---------------------------------------------------------------- lateness
def late_metrics(y, p):
    return {"auc": roc_auc_score(y, p), "log_loss": log_loss(y, np.clip(p, 1e-15, 1 - 1e-15)),
            "brier": brier_score_loss(y, p), "mean_pred": float(np.mean(p)), "actual_rate": float(np.mean(y)),
            "share_p0": float(np.mean(p == 0)), "share_p1": float(np.mean(p == 1))}


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
late_rows, rel_rows, probs = [], [], {}
# choose the HGB classifier settings by raw log loss (mean over A and B), then calibrate that one
raw_ll = {}
for g in LATE_GRID:
    lls = []
    for name, (tr, te) in PERIODS.items():
        m = make_hgb_classifier(FEATURES, **g).fit(X[tr], yl[tr])
        lls.append(log_loss(yl[te], m.predict_proba(X[te])[:, 1]))
    raw_ll[f"lr={g['learning_rate']} it={g['max_iter']}"] = (np.mean(lls), g)
log("HGB classifier grid: " + ", ".join(f"{k} {v[0]:.4f}" for k, v in raw_ll.items()))
best_key = min(raw_ll, key=lambda k: raw_ll[k][0])
HGB_LATE = raw_ll[best_key][1]

for name, (tr, te) in PERIODS.items():
    cands = {
        "RF balanced (current)": make_rf_classifier(FEATURES, "balanced"),
        "RF no class weight": make_rf_classifier(FEATURES, None),
        f"HGB raw ({best_key})": make_hgb_classifier(FEATURES, **HGB_LATE),
        "HGB + isotonic (cv=5)": CalibratedClassifierCV(make_hgb_classifier(FEATURES, **HGB_LATE), method="isotonic", cv=cv),
        "HGB + sigmoid (cv=5)": CalibratedClassifierCV(make_hgb_classifier(FEATURES, **HGB_LATE), method="sigmoid", cv=cv),
    }
    for cname, model in cands.items():
        model.fit(X[tr], yl[tr])
        p = model.predict_proba(X[te])[:, 1]
        probs[(name, cname)] = p
        for clip in (False, True):
            q = np.clip(p, CLIP_LO, CLIP_HI) if clip else p
            late_rows.append({"period": name, "model": cname, "clipped": clip, **late_metrics(yl[te], q)})
        b = pd.cut(p, BINS, include_lowest=True)
        r = pd.DataFrame({"bin": b, "p": p, "y": yl[te].to_numpy()}).groupby("bin", observed=False).agg(
            n=("y", "size"), mean_pred=("p", "mean"), actual=("y", "mean")).reset_index()
        r[["period", "model"]] = name, cname
        rel_rows.append(r)
        log(f"{name} | {cname}")
late = pd.DataFrame(late_rows)
late.to_csv(REPORTS / "phase8_late.csv", index=False)
rel = pd.concat(rel_rows, ignore_index=True)
rel["bin"] = rel["bin"].astype(str)
rel.to_csv(REPORTS / "phase8_reliability.csv", index=False)

for clip in (False, True):
    t = late[late["clipped"] == clip].pivot_table(index="model", columns="period",
                                                  values=["log_loss", "brier", "auc", "mean_pred"])
    for m in ("log_loss", "brier", "auc"):
        t[(m, "mean")] = t[m].mean(axis=1)
    t = t.sort_values(("log_loss", "mean"))
    print(f"\n=== Lateness, {'clipped to [0.01, 0.99]' if clip else 'raw'} (actual late rate: "
          + ", ".join(f"{p} {yl[te].mean():.4f}" for p, (_, te) in PERIODS.items()) + ") ===")
    print(t[["log_loss", "brier", "auc", "mean_pred"]].round(4).to_string())
    if not clip:
        print("share p = 0:", late[~late["clipped"]].groupby("model")["share_p0"].mean().round(4).to_dict())

print("\n=== Reliability (10 bins; mean predicted / actual; n) ===")
for name in PERIODS:
    for cname in late["model"].unique():
        r = rel[(rel["period"] == name) & (rel["model"] == cname) & (rel["n"] > 0)]
        print(f"{name} | {cname}: " + "; ".join(f"{b} {mp:.3f}/{a:.3f} (n={n})" for b, mp, a, n in
                                                   zip(r["bin"], r["mean_pred"], r["actual"], r["n"])))

# ---------------------------------------------------------------- calibration plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, name in zip(axes, PERIODS):
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
    for cname in late["model"].unique():
        r = rel[(rel["period"] == name) & (rel["model"] == cname) & (rel["n"] >= 30)]
        ax.plot(r["mean_pred"], r["actual"], "o-", ms=4, label=cname)
    ax.set_title(f"Reliability, test {name}")
    ax.set_xlabel("mean predicted late probability (bin)")
    ax.set_ylabel("actual late rate")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(REPORTS / "calibration.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------- summary
best_service = svc.index[0]
lt = late[late["clipped"]].groupby("model")[["log_loss", "brier", "auc"]].mean().sort_values("log_loss")
(REPORTS / "metrics_phase8.json").write_text(json.dumps({
    "features": FEATURES,
    "periods": {k: {"n_train": int(tr.sum()), "n_test": int(te.sum()), "test_late_rate": round(float(yl[te].mean()), 4)}
                for k, (tr, te) in PERIODS.items()},
    "calibration_cv": "StratifiedKFold(5, shuffle=True, random_state=42) on the training portion",
    "clip": [CLIP_LO, CLIP_HI],
    "service": service.to_dict("records"),
    "service_best_by_mean_mae": best_service,
    "hgb_late_grid_mean_raw_logloss": {k: v[0] for k, v in raw_ll.items()},
    "hgb_late_params": HGB_LATE,
    "late": late.to_dict("records"),
    "late_ranking_clipped_mean": lt.round(5).to_dict("index"),
}, indent=2, default=str))
print(f"\nBest service model (mean MAE): {best_service}")
print("Late models by mean clipped log loss:\n" + lt.round(4).to_string())
print(f"Saved: task1/reports/phase8_service.csv, phase8_late.csv, phase8_reliability.csv, calibration.png, "
      f"metrics_phase8.json ({time.time() - t0:.0f}s)")
