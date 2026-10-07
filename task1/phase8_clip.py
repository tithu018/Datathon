"""Task 1 Phase 8 addendum: metrics of the chosen lateness model (HGB + sigmoid,
CalibratedClassifierCV cv=5 stratified + shuffled) on test periods A and B with the
final clip [0.001, 0.999] (fixed choice, not tuned), next to raw and [0.01, 0.99].
Writes task1/reports/metrics_phase8_clip.json."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from evaluation import SEED
from features import FEATURES
from models import make_hgb_classifier

df = pd.read_csv(Path("task1") / "train_features.csv")
od = pd.to_datetime(df["order_date"])
PERIODS = {"A (2026-01-03..02-14)": (od < "2026-01-03", od >= "2026-01-03"),
           "B (2025-02-16..03-28)": (od < "2025-02-16", (od >= "2025-02-16") & (od <= "2025-03-28"))}
X, y = df[FEATURES], df["late"]
out = {}
for name, (tr, te) in PERIODS.items():
    m = CalibratedClassifierCV(make_hgb_classifier(FEATURES, learning_rate=0.05, max_iter=600), method="sigmoid",
                               cv=StratifiedKFold(5, shuffle=True, random_state=SEED)).fit(X[tr], y[tr])
    p = m.predict_proba(X[te])[:, 1]
    out[name] = {}
    for label, (lo, hi) in {"raw": (0, 1), "clip_0.01": (0.01, 0.99), "clip_0.001": (0.001, 0.999)}.items():
        q = np.clip(p, lo, hi)
        out[name][label] = {"log_loss": round(log_loss(y[te], np.clip(q, 1e-15, 1 - 1e-15)), 4),
                            "brier": round(brier_score_loss(y[te], q), 4), "auc": round(roc_auc_score(y[te], q), 4),
                            "mean_pred": round(float(q.mean()), 4), "actual": round(float(y[te].mean()), 4)}
t = pd.DataFrame({(n, c): v for n, d in out.items() for c, v in d.items()}).T
print(t.to_string())
for c in ("raw", "clip_0.01", "clip_0.001"):
    print(c, "mean log loss", round(np.mean([out[n][c]["log_loss"] for n in out]), 4),
          "mean Brier", round(np.mean([out[n][c]["brier"] for n in out]), 4))
(Path("task1") / "reports" / "metrics_phase8_clip.json").write_text(json.dumps(out, indent=2))
