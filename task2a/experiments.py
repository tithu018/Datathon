"""Task 2A Phase 4: model improvements, selection and the confirmation run.

Selection uses ONLY the 4 selection periods (mean WAPE; headline shown separately).
The confirmation period (2025-W02..W11) is evaluated by `--confirm`, once, on the
saved recommended configuration and the baselines; it refuses to run twice unless
--force is given.

Run from the repo root after task2a/prepare_features.py:
    python task2a/experiments.py            # selection -> reports/phase4_*.csv, phase4_config.json
    python task2a/experiments.py --confirm  # one-off confirmation of phase4_config.json
    python task2a/experiments.py --score-config  # selection-period metrics of the saved config only
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import (CONFIRMATION_PERIOD, HEADLINE, SELECTION_PERIODS, load_history,  # noqa: E402
                        make_context, score)
from forecast import Forecaster, spec_key  # noqa: E402
from models import BRANDS, SERIES, WEEK_KEYS, baseline_forecasts  # noqa: E402

REPORTS = Path("task2a") / "reports"
CONFIG_PATH = REPORTS / "phase4_config.json"
WEIGHTS = [round(float(w), 1) for w in np.arange(0, 1.01, 0.1)]
SERIES_NAMES = [f"{d} {b}" for d, b in SERIES]
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)

daily, weekly, outlets = load_history()


def build(periods):
    fcs, actuals = {}, []
    for name, (y, w0, w1) in periods.items():
        ctx, act = make_context(daily, weekly, outlets, y, w0, w1)
        fcs[name] = Forecaster(ctx)
        actuals.append(act.assign(period=name))
    return fcs, pd.concat(actuals, ignore_index=True)


def summarize(s, chilled=None):
    row = {"wape_mean4": s["overall"]["mean"]["wape"], "wape_headline": s["overall"]["headline"]["wape"],
           "bias_mean4": s["overall"]["mean"]["bias"]}
    for b in BRANDS:
        row[f"{b}_wape_mean4"] = s["brand"][b]["mean"]["wape"]
        row[f"{b}_wape_headline"] = s["brand"][b]["headline"]["wape"]
    for sn in SERIES_NAMES:
        row[f"bias {sn}"] = s["series"][sn]["mean"]["bias"]
    if chilled is not None:
        row["chilled_wape_mean4"] = chilled["overall"]["mean"]["wape"]
        row["chilled_wape_headline"] = chilled["overall"]["headline"]["wape"]
        row["chilled_bias_mean4"] = chilled["overall"]["mean"]["bias"]
    return row


def fmt(df):
    out = df.copy()
    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].map(lambda v: f"{v:+.1%}" if "bias" in c else f"{v:.2%}")
    return out


if "--score-config" in sys.argv:
    # Selection-period metrics of the saved (possibly user-edited) config; no selection, no confirmation
    config = json.loads(CONFIG_PATH.read_text())["config"]
    fcs, actual = build(SELECTION_PERIODS)
    p = pd.concat([fc.run_config(config).assign(period=n) for n, fc in fcs.items()], ignore_index=True)
    s, c = score(p, actual), score(p, actual, "chilled")
    row = pd.DataFrame([summarize(s, c)])
    row.to_csv(REPORTS / "phase4_final_config_score.csv", index=False)
    print("Saved config:", json.dumps(config))
    print(fmt(row).T.to_string(header=False))
    print("Total WAPE per period:", {k: f"{v:.2%}" for k, v in s["per_period_overall"].items()})
    print("Per-series WAPE mean4:", {sn: f"{s['series'][sn]['mean']['wape']:.2%}" for sn in SERIES_NAMES})

elif "--confirm" not in sys.argv:
    fcs, actual = build(SELECTION_PERIODS)

    def preds(spec):
        return pd.concat([fc.run(spec).assign(period=p) for p, fc in fcs.items()], ignore_index=True)

    def brand_score(spec, brand):
        p = preds(spec)
        s = score(p[p["brand"] == brand], actual)
        return s["brand"][brand]

    def config_score(config):
        p = pd.concat([fc.run_config(config).assign(period=p) for p, fc in fcs.items()], ignore_index=True)
        return score(p, actual), score(p, actual, "chilled")

    def cand_row(name, spec, brand):
        s = brand_score(spec, brand)
        return {"brand": brand, "candidate": name, "spec": spec_key(spec), "wape_mean4": s["mean"]["wape"],
                "wape_headline": s["headline"]["wape"], "bias_mean4": s["mean"]["bias"]}

    candidates, improvement = [], []

    def add_row(label, config, note=""):
        s, c = config_score(config)
        improvement.append({"step": label, "note": note, **summarize(s, c)})

    def all_brands(spec, chilled):
        return {"Fresh": spec, "Style": spec, "Tech": spec, "chilled": chilled}

    # ------------------------------------------------ 0/1. GLM and the after_midweek_closure ablation
    G_on, G_off = {"model": "glm"}, {"model": "glm", "include_midweek": False}
    share = {"method": "share"}
    s_on, _ = config_score(all_brands(G_on, share))
    s_off, _ = config_score(all_brands(G_off, share))
    keep_mid = s_on["overall"]["mean"]["wape"] < s_off["overall"]["mean"]["wape"]
    G = G_on if keep_mid else G_off
    mid = {} if keep_mid else {"include_midweek": False}
    print("1. after_midweek_closure ablation (GLM, all brands, total WAPE):")
    print(f"   with: mean4 {s_on['overall']['mean']['wape']:.3%}, headline {s_on['overall']['headline']['wape']:.3%}; "
          f"without: mean4 {s_off['overall']['mean']['wape']:.3%}, headline {s_off['overall']['headline']['wape']:.3%}"
          f" -> {'KEEP' if keep_mid else 'DROP'}")
    for b in BRANDS:
        print(f"   {b}: with {s_on['brand'][b]['mean']['wape']:.3%}, without {s_off['brand'][b]['mean']['wape']:.3%}")
    add_row("0. Phase 3 GLM (all brands)", all_brands(G_on, share), "with after_midweek_closure")
    add_row("1. GLM without after_midweek_closure", all_brands(G_off, share), "flag dropped" if not keep_mid else "flag kept (variant rejected)")

    # ------------------------------------------------ 2. global HGB, one modest grid
    print("\n2. Global HistGradientBoosting (Poisson), grid on total WAPE (all brands):")
    grid = []
    for depth in (3, 4):
        for lr in (0.03, 0.06):
            for it in (300, 600):
                spec = {"model": "hgb", "max_depth": depth, "learning_rate": lr, "max_iter": it, **mid}
                s = score(preds(spec), actual)
                grid.append({"max_depth": depth, "learning_rate": lr, "max_iter": it,
                             "wape_mean4": s["overall"]["mean"]["wape"], "wape_headline": s["overall"]["headline"]["wape"],
                             **{f"{b}_wape": s["brand"][b]["mean"]["wape"] for b in BRANDS}, "spec": spec})
    grid = pd.DataFrame(grid).sort_values("wape_mean4")
    print(fmt(grid.drop(columns="spec")).to_string(index=False))
    H = grid.iloc[0]["spec"]
    print(f"   chosen: {spec_key(H)} (min_samples_leaf 40, l2 1.0)")
    add_row("2. HGB alone (all brands)", all_brands(H, share), spec_key(H))

    # ------------------------------------------------ 3. GLM + HGB blend, weight per brand
    print("\n3. GLM + HGB blend (weight on HGB), brand WAPE mean4:")
    blend_tab, best_w = {}, {}
    for b in BRANDS:
        blend_tab[b] = {w: brand_score({"model": "blend", "parts": [[G, 1 - w], [H, w]]}, b)["mean"]["wape"] for w in WEIGHTS}
        best_w[b] = min(blend_tab[b], key=blend_tab[b].get)
    print(fmt(pd.DataFrame(blend_tab).rename_axis("w_hgb")).to_string())
    print("   chosen weights on HGB:", best_w)

    def gh(w):
        return G if w == 0 else (H if w == 1 else {"model": "blend", "parts": [[G, round(1 - w, 1)], [H, w]]})

    cfg = {b: gh(best_w[b]) for b in BRANDS}
    cfg["chilled"] = share
    add_row("3. GLM+HGB blend (per-brand weight)", cfg, str(best_w))
    for b in BRANDS:
        candidates.append(cand_row(f"GLM+HGB w={best_w[b]}", cfg[b], b))
        candidates.append(cand_row("GLM", G, b))
        candidates.append(cand_row("HGB", H, b))

    # ------------------------------------------------ 4. Style: outlet x weekday, recency
    print("\n4. Style candidates (Style WAPE):")
    so = {"model": "style_outlet", **mid}
    style_specs = {"GLM": G, "GLM+HGB": cfg["Style"], "outlet x weekday": so}
    for hl in (26, 52):
        style_specs[f"outlet, half-life {hl}w"] = {**so, "halflife_weeks": hl}
        style_specs[f"GLM, half-life {hl}w"] = {**G, "halflife_weeks": hl}
    style_specs["outlet, last 52w"] = {**so, "window_weeks": 52}
    style_specs["GLM, last 52w"] = {**G, "window_weeks": 52}
    st = pd.DataFrame([cand_row(n, s, "Style") for n, s in style_specs.items()]).sort_values("wape_mean4")
    best_single = st[~st["candidate"].str.startswith("GLM+HGB")].iloc[0]
    S1 = json.loads(best_single["spec"])
    sb = {w: brand_score({"model": "blend", "parts": [[S1, 1 - w], [H, w]]}, "Style")["mean"]["wape"] for w in WEIGHTS}
    w_s = min(sb, key=sb.get)
    S_best = S1 if w_s == 0 else {"model": "blend", "parts": [[S1, round(1 - w_s, 1)], [H, w_s]]}
    st = pd.concat([st, pd.DataFrame([cand_row(f"{best_single['candidate']} + HGB w={w_s}", S_best, "Style")])])
    print(fmt(st.drop(columns=["brand", "spec"])).to_string(index=False))
    print(f"   best single Style model: {best_single['candidate']}; HGB blend weight {w_s}")
    candidates += st.to_dict("records")
    cfg4 = {**cfg, "Style": so}
    add_row("4. + Style outlet x weekday", cfg4, "Style = outlet GLM")
    cfg5 = {**cfg, "Style": S_best}
    add_row("5. + Style best (recency / blend)", cfg5, f"Style = {best_single['candidate']} + HGB w={w_s}")

    # ------------------------------------------------ 5. Tech
    print("\n5. Tech candidates (Tech WAPE):")
    tech = []
    smoothers = [("mean", n) for n in (8, 13, 26, 52)] + [("ewm", h) for h in (4, 8, 13, 26)]
    for kind, param in smoothers:
        tech.append(cand_row(f"(a) {kind} {param}", {"model": "tech_smooth", "kind": kind, "param": param}, "Tech"))
        tech.append(cand_row(f"(c) calendar-shaped, {kind} {param}",
                             {"model": "tech_calendar", "kind": kind, "param": param}, "Tech"))
    tech = pd.DataFrame(tech)
    A = json.loads(tech[tech["candidate"].str.startswith("(a)")].sort_values("wape_mean4").iloc[0]["spec"])
    C = json.loads(tech[tech["candidate"].str.startswith("(c)")].sort_values("wape_mean4").iloc[0]["spec"])
    tb = {w: brand_score({"model": "blend", "parts": [[G, round(1 - w, 1)], [A, w]]}, "Tech")["mean"]["wape"] for w in WEIGHTS}
    w_t = min(tb, key=tb.get)
    B = A if w_t == 1 else (G if w_t == 0 else {"model": "blend", "parts": [[G, round(1 - w_t, 1)], [A, w_t]]})
    tech = pd.concat([tech, pd.DataFrame([cand_row(f"(b) GLM + (a) best, w_smooth={w_t}", B, "Tech"),
                                          cand_row("GLM", G, "Tech"), cand_row(f"GLM+HGB w={best_w['Tech']}", cfg["Tech"], "Tech")])])
    tech = tech.sort_values("wape_mean4")
    print(fmt(tech.drop(columns=["brand", "spec"])).to_string(index=False))
    print("   (b) blend curve (weight on smoothed mean):", {w: f"{v:.2%}" for w, v in tb.items()})
    candidates += tech.to_dict("records")
    add_row("6. + Tech (a) smoothed mean", {**cfg5, "Tech": A}, spec_key(A))
    add_row("7. + Tech (b) GLM + smoothed blend", {**cfg5, "Tech": B}, f"w_smooth={w_t}")
    add_row("8. + Tech (c) calendar-shaped level", {**cfg5, "Tech": C}, spec_key(C))

    # ------------------------------------------------ final per-brand choice
    cand = pd.DataFrame(candidates).drop_duplicates(["brand", "spec"])
    final = {}
    for b in BRANDS:
        best = cand[cand["brand"] == b].sort_values("wape_mean4").iloc[0]
        final[b] = json.loads(best["spec"])
        print(f"   final {b}: {best['candidate']} (WAPE mean4 {best['wape_mean4']:.2%}, headline {best['wape_headline']:.2%})")

    # ------------------------------------------------ 6. chilled
    print("\n6. Chilled: GLM on chilled directly vs share (depot x month) x final Fresh total:")
    ch = {}
    for method in ("direct", "share"):
        c = {**final, "chilled": {"method": method, "glm": {"include_midweek": keep_mid} if not keep_mid else {}}}
        s_tot, s_ch = config_score(c)
        ch[method] = s_ch
        print(f"   {method:6s}: chilled WAPE mean4 {s_ch['overall']['mean']['wape']:.2%}, headline "
              f"{s_ch['overall']['headline']['wape']:.2%}, bias {s_ch['overall']['mean']['bias']:+.2%}; "
              + ", ".join(f"{sn} {s_ch['series'][sn]['mean']['wape']:.2%}" for sn in s_ch["series"]))
    chilled_method = min(ch, key=lambda m: ch[m]["overall"]["mean"]["wape"])
    final["chilled"] = {"method": chilled_method, "glm": {} if keep_mid else {"include_midweek": False}}
    add_row("9. Final (best per brand + chilled)", final, f"chilled={chilled_method}")

    imp = pd.DataFrame(improvement)
    imp.to_csv(REPORTS / "phase4_improvement.csv", index=False)
    cand.to_csv(REPORTS / "phase4_candidates.csv", index=False)
    print("\n=== Improvement table (total volume; mean over 4 selection periods and headline) ===")
    main_cols = ["step", "wape_mean4", "wape_headline", "bias_mean4"] + [f"{b}_wape_mean4" for b in BRANDS] + \
                ["chilled_wape_mean4", "chilled_wape_headline"]
    print(fmt(imp[main_cols]).to_string(index=False))
    print("\nMean signed bias per series (mean of 4 periods):")
    print(fmt(imp[["step"] + [f"bias {sn}" for sn in SERIES_NAMES]]).to_string(index=False))
    print("\nNotes:", "; ".join(f"{r['step'][:2]} {r['note']}" for _, r in imp.iterrows() if r["note"]))

    # per-period overall WAPE of the final config
    s_fin, _ = config_score(final)
    print("\nFinal config, total WAPE per period:", {k: f"{v:.2%}" for k, v in s_fin["per_period_overall"].items()})
    print("Final config, per-series WAPE mean4 / bias:",
          {sn: f"{s_fin['series'][sn]['mean']['wape']:.2%} / {s_fin['series'][sn]['mean']['bias']:+.2%}" for sn in SERIES_NAMES})
    # Never overwrite a config that carries the user's final decisions (Phase 4 approval)
    out_cfg = CONFIG_PATH
    if CONFIG_PATH.exists() and "user_decisions" in json.loads(CONFIG_PATH.read_text()):
        out_cfg = REPORTS / "phase4_config_rule.json"
    out_cfg.write_text(json.dumps({"config": final, "after_midweek_closure": keep_mid,
                                   "hgb": H, "selection_periods": list(SELECTION_PERIODS)}, indent=2))
    print(f"\nSaved: {REPORTS / 'phase4_improvement.csv'}, {REPORTS / 'phase4_candidates.csv'}, {out_cfg}")

else:
    out_path = REPORTS / "phase4_confirmation.csv"
    if out_path.exists() and "--force" not in sys.argv:
        sys.exit(f"{out_path} exists: the confirmation period has already been evaluated (use --force to override).")
    saved = json.loads(CONFIG_PATH.read_text())
    config = saved["config"]
    name = list(CONFIRMATION_PERIOD)[0]
    fcs, actual = build(CONFIRMATION_PERIOD)
    fc = fcs[name]
    models = {"recommended config": fc.run_config(config),
              "Phase 3 GLM (reference)": fc.run_config({b: {"model": "glm"} for b in BRANDS} | {"chilled": {"method": "direct", "glm": {}}})}
    base_rows = []
    for depot, brand in SERIES:
        wtr = fc.ctx.weekly_train[(fc.ctx.weekly_train["depot"] == depot) & (fc.ctx.weekly_train["brand"] == brand)]
        tw = fc.ctx.target_weeks[(fc.ctx.target_weeks["depot"] == depot) & (fc.ctx.target_weeks["brand"] == brand)]
        bt = baseline_forecasts(wtr.sort_values(["iso_year", "iso_week"]), tw, "total_volume")
        bc = baseline_forecasts(wtr.sort_values(["iso_year", "iso_week"]), tw, "chilled_volume")
        for b in ("last8_mean", "same_week_last_year"):
            base_rows.append(tw.assign(model=b, total=bt[b].values, chilled=bc[b].values))
    base = pd.concat(base_rows)
    for b in ("last8_mean", "same_week_last_year"):
        models[b] = base[base["model"] == b].drop(columns="model")
    rows = []
    for m, p in models.items():
        p = p.assign(period=name)
        s, c = score(p, actual, headline=name), score(p, actual, "chilled", headline=name)
        rows.append({"model": m, "total_wape": s["overall"]["mean"]["wape"], "total_bias": s["overall"]["mean"]["bias"],
                     **{f"{b}_wape": s["brand"][b]["mean"]["wape"] for b in BRANDS},
                     "chilled_wape": c["overall"]["mean"]["wape"],
                     **{f"bias {sn}": s["series"][sn]["mean"]["bias"] for sn in SERIES_NAMES}})
    res = pd.DataFrame(rows)
    res.to_csv(out_path, index=False)
    print(f"=== Confirmation period {name}: evaluated once ===")
    print(fmt(res).to_string(index=False))
    print(f"Saved: {out_path}")
