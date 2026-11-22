"""
model.py — duomenų valymas, požymių inžinerija, Baseline vs Proposed
eksperimentai SOH (sveikatos būklės) ir RUL (likusio tarnavimo laiko)
prognozei, SHAP paaiškinamumas.

Paleidimas:  python code/model.py   (iš projekto šakninio katalogo)
Rezultatai:  data/battery_clean.csv, data/metrics.json, data/predictions_*.csv,
             data/models/*.json (XGBoost modeliai demonstracinei programai)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split

SEED = 42
DATA = Path("data")
EOL_SOH = 0.80
REPLACE_SOON = 200          # „keisti artimiausiu metu“: RUL <= 200 ciklų (~3 mėn.)
TEMP_COEF = 0.03            # literatūrinis varžos temp. koeficientas ~2–3 %/°C

# Fiziškai galimos ribos 12 V švino-rūgšties akumuliatoriui
RANGES = {"ocv_v": (10.5, 13.3), "v_crank_min_v": (5.0, 13.0),
          "temp_c": (-40, 60), "charge_time_min": (0.2, 60)}

FEATS_ALL = ["cycle", "temp_c", "ocv_v", "dv", "dv_25", "dv_25_roll",
             "dv_25_slope", "charge_time_min", "charge_roll",
             "temp_mean_cum", "heat_stress_cum", "short_trip_share"]
FEATS_NO_TEMP = ["cycle", "ocv_v", "dv", "charge_time_min", "short_trip_share"]


# ---------------------------------------------------------------- valymas
def clean(raw):
    df = raw.sort_values(["battery_id", "cycle"]).copy()
    report = {"rows": int(len(df)), "missing_before": {}, "outliers_removed": {}}
    for c in RANGES:
        report["missing_before"][c] = int(df[c].isna().sum())
    for c, (lo, hi) in RANGES.items():
        bad = (df[c] < lo) | (df[c] > hi)
        report["outliers_removed"][c] = int(bad.sum())
        df.loc[bad, c] = np.nan
    # Interpoliacija kiekvieno akumuliatoriaus laiko eilutėje
    cols = list(RANGES)
    df[cols] = df.groupby("battery_id")[cols].transform(
        lambda s: s.interpolate(limit_direction="both"))
    report["missing_after"] = int(df[cols].isna().sum().sum())
    return df, report


# ------------------------------------------------------------- požymiai
def build_features(df):
    df = df.copy()
    g = df.groupby("battery_id")
    df["dv"] = df.ocv_v - df.v_crank_min_v
    # Įtampos kritimas perskaičiuotas į 25 °C (tiesinė temperatūrinė kompensacija)
    df["dv_25"] = df.dv / (1 + TEMP_COEF * (25 - df.temp_c)).clip(lower=0.3)
    g = df.groupby("battery_id")
    df["dv_25_roll"] = g.dv_25.transform(lambda s: s.rolling(8, min_periods=1).mean())
    df["charge_roll"] = g.charge_time_min.transform(
        lambda s: s.rolling(8, min_periods=1).mean())

    def slope(s, w=15):
        x = np.arange(w)
        return s.rolling(w, min_periods=3).apply(
            lambda y: np.polyfit(x[:len(y)], y, 1)[0], raw=True)
    df["dv_25_slope"] = g.dv_25_roll.transform(slope).fillna(0.0)
    df["temp_mean_cum"] = g.temp_c.transform(lambda s: s.expanding().mean())
    # Sukauptas šiluminis stresas (Arenijaus taisyklė: x2 kas +10 °C)
    df["heat_stress_cum"] = g.temp_c.transform(
        lambda s: (2.0 ** ((s + 6 - 25) / 10) * 10).cumsum())
    return df


# ---------------------------------------------------------------- metrikos
def reg_metrics(y, p):
    return {"RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAE": float(mean_absolute_error(y, p)),
            "R2": float(r2_score(y, p))}


def clf_metrics(y_true_rul, y_pred_rul):
    t = (y_true_rul <= REPLACE_SOON).astype(int)
    p = (y_pred_rul <= REPLACE_SOON).astype(int)
    return {"F1": float(f1_score(t, p)), "Precision": float(precision_score(t, p)),
            "Recall": float(recall_score(t, p)),
            "Accuracy": float((t == p).mean()),
            "confusion": confusion_matrix(t, p).tolist()}


def make_xgb(**kw):
    params = dict(n_estimators=1500, learning_rate=0.03, max_depth=5,
                  min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                  reg_lambda=1.0, random_state=SEED, n_jobs=4,
                  early_stopping_rounds=100, eval_metric="rmse")
    params.update(kw)
    return xgb.XGBRegressor(**params)


def main():
    raw = pd.read_csv(DATA / "battery_raw.csv")
    df, clean_report = clean(raw)
    df = build_features(df)
    df.to_csv(DATA / "battery_clean.csv", index=False)
    M = {"cleaning": clean_report,
         "dataset": {"batteries": int(df.battery_id.nunique()), "rows": int(len(df))}}

    # --- skaidymas pagal akumuliatorių (jokio duomenų nutekėjimo) ----------
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    trval_idx, te_idx = next(gss.split(df, groups=df.battery_id))
    trval, test = df.iloc[trval_idx], df.iloc[te_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.18, random_state=SEED)
    tr_i, va_i = next(gss2.split(trval, groups=trval.battery_id))
    train, val = trval.iloc[tr_i], trval.iloc[va_i]
    M["split"] = {k: {"batteries": int(d.battery_id.nunique()), "rows": int(len(d))}
                  for k, d in [("train", train), ("val", val), ("test", test)]}

    # ====================== 1. SOH ==========================================
    soh = {}
    preds = test[["battery_id", "cycle", "temp_c", "soh_true", "rul_cycles",
                  "cycles_per_day"]].copy()

    lr = LinearRegression().fit(train[["cycle"]], train.soh_true)
    preds["soh_baseline"] = lr.predict(test[["cycle"]])
    soh["Baseline: LinearRegression (tik ciklai)"] = reg_metrics(test.soh_true, preds.soh_baseline)

    lr_all = LinearRegression().fit(train[FEATS_ALL], train.soh_true)
    preds["soh_lr_all"] = lr_all.predict(test[FEATS_ALL])
    soh["LinearRegression (visi požymiai)"] = reg_metrics(test.soh_true, preds.soh_lr_all)

    rf = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=5,
                               random_state=SEED, n_jobs=4).fit(train[FEATS_ALL], train.soh_true)
    preds["soh_rf"] = rf.predict(test[FEATS_ALL])
    soh["RandomForest (visi požymiai)"] = reg_metrics(test.soh_true, preds.soh_rf)

    xs = make_xgb()
    xs.fit(train[FEATS_ALL], train.soh_true,
           eval_set=[(train[FEATS_ALL], train.soh_true), (val[FEATS_ALL], val.soh_true)],
           verbose=False)
    preds["soh_xgb"] = xs.predict(test[FEATS_ALL])
    soh["Proposed: XGBoost (visi požymiai)"] = reg_metrics(test.soh_true, preds.soh_xgb)
    ev = xs.evals_result()
    pd.DataFrame({"train_rmse": ev["validation_0"]["rmse"],
                  "val_rmse": ev["validation_1"]["rmse"]}).to_csv(
        DATA / "training_curve_soh.csv", index_label="iteration")
    M["soh_best_iteration"] = int(xs.best_iteration)

    # Abliacija: be temperatūros informacijos (nesėkmės eksperimentas)
    xs_nt = make_xgb().fit(train[FEATS_NO_TEMP], train.soh_true,
                           eval_set=[(val[FEATS_NO_TEMP], val.soh_true)], verbose=False)
    preds["soh_xgb_notemp"] = xs_nt.predict(test[FEATS_NO_TEMP])
    soh["Abliacija: XGBoost be temperatūros"] = reg_metrics(test.soh_true, preds.soh_xgb_notemp)
    M["soh"] = soh

    # Paklaida pagal temperatūros intervalus
    bins = [-40, -5, 5, 15, 60]
    labels = ["< -5 °C", "-5…5 °C", "5…15 °C", "> 15 °C"]
    preds["temp_bin"] = pd.cut(preds.temp_c, bins, labels=labels)
    by_t = {}
    for b, d in preds.groupby("temp_bin", observed=True):
        by_t[str(b)] = {
            "n": int(len(d)),
            "MAE_xgb": float(mean_absolute_error(d.soh_true, d.soh_xgb)),
            "MAE_xgb_notemp": float(mean_absolute_error(d.soh_true, d.soh_xgb_notemp)),
            "MAE_baseline": float(mean_absolute_error(d.soh_true, d.soh_baseline))}
    M["soh_error_by_temp"] = by_t

    # GroupKFold kryžminė patikra (stabilumui įvertinti)
    gkf = GroupKFold(n_splits=5)
    cv = {"baseline": [], "xgb": []}
    for tr, te in gkf.split(trval, groups=trval.battery_id):
        a, b = trval.iloc[tr], trval.iloc[te]
        m0 = LinearRegression().fit(a[["cycle"]], a.soh_true)
        cv["baseline"].append(np.sqrt(mean_squared_error(b.soh_true, m0.predict(b[["cycle"]]))))
        m1 = make_xgb(n_estimators=max(xs.best_iteration, 50), early_stopping_rounds=None)
        m1.fit(a[FEATS_ALL], a.soh_true, verbose=False)
        cv["xgb"].append(np.sqrt(mean_squared_error(b.soh_true, m1.predict(b[FEATS_ALL]))))
    M["soh_cv_rmse"] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                        for k, v in cv.items()}

    # ====================== 2. RUL ==========================================
    pre = lambda d: d[d.soh_true > EOL_SOH]
    trR, vaR, teR = pre(train), pre(val), pre(test)
    rul = {}
    lr_r = LinearRegression().fit(trR[["cycle"]], trR.rul_cycles)
    xr = make_xgb(objective="reg:squarederror")
    xr.fit(trR[FEATS_ALL], trR.rul_cycles, eval_set=[(vaR[FEATS_ALL], vaR.rul_cycles)],
           verbose=False)
    p_base = np.clip(lr_r.predict(teR[["cycle"]]), 0, None)
    p_xgb = np.clip(xr.predict(teR[FEATS_ALL]), 0, None)
    rul["Baseline: LinearRegression (tik ciklai)"] = reg_metrics(teR.rul_cycles, p_base)
    lr_r_all = LinearRegression().fit(trR[FEATS_ALL], trR.rul_cycles)
    rul["LinearRegression (visi požymiai)"] = reg_metrics(
        teR.rul_cycles, np.clip(lr_r_all.predict(teR[FEATS_ALL]), 0, None))
    # 1-as bandymas (atmestas kaip galutinis): grynas XGBoost
    rul["1-as bandymas: XGBoost (grynas)"] = reg_metrics(teR.rul_cycles, p_xgb)
    # Galutinis: hibridas = tiesinė tendencija + XGBoost liekanų korekcija
    res = lambda d: d.rul_cycles - lr_r_all.predict(d[FEATS_ALL])
    xh = make_xgb(max_depth=3, learning_rate=0.02)
    xh.fit(trR[FEATS_ALL], res(trR), eval_set=[(vaR[FEATS_ALL], res(vaR))], verbose=False)
    hybrid = lambda X: np.clip(lr_r_all.predict(X) + xh.predict(X), 0, None)
    p_hyb = hybrid(teR[FEATS_ALL])
    rul["Proposed: LR + XGBoost liekanos (hibridas)"] = reg_metrics(teR.rul_cycles, p_hyb)
    M["rul"] = rul

    # Neapibrėžtis: kvantilinė regresija (10 % ir 90 %)
    xq = make_xgb(objective="reg:quantileerror", quantile_alpha=np.array([0.1, 0.9]),
                  eval_metric=None)
    xq.fit(trR[FEATS_ALL], trR.rul_cycles, eval_set=[(vaR[FEATS_ALL], vaR.rul_cycles)],
           verbose=False)
    q = np.clip(xq.predict(teR[FEATS_ALL]), 0, None)
    cover = float(((teR.rul_cycles >= q[:, 0]) & (teR.rul_cycles <= q[:, 1])).mean())
    M["rul_interval_80_coverage"] = cover
    M["rul_interval_mean_width"] = float(np.mean(q[:, 1] - q[:, 0]))

    pr = teR[["battery_id", "cycle", "soh_true", "rul_cycles", "cycles_per_day"]].copy()
    pr["rul_baseline"], pr["rul_xgb_pure"], pr["rul_xgb"] = p_base, p_xgb, p_hyb
    pr["rul_q10"], pr["rul_q90"] = q[:, 0], q[:, 1]
    pr.to_csv(DATA / "predictions_rul.csv", index=False)
    # RUL mėnesiais (pagal kiekvieno automobilio užvedimų/dieną)
    m_true = pr.rul_cycles / pr.cycles_per_day / 30.4
    M["rul_months_MAE"] = {
        "baseline": float(mean_absolute_error(m_true, pr.rul_baseline / pr.cycles_per_day / 30.4)),
        "xgb": float(mean_absolute_error(m_true, pr.rul_xgb / pr.cycles_per_day / 30.4))}

    # RUL paklaida pagal akumuliatoriaus būklę (ankstyvas vs vėlyvas gyvavimas)
    sb = pd.cut(pr.soh_true, [0.80, 0.85, 0.90, 0.95, 1.0],
                labels=["0,80–0,85", "0,85–0,90", "0,90–0,95", "0,95–1,00"])
    M["rul_mae_by_soh"] = {str(k): {"n": int(len(d)),
                                    "MAE_baseline": float(mean_absolute_error(d.rul_cycles, d.rul_baseline)),
                                    "MAE_xgb": float(mean_absolute_error(d.rul_cycles, d.rul_xgb)),
                                    "MAE_xgb_pure": float(mean_absolute_error(d.rul_cycles, d.rul_xgb_pure))}
                           for k, d in pr.groupby(sb, observed=True)}

    # „Keisti artimiausiu metu“ klasifikacija (visos testo eilutės, įskaitant po EOL)
    p_all_base = np.clip(lr_r.predict(test[["cycle"]]), 0, None)
    p_all_xgb = hybrid(test[FEATS_ALL])
    p_all_xgb = np.where(preds.soh_xgb <= EOL_SOH, 0, p_all_xgb)  # SOH jau <= 80 % -> keisti
    M["replace_soon"] = {
        "Baseline: LinearRegression (tik ciklai)": clf_metrics(test.rul_cycles.values, p_all_base),
        "Proposed: XGBoost SOH + hibridinis RUL": clf_metrics(test.rul_cycles.values, p_all_xgb)}
    preds["rul_baseline_all"], preds["rul_xgb_all"] = p_all_base, p_all_xgb
    preds.to_csv(DATA / "predictions_soh.csv", index=False)

    # ====================== 3. Nesėkmė: atsitiktinis eilučių skaidymas =====
    a, b = train_test_split(pre(df), test_size=0.15, random_state=SEED)
    leak = make_xgb(early_stopping_rounds=None, n_estimators=600)
    leak.fit(a[FEATS_ALL], a.rul_cycles, verbose=False)
    M["leakage_random_split_rul"] = reg_metrics(b.rul_cycles, leak.predict(b[FEATS_ALL]))

    # ====================== 4. SHAP =========================================
    sample = test[FEATS_ALL].sample(min(1500, len(test)), random_state=SEED)
    sv = shap.TreeExplainer(xs).shap_values(sample)
    np.save(DATA / "shap_values_soh.npy", sv)
    sample.to_csv(DATA / "shap_sample_soh.csv", index=False)
    M["shap_mean_abs_soh"] = dict(sorted(
        {f: float(v) for f, v in zip(FEATS_ALL, np.abs(sv).mean(0))}.items(),
        key=lambda kv: -kv[1]))

    # ====================== 4b. Vieno matavimo režimas (be istorijos) =====
    # Kaip demonstracinėje programoje: slenkantys vidurkiai = dabartinė reikšmė,
    # tendencija = 0, šiluminis stresas apytiksliai iš klimato vidurkio.
    one = test.copy()
    one["dv_25_roll"], one["charge_roll"], one["dv_25_slope"] = one.dv_25, one.charge_time_min, 0.0
    one["temp_mean_cum"] = one.climate_mean_c
    one["heat_stress_cum"] = one.cycle * 2.0 ** ((one.climate_mean_c + 6 - 25) / 10)
    M["single_measurement_soh"] = reg_metrics(one.soh_true, xs.predict(one[FEATS_ALL]))

    # ====================== 5. Išsaugojimas =================================
    (DATA / "models").mkdir(exist_ok=True)
    xs.save_model(DATA / "models" / "soh_xgb.json")
    xh.save_model(DATA / "models" / "rul_resid_xgb.json")
    (DATA / "models" / "rul_linear.json").write_text(json.dumps(
        {"coef": lr_r_all.coef_.tolist(), "intercept": float(lr_r_all.intercept_)}))
    xq.save_model(DATA / "models" / "rul_quantile_xgb.json")
    (DATA / "models" / "features.json").write_text(json.dumps(FEATS_ALL))
    M["test_battery_ids"] = sorted(int(i) for i in test.battery_id.unique())
    (DATA / "metrics.json").write_text(json.dumps(M, indent=2, ensure_ascii=False))
    print(json.dumps({k: M[k] for k in ["soh", "rul", "replace_soon", "leakage_random_split_rul",
                                        "rul_interval_80_coverage", "soh_cv_rmse",
                                        "rul_months_MAE", "soh_error_by_temp"]},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
