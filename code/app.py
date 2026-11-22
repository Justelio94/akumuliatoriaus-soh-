"""
app.py — Streamlit demonstracinė programa.
Paleidimas:  streamlit run code/app.py   (iš projekto šakninio katalogo)

Du režimai:
 1) Įkelti matavimų žurnalą (CSV iš OBD-II programėlės) — tiksliausia;
 2) Įvesti vieną matavimą ranka — apytikslis įvertinimas.
Programa tik PATARIA (informacinis sprendimas): ji nevaldo jokio įrenginio,
sprendimą keisti akumuliatorių priima žmogus.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from model import EOL_SOH, REPLACE_SOON, TEMP_COEF, build_features, clean  # noqa: E402

MD = ROOT / "data" / "models"
FEATS = json.loads((MD / "features.json").read_text())


def load_models():
    m = {}
    m["rul_linear"] = json.loads((MD / "rul_linear.json").read_text())
    for k in ["soh_xgb", "rul_resid_xgb", "rul_quantile_xgb"]:
        r = xgb.XGBRegressor(); r.load_model(MD / f"{k}.json"); m[k] = r
    return m


def single_row_features(cycle, temp, ocv, v_min, t_charge, temp_avg, short_trip):
    dv = ocv - v_min
    dv_25 = dv / max(1 + TEMP_COEF * (25 - temp), 0.3)
    return pd.DataFrame([{
        "cycle": cycle, "temp_c": temp, "ocv_v": ocv, "dv": dv, "dv_25": dv_25,
        "dv_25_roll": dv_25, "dv_25_slope": 0.0, "charge_time_min": t_charge,
        "charge_roll": t_charge, "temp_mean_cum": temp_avg,
        "heat_stress_cum": cycle * 2.0 ** ((temp_avg + 6 - 25) / 10),
        "short_trip_share": short_trip}])[FEATS]


def predict(models, X):
    soh = float(models["soh_xgb"].predict(X)[0])
    lin = models["rul_linear"]
    rul_lin = float(X.values[0] @ np.array(lin["coef"]) + lin["intercept"])
    rul = max(rul_lin + float(models["rul_resid_xgb"].predict(X)[0]), 0.0)  # hibridas
    q = np.clip(models["rul_quantile_xgb"].predict(X)[0], 0, None)
    if soh <= EOL_SOH:
        rul, q = 0.0, np.array([0.0, 0.0])
    return soh, rul, q


def recommendation(soh, rul):
    if soh <= EOL_SOH or rul <= REPLACE_SOON:
        return "🔴 Rekomenduojama planuoti akumuliatoriaus keitimą artimiausiu metu."
    if rul <= 3 * REPLACE_SOON:
        return "🟠 Akumuliatorius senėja — patikrinkite prieš žiemą, stebėkite užvedimą."
    return "🟢 Akumuliatoriaus būklė gera."


def main():
    import streamlit as st
    st.set_page_config(page_title="Akumuliatoriaus SOH", page_icon="🔋")
    st.title("🔋 Automobilio akumuliatoriaus būklės (SOH) vertinimas")
    st.caption("Informacinis įrankis. Galutinį sprendimą priima vairuotojas arba servisas.")
    models = load_models()

    mode = st.radio("Duomenų įvedimas", ["Įkelti matavimų žurnalą (CSV)", "Vienas matavimas ranka"])
    if mode.startswith("Įkelti"):
        up = st.file_uploader("CSV stulpeliai: cycle, temp_c, ocv_v, v_crank_min_v, "
                              "charge_time_min, short_trip_share", type="csv")
        if up is None:
            st.info("Pavyzdinis žurnalas: data/demo_log.csv")
            return
        log = pd.read_csv(up)
        log["battery_id"] = 0
        df, _ = clean(log)
        df = build_features(df)
        X = df[FEATS].tail(1)
        df["soh_pred"] = models["soh_xgb"].predict(df[FEATS])
        st.line_chart(df.set_index("cycle")[["soh_pred"]], y_label="Prognozuota SOH")
    else:
        c1, c2 = st.columns(2)
        cycle = c1.number_input("Ciklų (užvedimų) skaičius", 0, 10000, 1500, 50)
        temp = c1.number_input("Dabartinė temperatūra, °C", -30.0, 45.0, 5.0)
        temp_avg = c1.number_input("Vid. metinė temperatūra regione, °C", -5.0, 25.0, 7.0)
        ocv = c2.number_input("Ramybės įtampa (rytą), V", 11.0, 13.2, 12.55, 0.01)
        v_min = c2.number_input("Min. įtampa užvedant, V", 6.0, 12.8, 10.3, 0.05)
        t_ch = c2.number_input("Laikas iki 14,2 V, min", 0.5, 40.0, 7.0, 0.5)
        short = st.slider("Trumpų kelionių (< 15 min) dalis", 0.0, 1.0, 0.3)
        X = single_row_features(cycle, temp, ocv, v_min, t_ch, temp_avg, short)

    soh, rul, q = predict(models, X)
    a, b = st.columns(2)
    a.metric("SOH", f"{soh*100:.1f} %")
    b.metric("Likęs tarnavimo laikas", f"~{rul:.0f} ciklų",
             help=f"80 % intervalas: {q[0]:.0f}–{q[1]:.0f} ciklų")
    st.write(recommendation(soh, rul))


if __name__ == "__main__":
    main()
