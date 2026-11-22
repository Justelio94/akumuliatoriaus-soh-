"""
generate_data.py — sintetinių 12 V švino-rūgšties automobilio akumuliatorių
senėjimo duomenų generatorius.

Imituojamas automobilių parkas (N akumuliatorių). Kiekvienas akumuliatorius
eksploatuojamas skirtingame klimate, su skirtingu važiavimo profiliu
(trumpų kelionių dalis, užvedimų per dieną) ir gamybine sklaida.
Kas MEAS_EVERY ciklų (užvedimas + įkrovimas generatoriumi) užfiksuojami tik
tie dydžiai, kuriuos galima išmatuoti BE specialios įrangos
(OBD-II adapteris už ~10 € ir telefono programėlė arba paprastas multimetras):

  * ramybės įtampa (OCV) po nakties stovėjimo, V
  * minimali įtampa užvedimo metu, V  -> įtampos kritimas dV = OCV - V_min
  * temperatūra (OBD įsiurbiamo oro / aplinkos), °C
  * įkrovimo laikas, kol generatorius pakelia įtampą iki 14,2 V, min
  * ciklų skaičius

Tikroji SOH (talpos santykis C/C_nom) žinoma tik generatoriui — ji yra
modelio taikinys. RUL = ciklų skaičius, kol SOH nukris iki 0,80.

Fizikinės prielaidos (žr. ataskaitos IV skyrių):
  * Senėjimo greitis auga dvigubai kas +10 °C (Arenijaus tipo taisyklė).
  * Trumpos kelionės -> nepilnas įkrovimas -> sulfatacija -> greitesnis senėjimas.
  * Artėjant prie gyvavimo pabaigos senėjimas greitėja („kelio“ efektas).
  * Vidinė varža auga mažėjant SOH ir šaltyje; dV = I_užv * R(T).
Į neapdorotus duomenis tyčia įterpiami jutiklių triukšmas, praleistos reikšmės
ir fiziškai neįmanomi šuoliai, kad duomenų valymo etapas būtų realus.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EOL_SOH = 0.80
MEAS_EVERY = 10          # matavimas kas 10 ciklų
STOP_SOH = 0.70          # simuliacija tęsiama iki šios ribos


def simulate_battery(bid, rng):
    # --- akumuliatoriaus ir eksploatacijos savybės -------------------------
    climate_mean = rng.uniform(-1.0, 17.0)       # metinė vidutinė temp., °C
    climate_amp = rng.uniform(8.0, 14.0)         # sezoninė amplitudė, °C
    short_trip = rng.beta(2.0, 3.0)              # trumpų kelionių dalis 0..1
    cycles_per_day = rng.uniform(1.0, 3.0)
    unit_var = rng.lognormal(0.0, 0.18)          # gamybinė sklaida
    R0 = rng.normal(0.0060, 0.0005)              # nauja vidinė varža, Ω
    start_doy = rng.integers(0, 365)

    base = 0.95e-4                               # bazinis talpos nykimas / ciklą
    soh, cycle = 1.0, 0
    rows = []
    while soh > STOP_SOH and cycle < 8000:
        day = start_doy + cycle / cycles_per_day
        T = climate_mean + climate_amp * np.sin(2 * np.pi * (day - 110) / 365) \
            + rng.normal(0, 3.0)
        T_batt = T + 6.0                         # po kapotu šilčiau
        acc_T = 2.0 ** ((T_batt - 25.0) / 10.0)
        acc_trip = 1.0 + 1.6 * short_trip
        knee = 1.0 + 4.0 * max(0.0, 0.86 - soh) / 0.16
        soh -= base * acc_T * acc_trip * knee * unit_var * rng.lognormal(0, 0.1)
        cycle += 1

        if cycle % MEAS_EVERY == 0:
            soc = np.clip(0.96 - 0.28 * short_trip + rng.normal(0, 0.03), 0.4, 1.0)
            R = R0 * (1 + 4.0 * (1 - soh) ** 1.3) * np.exp(0.025 * (25 - T))
            I_crank = 150.0 * (1 + 0.012 * (25 - T))
            ocv = 11.90 + 0.80 * soc - 0.6 * (1 - soh) - 0.002 * (25 - T) \
                + rng.normal(0, 0.02)
            dv = I_crank * R + rng.normal(0, 0.06)
            t_charge = (4.0 + 26.0 * (1 - soc)) * soh ** 2 \
                * (1 + 0.015 * (25 - T)) + rng.normal(0, 0.6)
            rows.append(dict(
                battery_id=bid, cycle=cycle,
                temp_c=round(T, 2),
                ocv_v=round(ocv, 3),
                v_crank_min_v=round(ocv - dv, 3),
                charge_time_min=round(max(t_charge, 0.5), 2),
                short_trip_share=round(short_trip, 3),
                cycles_per_day=round(cycles_per_day, 3),
                soh_true=round(soh, 4),
            ))
    df = pd.DataFrame(rows)
    # RUL: ciklai iki pirmo SOH <= EOL (tiesinė interpoliacija tarp matavimų)
    below = df.index[df.soh_true <= EOL_SOH]
    eol_cycle = df.loc[below[0], "cycle"] if len(below) else df.cycle.max()
    df["eol_cycle"] = eol_cycle
    df["rul_cycles"] = (eol_cycle - df.cycle).clip(lower=0)
    df["climate_mean_c"] = round(climate_mean, 2)
    return df


def corrupt(df, rng):
    """Įterpia realistiškus jutiklių defektus (praleistos reikšmės, šuoliai)."""
    df = df.copy()
    n = len(df)
    miss = rng.random(n) < 0.015
    df.loc[miss, "temp_c"] = np.nan
    miss2 = rng.random(n) < 0.010
    df.loc[miss2, "charge_time_min"] = np.nan
    spike = rng.random(n) < 0.006
    df.loc[spike, "v_crank_min_v"] = rng.choice([0.0, 15.8, 3.2], spike.sum())
    spike2 = rng.random(n) < 0.004
    df.loc[spike2, "ocv_v"] = rng.choice([0.0, 16.5], spike2.sum())
    return df


def main(n=80, seed=42, out="data"):
    rng = np.random.default_rng(seed)
    df = pd.concat([simulate_battery(i, rng) for i in range(n)], ignore_index=True)
    df = corrupt(df, rng)
    Path(out).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(out) / "battery_raw.csv", index=False)
    print(f"Sugeneruota {len(df)} matavimų iš {n} akumuliatorių -> {out}/battery_raw.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    main(a.n, a.seed, a.out)

"""
make_figures.py — visi ataskaitos paveikslai iš data/ rezultatų.
Paleidimas: python code/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.patches import FancyBboxPatch

D, F = Path("data"), Path("figures")
F.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3})
C_BASE, C_PROP, C_TRUE, C_EOL = "#9a9a9a", "#1f6fb4", "#222222", "#c0392b"

raw = pd.read_csv(D / "battery_raw.csv")
df = pd.read_csv(D / "battery_clean.csv")
ps = pd.read_csv(D / "predictions_soh.csv")
pr = pd.read_csv(D / "predictions_rul.csv")
M = json.loads((D / "metrics.json").read_text())


def save(fig, name):
    fig.tight_layout()
    fig.savefig(F / name, bbox_inches="tight")
    plt.close(fig)


# 1 pav. SOH trajektorijos pagal klimatą
fig, ax = plt.subplots(figsize=(7, 3.6))
cmap = plt.get_cmap("coolwarm")
bids = df.battery_id.unique()
for b in bids:
    d = df[df.battery_id == b]
    cm = d.climate_mean_c.iloc[0]
    ax.plot(d.cycle, d.soh_true, color=cmap((cm + 1) / 18), lw=0.8, alpha=0.75)
ax.axhline(0.80, color=C_EOL, ls="--", lw=1.2)
ax.text(ax.get_xlim()[1] * 0.98, 0.805, "EOL riba: SOH = 0,80", color=C_EOL,
        ha="right", va="bottom")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-1, 17))
fig.colorbar(sm, ax=ax, label="Vid. metinė temperatūra, °C")
ax.set_xlabel("Ciklų skaičius"); ax.set_ylabel("SOH (C / C_nom)")
ax.set_title("Visų 80 akumuliatorių SOH trajektorijos")
save(fig, "fig1_soh_trajektorijos.png")

# 2 pav. Neapdoroti vs išvalyti duomenys (vieno akumuliatoriaus pavyzdys)
bad_b = raw[(raw.v_crank_min_v < 5) | (raw.v_crank_min_v > 13)].battery_id.value_counts().index[0]
r, c = raw[raw.battery_id == bad_b], df[df.battery_id == bad_b]
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].plot(r.cycle, r.v_crank_min_v, ".", ms=2.5, color=C_EOL, label="neapdoroti")
ax[0].plot(c.cycle, c.v_crank_min_v, "-", lw=0.8, color=C_PROP, label="išvalyti")
ax[0].set_title(f"Min. užvedimo įtampa, akumuliatorius #{bad_b}")
ax[0].set_xlabel("Ciklai"); ax[0].set_ylabel("V"); ax[0].legend()
ax[1].plot(c.cycle, c.dv, lw=0.6, color=C_BASE, label="ΔV (išmatuotas)")
ax[1].plot(c.cycle, c.dv_25_roll, lw=1.4, color=C_PROP, label="ΔV₂₅ (kompensuotas, slenkantis vid.)")
ax[1].set_title("Įtampos kritimas: temperatūrinė kompensacija")
ax[1].set_xlabel("Ciklai"); ax[1].set_ylabel("V"); ax[1].legend(fontsize=7)
save(fig, "fig2_valymas_kompensacija.png")

# 3 pav. Histogramos + koreliacijų matrica
feats = ["soh_true", "cycle", "temp_c", "ocv_v", "dv", "dv_25_roll",
         "charge_time_min", "heat_stress_cum", "short_trip_share"]
names = ["SOH", "Ciklai", "Temp.", "OCV", "ΔV", "ΔV₂₅ roll", "Įkrov. laikas",
         "Šil. stresas", "Trump. kel."]
fig = plt.figure(figsize=(9, 3.8))
for i, (f, n) in enumerate(zip(["dv", "temp_c", "charge_time_min", "ocv_v"],
                               ["ΔV, V", "Temperatūra, °C", "Įkrovimo laikas, min", "OCV, V"])):
    a = fig.add_subplot(2, 4, [1, 2, 5, 6][i])
    a.hist(df[f], bins=40, color=C_PROP, alpha=0.85)
    a.set_title(n, fontsize=8); a.tick_params(labelsize=7)
a = fig.add_subplot(1, 2, 2)
corr = df[feats].corr()
im = a.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
a.set_xticks(range(len(names))); a.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
a.set_yticks(range(len(names))); a.set_yticklabels(names, fontsize=7)
for i in range(len(feats)):
    for j in range(len(feats)):
        a.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
               color="white" if abs(corr.iloc[i, j]) > 0.6 else "black")
a.grid(False); a.set_title("Pirsono koreliacijos")
fig.colorbar(im, ax=a, fraction=0.04)
save(fig, "fig3_histogramos_koreliacijos.png")

# 4 pav. Mokymo kreivė
tc = pd.read_csv(D / "training_curve_soh.csv")
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(tc.iteration, tc.train_rmse, color=C_BASE, label="mokymo RMSE")
ax.plot(tc.iteration, tc.val_rmse, color=C_PROP, label="validavimo RMSE")
ax.axvline(M["soh_best_iteration"], color=C_EOL, ls="--", lw=1,
           label=f"geriausia iteracija ({M['soh_best_iteration']})")
ax.set_yscale("log"); ax.set_xlabel("Medžių skaičius (iteracija)"); ax.set_ylabel("RMSE (SOH)")
ax.set_title("XGBoost SOH modelio mokymo kreivė (early stopping)"); ax.legend()
save(fig, "fig4_mokymo_kreive.png")

# 5 pav. Prognozė vs tikroji SOH
fig, ax = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
for a, col, t, clr in [(ax[0], "soh_baseline", "Baseline: LinearRegression (ciklai)", C_BASE),
                       (ax[1], "soh_xgb", "Proposed: XGBoost", C_PROP)]:
    a.scatter(ps.soh_true, ps[col], s=3, alpha=0.35, color=clr)
    a.plot([0.7, 1.0], [0.7, 1.0], color=C_TRUE, lw=1)
    rm = np.sqrt(((ps.soh_true - ps[col]) ** 2).mean())
    a.set_title(f"{t}\nRMSE = {rm:.4f}")
    a.set_xlabel("Tikroji SOH"); a.set_xlim(0.69, 1.01); a.set_ylim(0.6, 1.05)
ax[0].set_ylabel("Prognozuota SOH")
save(fig, "fig5_soh_prognoze_vs_tikroji.png")

# 6 pav. SHAP
sv = np.load(D / "shap_values_soh.npy")
X = pd.read_csv(D / "shap_sample_soh.csv")
shap.summary_plot(sv, X, show=False, max_display=12, plot_size=(7, 4))
plt.title("SHAP: požymių įtaka SOH prognozei (XGBoost)")
plt.savefig(F / "fig6_shap.png", bbox_inches="tight", dpi=150); plt.close()

# 7 pav. RUL vienam testo akumuliatoriui + paklaida pagal SOH
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
life = pr.groupby("battery_id").rul_cycles.max()
b = (life - life.median()).abs().idxmin()   # tipinio (medianinio) gyvavimo akumuliatorius
d = pr[pr.battery_id == b]
ax[0].fill_between(d.cycle, d.rul_q10, d.rul_q90, color=C_PROP, alpha=0.18, label="80 % intervalas")
ax[0].plot(d.cycle, d.rul_cycles, color=C_TRUE, lw=1.4, label="tikrasis RUL")
ax[0].plot(d.cycle, d.rul_baseline, color=C_BASE, lw=1, ls="--", label="Baseline")
ax[0].plot(d.cycle, d.rul_xgb, color=C_PROP, lw=1, label="Hibridas (LR + XGBoost)")
ax[0].set_xlabel("Ciklai"); ax[0].set_ylabel("RUL, ciklai")
ax[0].set_title(f"RUL prognozė, testo akumuliatorius #{b}"); ax[0].legend(fontsize=7)
mb = M["rul_mae_by_soh"]
k = list(mb)
x = np.arange(len(k))
ax[1].bar(x - 0.27, [mb[i]["MAE_baseline"] for i in k], 0.27, color=C_BASE, label="Baseline")
ax[1].bar(x, [mb[i]["MAE_xgb_pure"] for i in k], 0.27, color="#9ec5e8", label="Grynas XGBoost")
ax[1].bar(x + 0.27, [mb[i]["MAE_xgb"] for i in k], 0.27, color=C_PROP, label="Hibridas")
ax[1].set_xticks(x); ax[1].set_xticklabels(k); ax[1].set_xlabel("Tikroji SOH")
ax[1].set_ylabel("RUL MAE, ciklai"); ax[1].set_title("RUL paklaida pagal būklę"); ax[1].legend()
save(fig, "fig7_rul.png")

# 8 pav. Konfuzijos matricos
fig, ax = plt.subplots(1, 2, figsize=(7, 3))
for a, (name, m) in zip(ax, M["replace_soon"].items()):
    cm = np.array(m["confusion"])
    a.imshow(cm, cmap="Blues"); a.grid(False)
    for i in range(2):
        for j in range(2):
            a.text(j, i, cm[i, j], ha="center", va="center",
                   color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=10)
    a.set_xticks([0, 1]); a.set_xticklabels(["Tinkamas", "Keisti"])
    a.set_yticks([0, 1]); a.set_yticklabels(["Tinkamas", "Keisti"])
    a.set_xlabel("Prognozė"); a.set_ylabel("Tikrovė")
    a.set_title(f"{name.split(':')[0]}\nF1 = {m['F1']:.2f}", fontsize=9)
save(fig, "fig8_konfuzijos_matricos.png")

# 9 pav. Sistemos architektūra (blokinė schema)
fig, ax = plt.subplots(figsize=(9, 3.4)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(-0.1, 3.5)
W_, H_ = 1.75, 0.9
boxes = {"src": (0.1, 2.4, "OBD-II adapteris /\nmultimetras", "#f2f2f2"),
         "raw": (2.1, 2.4, "Neapdoroti matavimai\n(OCV, V_min, T, t_įkr)", "#f2f2f2"),
         "cln": (4.1, 2.4, "Valymas\n(ribos, interpoliacija)", "#e3eef9"),
         "fea": (6.1, 2.4, "Požymiai\n(ΔV₂₅, tendencijos,\nšil. stresas)", "#e3eef9"),
         "soh": (8.1, 2.4, "SOH modelis\n(XGBoost)", "#cfe2f3"),
         "rul": (8.1, 1.2, "RUL modelis\n(XGBoost + kvantiliai)", "#cfe2f3"),
         "shp": (6.1, 1.2, "SHAP\npaaiškinimas", "#fdf2e9"),
         "ui":  (6.1, 0.0, "Streamlit UI: SOH, RUL,\nrekomendacija vairuotojui", "#fdf2e9"),
         "hum": (8.1, 0.0, "Sprendimą priima\nžmogus (keisti / ne)", "#eaf5e4")}
for x0, y0, t, col in boxes.values():
    ax.add_patch(FancyBboxPatch((x0, y0), W_, H_, boxstyle="round,pad=0.03", fc=col, ec="#555", lw=0.8))
    ax.text(x0 + W_ / 2, y0 + H_ / 2, t, ha="center", va="center", fontsize=7.5)
def R(k): x, y = boxes[k][:2]; return (x + W_, y + H_ / 2)
def L(k): x, y = boxes[k][:2]; return (x, y + H_ / 2)
def B(k): x, y = boxes[k][:2]; return (x + W_ / 2, y)
def T(k): x, y = boxes[k][:2]; return (x + W_ / 2, y + H_)
arr = dict(arrowstyle="->", color="#444", lw=1)
for a_, b_ in [(R("src"), L("raw")), (R("raw"), L("cln")), (R("cln"), L("fea")), (R("fea"), L("soh")),
               ((7.85, 2.5), L("rul")), (B("soh"), T("shp")), (B("shp"), T("ui")),
               (B("rul"), (7.6, 0.93)), (R("ui"), L("hum"))]:
    ax.annotate("", b_, a_, arrowprops=arr)
ax.set_title("Sistemos architektūra ir duomenų srautas", fontsize=10)
save(fig, "fig9_architektura.png")

print("Paveikslai išsaugoti:", sorted(p.name for p in F.glob("*.png")))

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
