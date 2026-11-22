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
