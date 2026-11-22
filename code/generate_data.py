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
