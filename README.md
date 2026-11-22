# Akumuliatoriaus būklės (SOH) vertinimas be specialios įrangos

Individualus AI projektas, 1 tema. Regresiniu modeliu įvertinama 12 V automobilio
akumuliatoriaus sveikatos būklė (SOH) ir prognozuojamas likęs tarnavimo laikas (RUL).
Naudojami tik matavimai, kuriuos galima gauti pigiu OBD-II adapteriu arba multimetru:
ramybės įtampa, įtampos kritimas užvedant, temperatūra, įkrovimo laikas ir ciklų skaičius.

## Paleidimas

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python code/generate_data.py      # 1. sintetiniai duomenys -> data/battery_raw.csv
python code/model.py              # 2. valymas, požymiai, eksperimentai -> data/metrics.json
python code/make_figures.py       # 3. paveikslai -> figures/
streamlit run code/app.py         # 4. demonstracinė programa (naršyklėje)
```

Visos komandos leidžiamos iš projekto šakninio katalogo. Atsitiktinumo sėkla yra fiksuota (42),
todėl rezultatai atkartojami. Visas ciklas užtrunka apie 1 min. įprastu nešiojamuoju kompiuteriu,
GPU nereikia.

## Struktūra

| Kelias | Turinys |
|---|---|
| `code/generate_data.py` | Fizikinėmis prielaidomis pagrįstas 80 akumuliatorių senėjimo simuliatorius |
| `code/model.py` | Valymas, požymių inžinerija, Baseline ir Proposed palyginimas, RUL, SHAP |
| `code/make_figures.py` | 9 ataskaitos paveikslai |
| `code/app.py` | Streamlit sąsaja (CSV žurnalas arba vienas matavimas) |
| `data/` | Sugeneruoti duomenys, metrikos (`metrics.json`), modeliai (`models/`), `demo_log.csv` |
| `figures/` | PNG paveikslai |

## Svarbu

* **Duomenys sintetiniai.** Realūs NASA PCoE ir CALCE rinkiniai yra Li-ion elementų, o ne
  12 V švino-rūgšties akumuliatorių. Ataskaitos IV skyriuje pagrįsta, kodėl naudojamas simuliatorius
  ir kaip pereiti prie realių duomenų: pakanka pateikti CSV su tais pačiais stulpeliais, o `model.py`
  keisti nereikia.
* **Sprendimas informacinis.** Programa tik pataria ir nevaldo jokio įrenginio.

## Mano realūs duomenys (CSV formatas)

`cycle, temp_c, ocv_v, v_crank_min_v, charge_time_min, short_trip_share`.
Pavyzdys pateiktas faile `data/demo_log.csv`.
