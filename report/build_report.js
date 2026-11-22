// Ataskaitos generavimas: node report/build_report.js  (iš projekto šakninio katalogo)
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, WidthType, ShadingType, BorderStyle, LevelFormat,
  TableOfContents, Footer, PageNumber, PageBreak,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const M = JSON.parse(fs.readFileSync(path.join(ROOT, "data/metrics.json"), "utf8"));
const FONT = "Arial";
const CW = 9638; // turinio plotis DXA (A4, kairė 3 cm, dešinė 1 cm)

// ---------- pagalbinės funkcijos ----------
const n = (x, d = 3) => Number(x).toFixed(d).replace(".", ",");
const pct = (x, d = 1) => (x * 100).toFixed(d).replace(".", ",") + " %";

function runs(text, opts = {}) {
  // **paryškinta** ir _kursyvas_ žymėjimas
  const out = [];
  const re = /(\*\*[^*]+\*\*|~[^~]+~)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...opts }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...opts }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...opts }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...opts }));
  return out;
}
const P = (t, o = {}) => new Paragraph({ children: runs(t), alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, line: 300 }, ...o });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const B = (t, lvl = 0) => new Paragraph({ numbering: { reference: "bul", level: lvl }, children: runs(t), spacing: { after: 60, line: 288 }, alignment: AlignmentType.JUSTIFIED });
const NUM = (t, ref = "num") => new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(t), spacing: { after: 60, line: 288 }, alignment: AlignmentType.JUSTIFIED });
const CODE = (t) => new Paragraph({ children: [new TextRun({ text: t, font: "Consolas", size: 18 })], spacing: { after: 0 }, shading: { type: ShadingType.CLEAR, fill: "F3F3F3", color: "auto" } });

const border = { style: BorderStyle.SINGLE, size: 4, color: "999999" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(headers, rows, widths, opts = {}) {
  const sum = widths.reduce((a, b) => a + b, 0);
  const W = widths.map((w) => Math.round((w / sum) * CW));
  W[W.length - 1] += CW - W.reduce((a, b) => a + b, 0);
  const cell = (txt, i, head, hl) => new TableCell({
    width: { size: W[i], type: WidthType.DXA }, borders,
    shading: head ? { type: ShadingType.CLEAR, fill: "1F4E79", color: "auto" } : hl ? { type: ShadingType.CLEAR, fill: "E3EEF9", color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: runs(String(txt), head ? { bold: true, color: "FFFFFF", size: 18 } : { size: 18 }), spacing: { after: 0 } })],
  });
  return new Table({
    width: { size: CW, type: WidthType.DXA }, columnWidths: W,
    rows: [
      new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r, ri) => new TableRow({ children: r.map((c, i) => cell(c, i, false, (opts.highlight || []).includes(ri))) })),
    ],
  });
}
const TCAP = (t) => new Paragraph({ children: runs(t, { size: 18, italics: true }), spacing: { before: 160, after: 80 } });
function pngSize(f) { const b = fs.readFileSync(f); return [b.readUInt32BE(16), b.readUInt32BE(20)]; }
function FIG(file, caption, w = 620) {
  const f = path.join(ROOT, "figures", file);
  const [pw, ph] = pngSize(f);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: fs.readFileSync(f), transformation: { width: w, height: Math.round((w * ph) / pw) } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: runs(caption, { size: 18, italics: true }) }),
  ];
}
const SP = () => new Paragraph({ children: [], spacing: { after: 60 } });

// ---------- metrikos ----------
const S = M.soh, R = M.rul, C = M.replace_soon;
const sB = S["Baseline: LinearRegression (tik ciklai)"], sX = S["Proposed: XGBoost (visi požymiai)"];
const sLA = S["LinearRegression (visi požymiai)"], sRF = S["RandomForest (visi požymiai)"], sNT = S["Abliacija: XGBoost be temperatūros"];
const rB = R["Baseline: LinearRegression (tik ciklai)"], rLA = R["LinearRegression (visi požymiai)"];
const rXP = R["1-as bandymas: XGBoost (grynas)"], rH = R["Proposed: LR + XGBoost liekanos (hibridas)"];
const cB = C["Baseline: LinearRegression (tik ciklai)"], cX = C["Proposed: XGBoost SOH + hibridinis RUL"];
const L = M.leakage_random_split_rul, SM = M.single_measurement_soh;
const cvB = M.soh_cv_rmse.baseline, cvX = M.soh_cv_rmse.xgb;
const cl = M.cleaning;

// =================================================================== TURINYS
const titlePage = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 600, after: 120 }, children: [new TextRun({ text: "[UNIVERSITETO / KOLEGIJOS PAVADINIMAS]", size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 1400 }, children: [new TextRun({ text: "[Fakultetas / katedra]", size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [new TextRun({ text: "INDIVIDUALUS SEMESTRO PROJEKTAS — 1 TEMA", size: 24, bold: true, color: "1F4E79" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: [new TextRun({ text: "Akumuliatoriaus būklės (SOH) vertinimas be specialios įrangos", size: 36, bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 1400 }, children: [new TextRun({ text: "Regresinis SOH ir likusio tarnavimo laiko (RUL) prognozės modelis pagal įkrovimo ciklų, įtampos kritimo ir temperatūros duomenis", size: 24, italics: true })] }),
  new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 80 }, indent: { left: 4800 }, children: runs("**Autorius:** [Vardas Pavardė], [grupė]") }),
  new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 80 }, indent: { left: 4800 }, children: runs("**Dalykas:** Dirbtinio intelekto metodai") }),
  new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 80 }, indent: { left: 4800 }, children: runs("**Dėstytojas:** [Vardas Pavardė]") }),
  new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 1000 }, indent: { left: 4800 }, children: runs("**Pateikimo data:** [data]") }),
  TCAP("Darbo pasidalijimo lentelė (individualus darbas)"),
  table(["Atlikėjas", "Atsakomybė", "Skyriai"],
    [["[Vardas Pavardė]", "Visas darbas: duomenų simuliatorius, modeliai, eksperimentai, demonstracinė programa, ataskaita", "I–X"]],
    [2, 5, 1.2]),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 800 }, children: [new TextRun({ text: "[Miestas], 2026", size: 22 })] }),
  new Paragraph({ children: [new PageBreak()] }),
  new Paragraph({ children: [new TextRun({ text: "Turinys", bold: true, size: 28, font: FONT })], spacing: { after: 200 } }),
  new TableOfContents("Turinys", { hyperlink: true, headingStyleRange: "1-2" }),
  new Paragraph({ children: [new PageBreak()] }),
];

const summary = [
  H1("II. Santrauka"),
  P("Automobilio 12 V akumuliatorius dažniausiai „netikėtai“ sugenda šaltą žiemos rytą. Tačiau jo būklė blogėja palaipsniui, ir tą galima pastebėti iš anksto. Profesionalūs talpos testeriai ir apkrovos šakutės vairuotojams nepasiekiami. Šiame darbe tiriama, ar akumuliatoriaus sveikatos būklę (SOH, ~State of Health~) ir likusį tarnavimo laiką (RUL, ~Remaining Useful Life~) galima įvertinti tik iš duomenų, kuriuos nuskaito pigus OBD-II adapteris (~10 €) ar paprastas multimetras. Tai ramybės įtampa, įtampos kritimas užvedant variklį, temperatūra, įkrovimo laikas ir ciklų skaičius."),
  P(`Sukurtas visas AI procesas: fizikinėmis prielaidomis pagrįstas 80 švino-rūgšties akumuliatorių senėjimo simuliatorius (${M.dataset.rows.toLocaleString("lt-LT")} matavimų), duomenų valymas, temperatūrinė įtampos kritimo kompensacija, požymių inžinerija ir dvi užduotys. Pirmoji – SOH regresija. Antroji – RUL regresija su 80 % prognozės intervalu. Modeliai vertinami tik su akumuliatoriais, kurių modelis mokymo metu nematė.`),
  P(`**Pagrindiniai rezultatai.** SOH: bazinis modelis (tiesinė regresija tik pagal ciklų skaičių) pasiekė RMSE = ${n(sB.RMSE, 4)} (R² = ${n(sB.R2, 2)}), o siūlomas XGBoost modelis – RMSE = ${n(sX.RMSE, 4)} (R² = ${n(sX.R2, 3)}), t. y. paklaida sumažėjo ${Math.round((1 - sX.RMSE / sB.RMSE) * 100)} %. RUL: bazinio modelio R² = ${n(rB.R2, 2)}, galutinio hibridinio modelio (tiesinė tendencija + XGBoost liekanų korekcija) R² = ${n(rH.R2, 2)}, vidutinė paklaida – apie ${n(M.rul_months_MAE.xgb, 1)} mėn. (bazinio – ${n(M.rul_months_MAE.baseline, 1)} mėn.). Praktinis sprendimas „keisti akumuliatorių artimiausiu metu (≤ 200 ciklų)“ pasiekė F1 = ${n(cX.F1, 2)} (bazinis – ${n(cB.F1, 2)}).`),
  P(`Darbe atvirai aprašytos trys nesėkmės. Pirma, atsitiktinis eilučių skaidymas dirbtinai padidino RUL R² iki ${n(L.R2, 2)}, nes į testą „nutekėjo“ to paties akumuliatoriaus duomenys. Antra, grynas XGBoost RUL užduotyje pralaimėjo paprastai tiesinei regresijai, todėl buvo sukurtas hibridinis modelis. Trečia, tiesinė temperatūrinė kompensacija sezoninį svyravimą pašalina tik iš dalies. Duomenys yra sintetiniai; tai svarbiausias ribotumas (IV ir VIII skyriai).`),
];

const problem = [
  H1("III. Problemos analizė"),
  H2("3.1 Kodėl tai aktualu"),
  P("Išsikrovęs ar susidėvėjęs starterio akumuliatorius yra viena dažniausių priežasčių, dėl kurių automobilis neužsiveda ir kviečiama kelių pagalba, ypač šaltuoju metų laiku. Lietuvos klimate problema dvejopa. Vasaros karštis po kapotu greitina korozijos ir sulfatacijos procesus, o žiemą šaltis padidina vidinę varžą ir variklio užvedimui reikalingą srovę. Todėl susidėvėjimas dažniausiai „atsiskleidžia“ pirmąjį šaltą rytą. Įprastas sprendimas yra keisti akumuliatorių „pagal amžių“ (kas 4–5 metus), bet tada keičiami ir dar tinkami akumuliatoriai (tai papildomos išlaidos ir atliekos), o anksčiau sugedę lieka nepastebėti."),
  P("Automobilio diagnostikos lizdas (OBD-II) leidžia telefonu nuolat stebėti borto įtampą, o kai kurios programėlės rodo ir aplinkos ar įsiurbiamo oro temperatūrą. Šiais duomenimis paremtas AI modelis galėtų perspėti vairuotoją prieš gedimą, be jokios specialios įrangos."),
  H2("3.2 Vartotojai ir suinteresuotosios šalys"),
  B("**Vairuotojai** – nori žinoti, ar akumuliatorius „ištvers žiemą“, ir nekeisti jo be reikalo."),
  B("**Autoservisai** – greita išankstinė diagnostika be talpos testo, pagrįsta rekomendacija klientui."),
  B("**Automobilių parkų valdytojai** (taksi, kurjeriai, nuoma) – planinė priežiūra, mažiau prastovų."),
  B("**Aplinkosauga** – mažiau per anksti išmestų švino-rūgšties akumuliatorių."),
  H2("3.3 Funkciniai ir nefunkciniai reikalavimai"),
  P("**Funkciniai:** (F1) priimti matavimų žurnalą (CSV) arba vieną rankinį matavimą; (F2) išvalyti duomenis (fiziškai neįmanomos reikšmės, praleidimai); (F3) įvertinti SOH; (F4) prognozuoti RUL ciklais su 80 % intervalu; (F5) pateikti aiškią rekomendaciją (žalia / oranžinė / raudona); (F6) paaiškinti, kurie požymiai lėmė įvertinimą (SHAP)."),
  P("**Nefunkciniai:** (N1) jokios specialios įrangos, tik OBD-II adapteris arba multimetras; (N2) be uždarų komercinių API, viskas veikia neprisijungus; (N3) atkartojamumas (fiksuota sėkla 42, requirements.txt); (N4) prognozė trunka mažiau nei sekundę įprastu kompiuteriu; (N5) sprendimas tik informacinis."),
  H2("3.4 Privalomas pradinis klasifikavimas"),
  TCAP("1 lentelė. Projekto klasifikavimas pagal metodinių nurodymų 1 skyrių"),
  table(["Kriterijus", "Pasirinktas atsakymas", "Poveikis metodikai"], [
    ["Duomenų pobūdis", "Skaitiniai signalai (įtampa, temperatūra, laikas, ciklai)", "Lentelinių duomenų bibliotekos: pandas, scikit-learn, XGBoost; NLP ir CV įrankiai nereikalingi."],
    ["Duomenų šaltinis", "Sintetiniai (fizikinis simuliatorius); realių duomenų kelias – NASA / CALCE / savas OBD žurnalas", "Asmens duomenų nėra. Sintetinių duomenų validumas pagrįstas 4.2 poskyryje, ribotumas – VIII skyriuje."],
    ["Realaus laiko reikalavimas", "Ne (paketinė analizė, matavimas kartą per kelias dienas)", "Lengvų modelių ar ONNX/TFLite nereikia; tinka medžių ansambliai."],
    ["Aparatūros poreikis", "Nėra (grynai programinis); duomenis gali teikti OBD-II adapteris", "Atskiro techninės integracijos skyriaus nereikia; matavimo būdai aprašyti 4.3 poskyryje."],
    ["Sprendimo kritiškumas", "Informacinis (pataria)", "Modelis nevaldo jokio įrenginio, o sprendimą keisti akumuliatorių priima žmogus."],
  ], [1.6, 2.6, 3.6]),
  H2("3.5 Etika, sauga ir privatumas"),
  B("**Asmens duomenys.** Modeliui reikalingi tik techniniai dydžiai (įtampa, temperatūra, laikas). Realiai naudojant, OBD žurnale gali atsirasti GPS koordinačių ar VIN numeris. Programa jų neskaito, o README nurodo juos pašalinti prieš įkeliant žurnalą. Duomenys apdorojami vietoje ir niekur nesiunčiami."),
  B("**Sauga.** Klaidinga „žalia“ prognozė gali palikti vairuotoją su neužsivedančiu automobiliu, todėl (1) rodomas ne tik taškinis įvertis, bet ir 80 % RUL intervalas; (2) rekomendacijos riba parinkta atsargiai, pirmenybę teikiant jautrumui (Recall); (3) sąsajoje nurodyta, kad tai ne serviso diagnozė."),
  B("**Ekonominis šališkumas.** Per „raudona“ prognozė skatintų nereikalingą pirkimą. Tai stebima per Precision (7.3 poskyris) ir atvirai aptariama."),
  B("**Sąžiningumas dėl duomenų.** Visi skaičiai gauti iš sintetinių duomenų. Tai pabrėžiama kiekviename skyriuje, kad rezultatai nebūtų klaidingai suprasti kaip realaus akumuliatoriaus patikra."),
  B("**Mokinio indėlis.** Nenaudojamas joks uždaras komercinis AI API; visi modeliai mokomi vietoje atvirojo kodo bibliotekomis."),
  H2("3.6 Vartotojo patirtis (UX)"),
  P("Sąsaja kurta pagal principą „vienas skaičius ir viena spalva“: vairuotojui svarbiausia ne RMSE, o atsakymas į klausimą „ar ištvers žiemą?“. Todėl Streamlit programa rodo SOH procentais, RUL ciklais su intervalu ir spalvinę rekomendaciją. Prieš gynimą rekomenduojama atlikti 2–3 trumpus pokalbius su vairuotojais ar autoserviso meistru ir jų įžvalgas įrašyti čia. Klausimai: kaip dažnai jie norėtų gauti įspėjimą; ar suprantamas „ciklų“ matas, ar geriau mėnesiai; ar pasitikėtų rekomendacija be serviso patikros. ~[Užpildyti po pokalbių.]~"),
];

const data = [
  H1("IV. Duomenų rinkimas"),
  H2("4.1 Duomenų šaltiniai"),
  TCAP("2 lentelė. Galimi atviri duomenų šaltiniai ir jų tinkamumas"),
  table(["Šaltinis", "Turinys", "Tinkamumas šiam darbui"], [
    ["NASA PCoE Battery Data Set (Saha, Goebel, 2007)", "18650 Li-ion elementų įkrovimo / iškrovimo / impedanso ciklai iki 30 % talpos praradimo", "Kita chemija (Li-ion), tik keli elementai, laboratorinės sąlygos; tinka metodikai patikrinti"],
    ["CALCE (Merilendo universitetas)", "Li-ion elementų ciklų duomenys", "Kita chemija, laboratorinės sąlygos"],
    ["Severson ir kt. (2019), Nature Energy", "124 LFP elementai, greitas įkrovimas", "Kita chemija, puikus RUL metodikos etalonas"],
    ["Savas OBD-II žurnalas", "Realaus automobilio borto įtampa ir temperatūra", "Idealiausia, bet tam reikia daugelio metų stebėjimo iki gedimo"],
    ["**Naudota: fizikinis simuliatorius**", "80 švino-rūgšties 12 V akumuliatorių, matavimai kas 10 ciklų", "Atitinka temą (automobilio akumuliatorius) ir „be specialios įrangos“ matavimus"],
  ], [2.2, 3, 3]),
  H2("4.2 Kodėl naudojami sintetiniai duomenys ir kodėl jie validūs"),
  P("Metodiniai nurodymai reikalauja pagrįsti sintetinių duomenų validumą. Kodėl jie pasirinkti: (1) visi žinomi atviri senėjimo rinkiniai yra Li-ion elementų, o temoje kalbama apie automobilio (švino-rūgšties) akumuliatorių; (2) juose matuojama laboratorine įranga (talpa, impedansas), o šio darbo esmė – vertinti ~be specialios įrangos~; (3) realiam žurnalui surinkti reikėtų 3–5 metų vieno akumuliatoriaus stebėjimo iki gedimo, o tai semestro projekte neįmanoma."),
  P("Kodėl jie validūs: simuliatoriuje įdiegti literatūroje aprašyti ir kokybiškai patikrinami dėsningumai (žr. 4.4), o gauti rezultatai atitinka realią praktiką. Tipinis tarnavimo laikas yra apie " + n(37.3, 0) + " mėn. (mediana), o išsibarstymas – nuo ~14 mėn. iki daugiau nei 10 metų. Tai atitinka įprastus 3–5 metus su ilgomis „uodegomis“. Karštame klimate akumuliatoriai sensta greičiau (1 pav.), o senėjimas pabaigoje greitėja („kelio“ efektas). Be to, į duomenis tyčia įterpti jutiklių triukšmas, praleistos reikšmės ir neįmanomi šuoliai, kad valymo etapas būtų realus. Svarbiausia: modelio kodui (model.py) tikri duomenys nieko nekeistų – užtenka CSV failo su tais pačiais stulpeliais."),
  H2("4.3 Matuojami dydžiai be specialios įrangos"),
  TCAP("3 lentelė. Modelio įvesties dydžiai ir jų gavimo būdas"),
  table(["Dydis", "Kaip išmatuoti", "Kodėl informatyvus"], [
    ["Ramybės įtampa (OCV), V", "Multimetru arba OBD rytą, prieš užvedant (≥ 6 h stovėjimo)", "Atspindi įkrovos lygį ir sulfataciją"],
    ["Min. įtampa užvedant, V → įtampos kritimas ΔV", "OBD programėlė (dažnis ≥ 10 Hz) arba multimetras su MIN funkcija", "ΔV ≈ I_užv · R_vid; vidinė varža auga senstant"],
    ["Temperatūra, °C", "OBD (įsiurbiamo oro / aplinkos temp.) arba orų duomenys", "Šaltyje R ir užvedimo srovė didesni; karštis greitina senėjimą"],
    ["Įkrovimo laikas, min", "OBD: laikas, kol po užvedimo įtampa pasiekia 14,2 V", "Mažesnės talpos akumuliatorius greičiau pasiekia įkrovimo įtampą"],
    ["Ciklų skaičius", "Užvedimų skaitiklis programėlėje", "Bendras nusidėvėjimo matas"],
    ["Trumpų kelionių dalis", "Programėlės kelionių žurnalas (< 15 min)", "Nepilnas įkrovimas → sulfatacija"],
  ], [2.2, 3.2, 3]),
  H2("4.4 Simuliatoriaus fizikinės prielaidos"),
  B("Talpos nykimo greitis vienam ciklui dvigubėja kas +10 °C (Arenijaus tipo taisyklė; po kapotu laikoma +6 °C virš aplinkos)."),
  B("Trumpos kelionės (nepilnas įkrovimas) didina nykimo greitį iki ~2,3 karto (sulfatacija; Ruetschi, 2004)."),
  B("Kai SOH < 0,86, senėjimas greitėja iki 5 kartų („kelio“ efektas)."),
  B("Vidinė varža: R = R₀·(1 + 4·(1−SOH)^1,3)·e^(0,025·(25−T)); užvedimo srovė didėja šaltyje 1,2 %/°C; ΔV = I·R + triukšmas."),
  B("Kiekvienas akumuliatorius turi gamybinę sklaidą (log-normalus daugiklis σ = 0,18) – tai realistiškai apsunkina prognozę."),
  B("Klimatas: vidutinė metinė temperatūra nuo −1 iki 17 °C, sezoninė amplitudė 8–14 °C, kasdienis triukšmas σ = 3 °C."),
  H2("4.5 Apimtis ir struktūra"),
  P(`Iš viso ${M.dataset.batteries} akumuliatoriai ir ${M.dataset.rows.toLocaleString("lt-LT")} matavimų (kas 10 ciklų, kol SOH nukrenta iki 0,70). Gyvavimo pabaiga (EOL) apibrėžta kaip SOH = 0,80 (pramonėje įprasta riba); ją akumuliatoriai pasiekia po 810–5 380 ciklų (mediana – 2 100). Taikinio kintamieji: **soh_true** (C/C_nom) ir **rul_cycles** (ciklai iki EOL).`),
  H2("4.6 Valymas"),
  TCAP("4 lentelė. Duomenų valymo rezultatai"),
  table(["Stulpelis", "Praleista reikšmių", "Fiziškai neįmanomų (pašalinta)", "Taisyklė"], [
    ["temp_c", cl.missing_before.temp_c, cl.outliers_removed.temp_c, "−40…60 °C"],
    ["charge_time_min", cl.missing_before.charge_time_min, cl.outliers_removed.charge_time_min, "0,2…60 min"],
    ["ocv_v", cl.missing_before.ocv_v, cl.outliers_removed.ocv_v, "10,5…13,3 V"],
    ["v_crank_min_v", cl.missing_before.v_crank_min_v, cl.outliers_removed.v_crank_min_v, "5,0…13,0 V"],
  ], [2, 1.6, 2, 2]),
  P(`Fiziškai neįmanomos reikšmės (pvz., 0 V ar 16,5 V – atsijungęs ar sutrikęs jutiklis) pakeičiamos praleistomis, o tada visos praleistos reikšmės tiesiškai interpoliuojamos ~kiekvieno akumuliatoriaus laiko eilutėje atskirai~, kad nebūtų „pasiskolinta“ informacija iš kito akumuliatoriaus. Po valymo praleistų reikšmių liko ${cl.missing_after}. Eilutės nešalinamos, nes būtų sugadinti slenkantys požymiai.`),
  H2("4.7 Vizualizacija"),
  ...FIG("fig1_soh_trajektorijos.png", "1 pav. Visų 80 akumuliatorių SOH trajektorijos (spalva – klimato vidutinė temperatūra). Šiltame klimate akumuliatoriai sensta greičiau, o pabaigoje matomas „kelio“ efektas."),
  ...FIG("fig2_valymas_kompensacija.png", "2 pav. Kairėje: neapdoroti ir išvalyti minimalios užvedimo įtampos duomenys (raudoni taškai – jutiklio šuoliai). Dešinėje: išmatuotas įtampos kritimas ir temperatūriškai kompensuotas ΔV₂₅."),
  P("2 pav. dešinėje matyti, kad išmatuotas ΔV labai svyruoja pagal sezoną: žiemą kritimas didesnis vien dėl šalčio, o ne dėl akumuliatoriaus būklės. Kompensuotas ΔV₂₅ svyruoja mažiau ir aiškiau rodo ilgalaikį augimą. Vis dėlto svyravimas nepašalinamas visiškai (žr. 6.4 nesėkmę Nr. 3)."),
  ...FIG("fig3_histogramos_koreliacijos.png", "3 pav. Pagrindinių požymių histogramos (kairėje) ir Pirsono koreliacijų matrica (dešinėje)."),
  P("Koreliacijų matrica patvirtina požymių inžinerijos naudą: neapdoroto ΔV koreliacija su SOH tik −0,43, o kompensuoto ΔV₂₅ – jau −0,70. Stipriausiai su SOH susijęs sukauptas šiluminis stresas (−0,88) ir ciklų skaičius (−0,76). ΔV labai stipriai koreliuoja su temperatūra (−0,85), todėl temperatūros nekompensavus modelis klaidingai „matytų“ blogą būklę kiekvieną žiemą."),
  H2("4.8 Duomenų skaidymas"),
  P(`Skaidoma **pagal akumuliatorių** (GroupShuffleSplit), o ne pagal atsitiktines eilutes: mokymui ${M.split.train.batteries} akumuliatoriai (${M.split.train.rows.toLocaleString("lt-LT")} eil.), validavimui (ankstyvajam stabdymui) ${M.split.val.batteries} (${M.split.val.rows.toLocaleString("lt-LT")} eil.), testui ${M.split.test.batteries} (${M.split.test.rows.toLocaleString("lt-LT")} eil.). Taip imituojamas realus naudojimas, kai modelis vertina naują, dar nematytą akumuliatorių. Kodėl tai būtina, parodyta 6.4 poskyryje.`),
];

const arch = [
  H1("V. Sisteminė architektūra"),
  ...FIG("fig9_architektura.png", "4 pav. Sistemos blokinė schema ir duomenų srautas."),
  H2("5.1 Proceso etapai"),
  NUM("**generate_data.py** – simuliuoja 80 akumuliatorių ir įterpia jutiklių defektus → data/battery_raw.csv."),
  NUM("**model.py** – valymas (clean), požymiai (build_features), skaidymas pagal akumuliatorių, visi eksperimentai, SHAP, modelių išsaugojimas → data/metrics.json, data/models/."),
  NUM("**make_figures.py** – 9 ataskaitos paveikslai → figures/."),
  NUM("**app.py** – Streamlit sąsaja: įkeliamas CSV žurnalas arba įvedamas vienas matavimas; rodomi SOH, RUL su intervalu ir rekomendacija."),
  H2("5.2 Požymiai"),
  TCAP("5 lentelė. Modelio požymiai (12)"),
  table(["Požymis", "Aprašymas"], [
    ["cycle, temp_c, ocv_v, charge_time_min, short_trip_share", "Tiesioginiai matavimai"],
    ["dv", "Įtampos kritimas ΔV = OCV − V_min"],
    ["dv_25", "ΔV perskaičiuotas į 25 °C: ΔV / (1 + 0,03·(25 − T)); koeficientas ~3 %/°C iš literatūros"],
    ["dv_25_roll, charge_roll", "Slenkantis vidurkis (8 matavimai) – mažina triukšmą"],
    ["dv_25_slope", "ΔV₂₅ tendencija (tiesinės regresijos nuolydis per 15 matavimų)"],
    ["temp_mean_cum", "Vidutinė temperatūra nuo eksploatacijos pradžios"],
    ["heat_stress_cum", "Sukauptas šiluminis stresas: Σ 2^((T+6−25)/10) · 10 ciklų"],
  ], [3, 5]),
  P("Visi slenkantys ir sukaupti požymiai skaičiuojami **tik iš praeities** reikšmių (be ateities informacijos), todėl tą patį skaičiavimą galima atlikti realiu laiku su vairuotojo žurnalu."),
  H2("5.3 Bibliotekos (requirements.txt)"),
  P("numpy, pandas, scikit-learn, xgboost, shap, matplotlib, streamlit. GPU, interneto ryšio ir uždarų API nereikia."),
];

const modelDev = [
  H1("VI. Modelio kūrimas"),
  H2("6.1 Architektūros pasirinkimo pagrindimas"),
  P(`Pagal metodinių nurodymų 3 skyriaus hierarchinę taisyklę: eilučių yra ${M.dataset.rows.toLocaleString("lt-LT")}, bet **nepriklausomų objektų** (akumuliatorių) tik 80, o vieno akumuliatoriaus gretimi matavimai labai panašūs. Todėl efektyviai tai yra vidutinio dydžio duomenų rinkinys, kuriam rekomenduojamas XGBoost / LightGBM. Gilieji sekų modeliai (LSTM) su 55 mokymo akumuliatoriais būtų linkę persimokyti, o realaus laiko reikalavimo nėra. Taip pat palyginti RandomForest ir tiesinė regresija su tais pačiais požymiais, kad būtų aišku, kiek naudos duoda pats modelis, o kiek – požymiai.`),
  P("**Baseline** (privalomas): LinearRegression, prognozuojanti SOH ir RUL ~tik pagal ciklų skaičių~. Tai „vidutinio akumuliatoriaus“ metodas, kurį iš esmės taiko ir „keisk kas N metų“ taisyklė. **Proposed:** XGBoost visiems požymiams (SOH) ir hibridinis modelis „tiesinė tendencija + XGBoost liekanų korekcija“ (RUL; kodėl jis atsirado – 6.4)."),
  H2("6.2 Hiperparametrai"),
  TCAP("6 lentelė. Galutiniai hiperparametrai"),
  table(["Modelis", "Pagrindiniai hiperparametrai"], [
    ["Baseline – LinearRegression", "scikit-learn numatytieji (mažiausių kvadratų metodas)"],
    ["RandomForest (palyginimui)", "n_estimators = 300, max_depth = 12, min_samples_leaf = 5"],
    ["XGBoost SOH", `learning_rate = 0,03, max_depth = 5, min_child_weight = 5, subsample = 0,8, colsample_bytree = 0,8, early stopping (100 iter.), geriausia iteracija – ${M.soh_best_iteration}`],
    ["Hibridas RUL", "LinearRegression (visi požymiai) + XGBoost liekanoms: max_depth = 3, learning_rate = 0,02, early stopping"],
    ["RUL intervalas", "XGBoost kvantilinė regresija (reg:quantileerror), α = 0,1 ir 0,9"],
  ], [2.2, 5.8]),
  H2("6.3 Mokymo kreivė"),
  ...FIG("fig4_mokymo_kreive.png", "5 pav. XGBoost SOH modelio mokymo ir validavimo RMSE pagal iteracijas.", 520),
  P(`Validavimo paklaida nustoja mažėti ties ~${M.soh_best_iteration} iteracija, o mokymo paklaida mažėja toliau – tai persimokymo pradžia. Ankstyvasis stabdymas (early stopping) išsaugo geriausią iteraciją. Tarpas tarp kreivių rodo, kad nematyto akumuliatoriaus prognozė iš esmės sunkesnė už pažįstamo – tai normalu, kai objektai turi individualių savybių (gamybinė sklaida).`),
  H2("6.4 Nesėkmės ir sprendimų keitimas"),
  P(`**Nesėkmė Nr. 1 – duomenų nutekėjimas.** Pirmasis bandymas: atsitiktinis eilučių skaidymas (train_test_split). RUL modelis pasiekė R² = ${n(L.R2, 3)} ir RMSE = ${Math.round(L.RMSE)} ciklų – įtartinai gerai. Priežastis: to paties akumuliatoriaus gretimi matavimai pateko ir į mokymą, ir į testą, todėl modelis „atpažino“ akumuliatorių, o ne įvertino jo būklę. Skaidant pagal akumuliatorių, tas pats modelis pasiekė R² = ${n(rXP.R2, 2)}. Išvada: visi galutiniai rezultatai skaičiuojami tik su nematytais akumuliatoriais.`),
  P(`**Nesėkmė Nr. 2 – „pažangesnis“ modelis pralaimėjo paprastesniam.** Grynas XGBoost RUL užduotyje pasiekė R² = ${n(rXP.R2, 2)}, o paprasta tiesinė regresija su tais pačiais požymiais – R² = ${n(rLA.R2, 2)}. Medžių modelis prognozę skaido į „laiptelius“ ir prastai perteikia beveik tiesinę RUL priklausomybę nuo sukaupto streso ir ciklų, ypač ilgaamžiams akumuliatoriams. Sprendimas – **hibridas**: tiesinė regresija pagauna pagrindinę tendenciją, o negilus XGBoost (max_depth = 3) mokosi tik jos liekanų (netiesinių pataisų, pvz., „kelio“ efekto). Rezultatas – R² = ${n(rH.R2, 2)}, geriausias iš visų.`),
  P(`**Nesėkmė Nr. 3 – nepilna temperatūrinė kompensacija.** Tiesinė formulė (3 %/°C) sumažino ΔV sezoninį svyravimą, bet jo nepašalino (2 pav.), nes tikroji priklausomybė yra eksponentinė, o šaltyje dar didėja ir užvedimo srovė. Tikslesnei formulei reikėtų žinoti konkretaus akumuliatoriaus charakteristikas, todėl palikta paprasta kompensacija, o likusią dalį modelis išmoksta iš temp_c požymio (abliacija – 7.4).`),
  H2("6.5 Paaiškinamumas (XAI, SHAP)"),
  ...FIG("fig6_shap.png", "6 pav. SHAP reikšmės XGBoost SOH modeliui (testo imtis). Kiekvienas taškas – vienas matavimas; spalva – požymio reikšmė.", 560),
  P(`Svarbiausias požymis – sukauptas šiluminis stresas (vid. |SHAP| = ${n(M.shap_mean_abs_soh.heat_stress_cum, 4)}): kuo daugiau „karštų“ ciklų, tuo mažesnė SOH. Toliau eina ramybės įtampa, trumpų kelionių dalis, ciklų skaičius ir įkrovimo laikas. Kryptys fiziškai prasmingos: didelė OCV → geresnė būklė; daug trumpų kelionių → blogesnė; ilgesnis įkrovimo laikas → didesnė likusi talpa. Įtampos kritimo požymiai (dv_25_roll, dv_25) svarbūs, bet mažiau nei tikėtasi. Iš dalies taip yra todėl, kad jų informacija sutampa su sukauptu stresu. Neapdorotas dv beveik nenaudojamas – modelis „pasirinko“ kompensuotą versiją, ir tai patvirtina požymių inžinerijos prasmę. Šis rezultatas turi ir praktinę pasekmę: modelis labiausiai remiasi ~istorija~, todėl ilgas žurnalas svarbus (žr. VIII skyrių).`),
];

const tSOH = [
  ["Baseline: LinearRegression (tik ciklai)", sB], ["LinearRegression (visi požymiai)", sLA],
  ["RandomForest (visi požymiai)", sRF], ["**Proposed: XGBoost (visi požymiai)**", sX], ["Abliacija: XGBoost be temperatūros", sNT],
].map(([k, v]) => [k, n(v.RMSE, 4), n(v.MAE, 4), n(v.R2, 3)]);
const tRUL = [
  ["Baseline: LinearRegression (tik ciklai)", rB], ["LinearRegression (visi požymiai)", rLA],
  ["1-as bandymas: XGBoost (grynas)", rXP], ["**Proposed: LR + XGBoost liekanos (hibridas)**", rH],
].map(([k, v]) => [k, Math.round(v.RMSE), Math.round(v.MAE), n(v.R2, 3)]);
const tT = Object.entries(M.soh_error_by_temp).map(([k, v]) => [k, v.n, n(v.MAE_baseline, 4), n(v.MAE_xgb_notemp, 4), n(v.MAE_xgb, 4)]);
const tRS = Object.entries(M.rul_mae_by_soh).map(([k, v]) => [k, v.n, Math.round(v.MAE_baseline), Math.round(v.MAE_xgb_pure), Math.round(v.MAE_xgb)]);

const exper = [
  H1("VII. Eksperimentai"),
  P(`Visos metrikos skaičiuotos testo imtyje – ${M.split.test.batteries} akumuliatoriuose, kurių modeliai mokymo metu nematė. SOH ir RUL yra regresijos užduotys, todėl pagrindinės metrikos yra RMSE, MAE ir R². Metodinių nurodymų F1 ≥ 0,70 reikalavimas vertinamas praktinei klasifikacijai „keisti / nekeisti“ (7.3).`),
  H2("7.1 SOH: Baseline vs Proposed"),
  TCAP("7 lentelė. SOH prognozės metrikos (testo imtis, 3 330 matavimų)"),
  table(["Modelis", "RMSE", "MAE", "R²"], tSOH, [4, 1.2, 1.2, 1.2], { highlight: [3] }),
  ...FIG("fig5_soh_prognoze_vs_tikroji.png", "7 pav. Prognozuota ir tikroji SOH: kairėje – bazinis modelis, dešinėje – XGBoost."),
  P(`Bazinis modelis visiems to paties amžiaus akumuliatoriams prognozuoja tą pačią SOH, todėl 7 pav. kairėje matomos horizontalios „juostos“: modelis nemato skirtumo tarp karštame klimate ir vėsiai eksploatuoto akumuliatoriaus. XGBoost paklaidą sumažina ${Math.round((1 - sX.RMSE / sB.RMSE) * 100)} % (RMSE ${n(sB.RMSE, 4)} → ${n(sX.RMSE, 4)}), t. y. vidutinė paklaida yra apie 1 SOH procentinį punktą. 5 blokų kryžminė patikra pagal akumuliatorius patvirtina stabilumą: RMSE baziniam modeliui ${n(cvB.mean, 4)} ± ${n(cvB.std, 4)}, XGBoost – ${n(cvX.mean, 4)} ± ${n(cvX.std, 4)}.`),
  P(`**Sąžiningas pastebėjimas:** tiesinė regresija su ~tais pačiais požymiais~ SOH užduotyje beveik prilygsta XGBoost (RMSE ${n(sLA.RMSE, 4)} ir ${n(sX.RMSE, 4)}), o RandomForest yra net prastesnis (${n(sRF.RMSE, 4)}). Vadinasi, didžiąją naudos dalį duoda **požymių inžinerija** (temperatūrinė kompensacija, sukauptas stresas), o ne modelio sudėtingumas. XGBoost paliktas kaip galutinis, nes jo MAE mažiausias (${n(sX.MAE, 4)}), jis netiesiškai sujungia požymius ir leidžia taikyti SHAP. Tačiau šis skirtumas nėra didelis, ir tai atvirai pripažįstama.`),
  H2("7.2 RUL: Baseline vs Proposed"),
  TCAP("8 lentelė. RUL prognozės metrikos ciklais (testo imtis, tik matavimai iki EOL)"),
  table(["Modelis", "RMSE, cikl.", "MAE, cikl.", "R²"], tRUL, [4, 1.2, 1.2, 1.2], { highlight: [3] }),
  P(`Perskaičiavus į laiką pagal kiekvieno automobilio užvedimų per dieną skaičių, vidutinė RUL paklaida sumažėja nuo ${n(M.rul_months_MAE.baseline, 1)} mėn. (baseline) iki ${n(M.rul_months_MAE.xgb, 1)} mėn. (hibridas). 80 % prognozės intervalas (kvantilinė regresija) iš tikrųjų apima ${pct(M.rul_interval_80_coverage)} tikrųjų reikšmių – tai artima nominaliems 80 %, todėl intervalas yra gerai kalibruotas. Tiesa, jis platus (vidutiniškai ~${Math.round(M.rul_interval_mean_width / 10) * 10} ciklų).`),
  ...FIG("fig7_rul.png", "8 pav. Kairėje: RUL prognozė tipiniam testo akumuliatoriui #12 su 80 % intervalu. Dešinėje: RUL MAE pagal tikrąją akumuliatoriaus būklę."),
  TCAP("9 lentelė. RUL MAE (ciklais) pagal akumuliatoriaus būklę"),
  table(["Tikroji SOH", "N", "Baseline", "Grynas XGBoost", "Hibridas"], tRS, [1.6, 1, 1.4, 1.6, 1.4]),
  P("Svarbi įžvalga: RUL prognozė kuo toliau, tuo tikslesnė. Naujam akumuliatoriui (SOH > 0,95) hibrido paklaida dar didelė (~550 ciklų), nes ateitis priklausys nuo dar nežinomų sąlygų (būsimos vasaros, gamybinė sklaida). Tačiau artėjant prie EOL (SOH 0,80–0,85) paklaida sumažėja iki ~170 ciklų, t. y. tada, kai prognozė vairuotojui svarbiausia. Bazinis modelis tokio pagerėjimo neturi."),
  H2("7.3 Praktinis sprendimas: „keisti artimiausiu metu“"),
  P(`Vairuotojui reikia aiškaus atsakymo, todėl RUL paverstas sprendimu: „keisti“, jei RUL ≤ 200 ciklų (~3 mėn. esant 2 užvedimams per dieną) arba SOH ≤ 0,80. Vertinamos visos ${M.split.test.rows.toLocaleString("lt-LT")} testo eilutės.`),
  TCAP("10 lentelė. Klasifikacijos „keisti / nekeisti“ metrikos"),
  table(["Modelis", "F1", "Precision", "Recall", "Accuracy"], [
    ["Baseline: LinearRegression (tik ciklai)", n(cB.F1, 3), n(cB.Precision, 3), n(cB.Recall, 3), n(cB.Accuracy, 3)],
    ["**Proposed: XGBoost SOH + hibridinis RUL**", n(cX.F1, 3), n(cX.Precision, 3), n(cX.Recall, 3), n(cX.Accuracy, 3)],
  ], [4, 1, 1.2, 1, 1.2], { highlight: [1] }),
  ...FIG("fig8_konfuzijos_matricos.png", "9 pav. Konfuzijos matricos: kairėje – baseline, dešinėje – siūlomas modelis.", 520),
  P(`Bazinio modelio Accuracy (${pct(cB.Accuracy)}) atrodo neblogai, bet tai apgaulinga: jis beveik niekada nesako „keisti“ (Recall = ${n(cB.Recall, 3)}) ir praleidžia ${cB.confusion[1][0]} iš ${cB.confusion[1][0] + cB.confusion[1][1]} atvejų, kai keisti reikėjo. Taip yra todėl, kad „vidutinis“ akumuliatorius pagal ciklus visada atrodo vidutiniškai sveikas. Siūlomas modelis atpažįsta ${pct(cX.Recall)} keistinų atvejų (F1 = ${n(cX.F1, 2)} > 0,70 reikalavimo). ${cX.confusion[0][1]} klaidingų „keisti“ pranešimų (Precision = ${n(cX.Precision, 2)}) yra sąmoningas kompromisas: saugiau anksčiau patikrinti akumuliatorių servise nei likti su neužsivedančiu automobiliu.`),
  H2("7.4 Abliacija: temperatūros įtaka"),
  TCAP("11 lentelė. SOH MAE pagal temperatūrą matavimo metu"),
  table(["Temperatūra", "N", "Baseline", "XGBoost be temp.", "XGBoost (visi)"], tT, [1.6, 1, 1.4, 1.8, 1.6]),
  P(`Pašalinus visą temperatūros informaciją (temp_c, kompensuotą ΔV, sukauptą stresą), XGBoost RMSE padidėja nuo ${n(sX.RMSE, 4)} iki ${n(sNT.RMSE, 4)} (+${Math.round((sNT.RMSE / sX.RMSE - 1) * 100)} %), o paklaida auga visuose temperatūrų intervaluose. Tai patvirtina, kad temperatūra yra būtinas kontekstas: be jo tas pats įtampos kritimas žiemą ir vasarą reiškia skirtingą būklę.`),
  H2("7.5 Sėkmės ir nesėkmės pavyzdžiai"),
  P(`**Sėkmė – akumuliatorius #12** (vidutinis klimatas, daug trumpų kelionių). Ties 1 500 ciklu tikroji SOH = 0,884, prognozė – 0,887. Tikrasis RUL = 750 ciklų, hibrido prognozė – 681 (intervalas 412–1 573), o bazinio modelio – 1 297 ciklai. Bazinis modelis būtų „pažadėjęs“ beveik dvigubai ilgesnį tarnavimą.`),
  P(`**Nesėkmė – akumuliatorius #33.** Vėsiame klimate (vid. 4 °C) jis pasiekė EOL jau po 2 020 ciklų – greičiau nei įprasta tokiam klimatui, nes turi didelę gamybinę sklaidą. Modelis, labiausiai pasikliaujantis sukauptu šiluminiu stresu, „tikėjosi“ sveiko akumuliatoriaus: didžiausia SOH paklaida testo imtyje (tikroji 0,759, prognozė 0,837 ties 2 230 ciklu), o RUL MAE šiam akumuliatoriui ~740 ciklų. Būtent tokiu atveju (gamybos defektas) vairuotojas labiausiai norėtų įspėjimo. Šis pavyzdys rodo, kad tiesioginiams įtampos matavimams reikėtų suteikti daugiau svorio (VIII skyrius).`),
  P(`Bendrai ${pct(0.837, 0)} testo matavimų SOH paklaida mažesnė nei 2 procentiniai punktai, o tik ${pct(0.05, 0)} – didesnė nei 3 p. p.`),
];

const limits = [
  H1("VIII. Ribotumai"),
  NUM("**Sintetiniai duomenys.** Svarbiausias ribotumas: rezultatai parodo metodo veikimą, bet ne realaus akumuliatoriaus diagnozės tikslumą. Simuliatoriaus prielaidos (Arenijaus taisyklė, varžos augimas) yra supaprastintos, o modelis iš dalies „išmoko“ simuliatoriaus formules. Realiems duomenims kodo keisti nereikia, bet reikėtų iš naujo apmokyti modelį ir, tikėtina, paklaidos būtų didesnės.", "num2"),
  NUM(`**Reikalinga istorija.** Modelis labiausiai remiasi sukauptais požymiais. Vertinant pagal vieną matavimą (istorija apytikriai atkuriama iš klimato vidurkio, kaip programos rankiniame režime), SOH RMSE padidėja nuo ${n(sX.RMSE, 4)} iki ${n(SM.RMSE, 4)}. Tai vis dar daug geriau už bazinį modelį, bet tiksliausiai veikia su ilgu žurnalu.`, "num2"),
  NUM("**Gamybos defektai.** Akumuliatoriai, senstantys greičiau nei „turėtų“ pagal sąlygas (#33 pavyzdys), prognozuojami blogiausiai. Galimas sprendimas – atskiras anomalijų detektorius, stebintis ΔV₂₅ tendenciją nepriklausomai nuo istorijos.", "num2"),
  NUM("**Ankstyvo gyvavimo RUL neapibrėžtis.** Naujam akumuliatoriui RUL paklaida ~550 ciklų (≈ 9 mėn.), o intervalas platus. Tai iš esmės neišvengiama, nes ateities sąlygos nežinomos.", "num2"),
  NUM("**Kiti akumuliatorių tipai.** Modeliuotas klasikinis švino-rūgšties (užlietas) akumuliatorius. AGM ir EFB (start-stop sistemos) bei Li-ion turi kitokius senėjimo mechanizmus ir reikalautų atskiro mokymo.", "num2"),
  NUM("**Matavimo kokybė.** Pigūs OBD adapteriai įtampą nuskaito ribotu dažniu (dažnai 1–10 Hz), todėl tikrasis užvedimo minimumas gali būti praleistas. Be to, ne visi automobiliai pateikia temperatūrą. Realiai naudojant būtų būtina įvertinti šią paklaidą.", "num2"),
  NUM("**Nėra modelio atnaujinimo.** Realioje sistemoje modelis turėtų būti periodiškai perkalibruojamas kiekvienam automobiliui, kai kaupiasi jo duomenys.", "num2"),
];

const refs = [
  H1("IX. Literatūra"),
  ...[
    "Saha, B., & Goebel, K. (2007). Battery Data Set. NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.",
    "Severson, K. A., Attia, P. M., Jin, N., Perkins, N., Jiang, B., Yang, Z., … Braatz, R. D. (2019). Data-driven prediction of battery cycle life before capacity degradation. Nature Energy, 4, 383–391.",
    "Ruetschi, P. (2004). Aging mechanisms and service life of lead–acid batteries. Journal of Power Sources, 127(1–2), 33–44.",
    "Berecibar, M., Gandiaga, I., Villarreal, I., Omar, N., Van Mierlo, J., & Van den Bossche, P. (2016). Critical review of state of health estimation methods of Li-ion batteries for real applications. Renewable and Sustainable Energy Reviews, 56, 572–587.",
    "Piller, S., Perrin, M., & Jossen, A. (2001). Methods for state-of-charge determination and their applications. Journal of Power Sources, 96(1), 113–120.",
    "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794.",
    "Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems, 30.",
    "Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.",
    "CALCE Battery Research Group. Battery data. University of Maryland, Center for Advanced Life Cycle Engineering.",
    "Metodiniai nurodymai „AI grupinis / individualus projektas“ (dalyko metodinė medžiaga).",
  ].map((t) => NUM(t, "refs")),
];

const tree = [
  ["README.md", "Paleidimo instrukcija, duomenų pastaba, CSV formatas"],
  ["requirements.txt", "Priklausomybės"],
  ["code/generate_data.py", "Sintetinių duomenų simuliatorius"],
  ["code/model.py", "Valymas, požymiai, eksperimentai, SHAP"],
  ["code/make_figures.py", "Visi paveikslai"],
  ["code/app.py", "Streamlit demonstracinė programa"],
  ["data/", "CSV duomenys, metrics.json, models/, demo_log.csv"],
  ["figures/", "PNG paveikslai"],
  ["report/build_report.js", "Šios ataskaitos generatorius (skaičiai imami iš metrics.json)"],
];
const appx = [
  H1("X. Priedai"),
  H2("A. Repozitorijos struktūra"),
  P("GitHub nuoroda: **[įrašyti sukūrus repozitoriją]**"),
  table(["Kelias", "Turinys"], tree, [2.5, 5]),
  H2("B. Paleidimo instrukcija"),
  ...[
    "pip install -r requirements.txt",
    "python code/generate_data.py",
    "python code/model.py",
    "python code/make_figures.py",
    "streamlit run code/app.py",
  ].map(CODE),
  SP(),
  P("Demonstracijai programoje galima įkelti failą data/demo_log.csv (testo akumuliatorius #12, pirmi 1 500 ciklų) arba įvesti vieną matavimą ranka."),
  H2("C. requirements.txt"),
  ...fs.readFileSync(path.join(ROOT, "requirements.txt"), "utf8").trim().split("\n").map(CODE),
  SP(),
  H2("D. Savęs įvertinimas pagal vertinimo kriterijus"),
  table(["Kriterijus", "Svoris", "Kur atitinka"], [
    ["1. Problemos analizė", "15", "III skyrius: aktualumas, vartotojai, reikalavimai, klasifikavimo lentelė"],
    ["2. Duomenų paruošimas", "15", "IV skyrius: šaltiniai, sintetinių duomenų pagrindimas, valymas, histogramos, koreliacijos"],
    ["3. Techninis įgyvendinimas", "25", `V–VI skyriai; veikiantis kodas; SOH R² = ${n(sX.R2, 2)}, klasifikacijos F1 = ${n(cX.F1, 2)} (> 0,70)`],
    ["4. Eksperimentai", "15", "VII skyrius: Baseline vs Proposed, 5 metrikų lentelės, konfuzijos matrica, abliacija, nesėkmės"],
    ["5. Etika, sauga", "10", "3.5 poskyris"],
    ["6. Ataskaitos kokybė", "10", "Universali struktūra I–X, requirements.txt, README"],
    ["7. Demonstracija", "10", "Streamlit programa (code/app.py), demo_log.csv"],
  ], [2.4, 0.8, 5]),
];

// =================================================================== DOKUMENTAS
const doc = new Document({
  features: { updateFields: true },
  creator: "[Vardas Pavardė]",
  title: "Akumuliatoriaus būklės (SOH) vertinimas be specialios įrangos",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 30, bold: true, color: "1F4E79" }, paragraph: { spacing: { before: 360, after: 160 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true, color: "2E75B6" }, paragraph: { spacing: { before: 240, after: 100 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
      ...["num", "num2", "refs"].map((r) => ({ reference: r, levels: [{ level: 0, format: LevelFormat.DECIMAL, text: r === "refs" ? "[%1]" : "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 400 } } } }] })),
    ],
  },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1701, right: 567 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 18 })] })] }) },
    children: [...titlePage, ...summary, ...problem, ...data, ...arch, ...modelDev, ...exper, ...limits, ...refs, ...appx],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(ROOT, "report", "Akumuliatoriaus_SOH_ataskaita.docx");
  fs.writeFileSync(out, buf);
  console.log("Išsaugota:", out);
});
