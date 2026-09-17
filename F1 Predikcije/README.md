# Predikcija rezultata Formule 1 za 2025. i 2026.

Ovaj projekat je kompletna, reproducibilna osnova za seminarski rad. Polazi od dostavljenog historijskog skupa podataka do kraja 2024. godine, trenira model bez curenja budućih informacija, testira ga hronološki na sezonama 2019-2024, pravi prognozu za 2025. i poredi je sa završenim prvenstvom, zatim pravi sekvencijalnu prognozu i nowcast za 2026.

Važno: rezultat sportskog događaja nije moguće garantovati. Model daje vjerovatnoće i intervale, a ne sigurnu tvrdnju. Sudari, kvarovi, vrijeme, sigurnosno vozilo, razvoj bolida i promjene pravila ostaju djelimično ili potpuno nepredvidivi.

## Najbrže pokretanje

Otvoriti terminal u glavnom direktoriju projekta i pokrenuti:

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python run_analysis.py
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_analysis.py
```

Brza provjera sa manjim brojem simulacija:

```bash
python run_analysis.py --quick
```

Drugačiji broj Monte Carlo simulacija:

```bash
python run_analysis.py --simulations 5000 --seed 42
```

## Šta program generiše

Glavni rezultati se nalaze u `outputs/`:

- `metrics.json`: objedinjene metrike modela i poređenja;
- `analysis_summary.txt`: kratki sažetak najvažnijih nalaza;
- `tables/`: sve ulazne kontrole, prognoze, vjerovatnoće, poretci i poređenja u CSV formatu;
- `figures/`: grafikoni spremni za seminarski rad;
- `models/`: istrenirani CatBoost model i metapodaci.

## Četiri jasno razdvojena scenarija

1. Stroga prognoza 2025: model vidi samo podatke zaključno sa 2024. Rezultati 2025. koriste se tek poslije prognoze, radi poređenja.
2. Stroga dvogodišnja prognoza 2026: model koristi samo historiju do kraja 2024. i poznati kalendar/sastav za 2026, bez rezultata 2025. ili 2026. Ovaj scenario najstrože ispituje koliko se signal iz 2024. može prenijeti dvije sezone unaprijed.
3. Sekvencijalna predsezonska prognoza 2026: koristi historijski model i konačne rezultate 2025, jer su oni legitimno poznati prije sezone 2026. Za provjeru se poredi sa stanjem nakon 13. utrke 2026.
4. Nowcast 2026: koristi stvarne bodove nakon Italije, 6. septembra 2026, i simulira preostalih deset utrka. Ovaj scenario nije isto što i predsezonska prognoza; on predstavlja ažuriranu projekciju.

## Metod u jednoj rečenici

Za svakog vozača prije utrke formiraju se samo historijski dostupne karakteristike vozača, konstruktora i staze; CatBoost i transparentna recent-form bazna linija daju ansambl ocjene, a Monte Carlo simulacija pretvara ocjene u vjerovatnoće pobjede, bodove i poredak prvenstva.

## Struktura projekta

```text
F1_seminarski_projekat/
|-- data/
|   |-- historical/          # dostavljeni CSV fajlovi do 2024.
|   `-- external/            # službeni snapshoti 2025/2026 i kalendari
|-- notebooks/
|   `-- F1_predikcija_2025_2026.ipynb
|-- outputs/
|   |-- figures/
|   |-- models/
|   `-- tables/
|-- src/
|   |-- data_pipeline.py
|   |-- forecasting.py
|   |-- modeling.py
|   `-- reporting.py
|-- run_analysis.py
`-- requirements.txt
```

## Sprječavanje curenja podataka

Sve rolling i expanding varijable koriste `shift(1)`: trenutni rezultat nikada ne ulazi u sopstvenu predikciju. Testne sezone dolaze hronološki poslije treninga. Rezultati 2025. nisu korišteni za strogu prognozu 2025, a rezultati 2026. nisu korišteni za predsezonsku provjeru 2026.

## Zašto model koristi period 2005-2024

Arhiv počinje 1950, ali su se bodovanje, broj učesnika, pouzdanost i pravila znatno mijenjali. Moderni prozor daje 8.290 redova iz 394 utrke i bolje odgovara problemu 2025-2026. Stari redovi ostaju sačuvani za deskriptivnu analizu.

## Reproducibilnost i snapshot 2026.

Eksterni podaci su namjerno spremljeni kao fiksni snapshot. Stanje 2026. obuhvata utrke zaključno sa VN Italije 6. septembra 2026, a datum analize je 10. septembar 2026. Zbog toga ponovno pokretanje daje isto akademsko poređenje i nakon što sezona bude nastavljena.

Izvori snapshot podataka navedeni su u `data/external/SOURCES.txt` i u Word dokumentu.
