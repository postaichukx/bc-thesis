# Reprodukovateľná analýza datasetu Elliptic

Tento adresár obsahuje experimentálnu časť bakalárskej práce
`Blockchain analýza`. Kód pracuje s pôvodným datasetom Elliptic, verejným
mapovaním anonymizovaných identifikátorov na bitcoinové transakcie a údajmi
z rozhrania Blockbook. Surové údaje ani odvodený dataset sa do repozitára
neukladajú.

## Prostredie

Odporúčaná verzia je Python 3.12:

```bash
cd analysis
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[gcn,dev]'
```

## Kde nájsť jednotlivé časti

Čísla priečinkov sledujú poradie kapitol v práci. Čísla 04–06
vychádzajú z poradia pracovných kapitol v `includes/core.tex`.

| Kapitola | Priečinok v `bc_analysis/` | Hlavné súbory |
| --- | --- | --- |
| 3. Elliptic a klasifikácia transakcií | `kap_03_elliptic/` | `dataset_audit.py` kontroluje vstupné CSV. `feature_table_export.py` vytvára tabuľku a slovník príznakov. |
| 4. Nástroj BitcoinTxSubgraphBuilder a údaje blockchainu | `kap_04_blockchain/` | `blockbook_client.py` obsluhuje API a cache. `transaction_collection.py` vyberá vzorku a získava transakcie. `transaction_features.py` počíta príznaky. |
| 5. Návrh reprodukovateľného experimentu | `kap_05_experiment/` | `txid_matching.py` overuje párovanie. `dataset_splits.py` pripravuje varianty dát a rozdelenia. `model_training.py` trénuje a hodnotí modely. |
| 6. Overenie dát a vyhodnotenie | `kap_06_vysledky/` | `result_figures.py` tvorí obrázky zo zozbieraných údajov a výsledkov. |

Testy sú v `tests/` pod rovnakými názvami kapitol. Spoločný vstupný bod
`bc_analysis/cli.py` spúšťa príkazy uvedené nižšie.

## Vstupné súbory

### Audit pôvodných údajov (bod 1.2)

Audit načíta všetky časové kroky a nepotrebuje mapovanie ani databázový
server. Z koreňa projektu ho spustíme takto:

```bash
analysis/.venv/bin/python -m bc_analysis audit \
  --elliptic-dir analysis/data/raw/kap_03_elliptic/originalne_csv \
  --output-dir analysis/data/processed/kap_03_elliptic/audit_dat
```

Vznikne `audit.json` s pôvodom údajov, SHA-256, počtami a kontrolami
a `elliptic_nodes.csv.gz` s pôvodnými príznakmi a pripojenými označeniami.
`unknown` zostávajú medzi uzlami. Majú `label = -1` a `is_labeled = false`;
ide o chýbajúcu odpoveď, nie tretiu triedu. Príznaky a hrany môžu poskytnúť
neoznačený kontext podľa zvolenej metódy. Príznak `is_labeled` ešte nie je
trénovacou maskou: neskôr ho pretíname s konkrétnym časovým rozdelením.

Pri neplatných údajoch audit nepublikuje nový súbor uzlov a končí nenulovým
stavom; pri štrukturálnych chybách preruší načítanie s chybou. Pôvodné CSV
nemení. Surové aj odvodené záznamy zostávajú lokálne podľa `.gitignore`.

Prehľad práce je v `../WORK_LOG.md`, aktuálny kontext v
`../memory_codex.md` a plánované závislosti v `../WORKFLOW_UML.svg`.

#### Ako sme vykonali bod 1.2

1. Získali sme tri pôvodné CSV od vydavateľa Elliptic a zaznamenali pôvod,
   podmienky použitia a odtlačky SHA-256. Surové súbory nemeníme.
2. Overili sme formát: tabuľka príznakov nemá hlavičku a obsahuje interný
   identifikátor a 166 príznakov vrátane časového kroku. Ďalšie súbory
   obsahujú označenia tried a dvojice identifikátorov tvoriace hrany.
3. Skontrolovali sme duplicity, chýbajúce alebo neplatné čísla, rozsah
   časových krokov a zhodu identifikátorov medzi všetkými tromi súbormi.
4. Označenia sme pripojili podľa identifikátora, nie podľa poradia riadkov.
   Pri čítaní veľkého súboru používame bloky po 20 000 riadkoch.
5. Uchovali sme všetky transakcie aj hrany. Pre `unknown` sme vyznačili
   chýbajúci cieľ. Výstupom je kontrolný protokol a spojená tabuľka uzlov.

Overený súbor obsahuje 203 769 transakcií a 234 355 hrán v 49 časových
krokoch. Triedy: 42 019 `licit`, 4 545 `illicit`, 157 205 `unknown`.
Kontrolované duplicity, neplatné číselné hodnoty, chýbajúce označenia
a neexistujúce konce hrán sa v týchto vstupoch nevyskytli.
Tento krok overuje štruktúru a úplnosť dodaných súborov. Nepotvrdzuje
vecnú správnosť označení a ešte nevytvára mapovanie na blockchain.

#### Úplná tabuľka a slovník stĺpcov

```bash
analysis/.venv/bin/python -m bc_analysis export-table \
  --elliptic-dir analysis/data/raw/kap_03_elliptic/originalne_csv \
  --audit-dir analysis/data/processed/kap_03_elliptic/audit_dat \
  --output-dir analysis/data/processed/kap_03_elliptic/tabulky_priznakov
```

Export najprv porovná SHA-256 vstupov s úspešným auditom. Vytvorí
`elliptic_all_transactions.csv` so všetkými 203 769 riadkami a 171 stĺpcami:
interný identifikátor, štyri stĺpce označení a všetkých 166 príznakov.
Pôvodné číselné zápisy aj poradie riadkov zachováva bez prepočítania.
Oddeľovač je čiarka, desatinný znak bodka, kódovanie UTF-8.
Identifikátor a štyri stĺpce označení sú metadáta. Pri budúcom učení
ich nezaradíme medzi vstupné príznaky klasifikátora.
Grafové hrany zostávajú v pôvodnom `elliptic_txs_edgelist.csv`; samostatný
riadok transakcie ich nenahrádza.

`elliptic_column_dictionary.csv` dokumentuje každý stĺpec. Číslo vo forme
`feature_023__…` je pôvodné číslo príznaku. Názvy vychádzajú z
[obnovených významov Šafářa a spoluautorov](https://github.com/nesfit/DeanonymizedEllipticBitcoinDataset/blob/a5dbbcc655783003d942d0d54ad8404f0b7b3b93/README.md),
nie z oficiálneho zverejnenia významov spoločnosťou Elliptic.
Zo 166 príznakov je opísaných 151 vrátane časového kroku; 15 má označenie
`unknown_meaning`. Publikované hodnoty zostávajú v pôvodnej číselnej škále.
Napríklad stĺpec s významom počtu vstupov preto nečítame priamo ako
surový počet. Pri adresných príznakoch slovník upozorňuje na spoločný
referenčný blok 575 059 a pri nasledovníkoch na budúci kontext.

Malý Excel `elliptic_dictionary_and_examples.xlsx` je v podadresári
`outputs/stage_1_2_2026_11_03/`. Obsahuje slovník a prvých desať záznamov
každej triedy. Týchto 30 príkladov slúži na prezeranie stĺpcov, nie na
odhad pomeru tried ani na rozdelenie učenia. Úplné údaje obsahuje CSV.
`table_manifest.json` uchováva odtlačok exportu, zdroje, počty a výber
príkladov pre tento prehľad. `table_verification.json` zaznamenáva nezávislé
porovnanie všetkých číselných polí a identifikátorov s pôvodným CSV,
kontrolu označení, slovníka, ukážok a uložených filtrov a panelov Excelu.

#### Posúdenie notebooku EDA, Graph & Drift

Prečítali sme verziu 1 notebooku
[Oguz Enver Uzan: Bitcoin Anti Money Laundering: EDA, Graph & Drift](https://www.kaggle.com/code/oguzuzan/bitcoin-anti-money-laundering-eda-graph-drift),
získanú cez verejné rozhranie Kaggle 2026-11-03. Dostupný zdroj obsahoval
74 buniek bez uložených výstupov. Kód sme nespúšťali a uvádzané skóre
nepovažujeme za nezávisle overené výsledky.

Pre ďalšie etapy sú užitočné kontroly spojenia podľa identifikátora,
prehľad tried podľa času, analýza grafu a oddelenie učiacich, validačných
a testovacích krokov. Notebook používa kroky 1–29 na počiatočné učenie,
30–34 na validáciu a 35–49 na test. Neoznačené uzly zostávajú v grafe,
ale bez cieľa riadeného učenia. Mapovanie na reálne bitcoinové `txid`
ani blockchainový konektor notebook nerieši.

Pri vlastnom experimente upravíme tieto metodické rozhodnutia:

- Bunky 57 a 60 zoraďujú modely podľa testovacieho skóre a vyberajú z nich
  najlepší variant. My zvolíme model, príznaky aj prah na validácii;
  test ponecháme na záverečné vyhodnotenie.
- Výber časových zástupných príznakov podľa eta-kvadrát využíva všetky
  kroky. Ak takýto výber zaradíme do modelovania, vypočítame ho iba
  z dovolených učiacich údajov.
- Vstupné a výstupné stupne opisujú uložený graf Elliptic. Vrchol bez
  predchodcu v tejto vzorke nemusí byť pôvodným zdrojom prostriedkov.
- Dostupnosť agregátov a nasledovníkov posúdime vzhľadom na okamih
  predikcie; vlastnosti retrospektívneho grafu nie sú automaticky
  dostupné pri vyhodnotení novej transakcie.

Notebook nám slúži ako podnet na kontroly a budúce experimenty.
Audit 1.2 nenahrádza tréningom ani hotovými výsledkami z cudzieho kódu.

### Starší experimentálny postup

Nasledujúce príkazy už v projekte existovali. Používajú šesť vybraných
časových krokov a SQLite cache. Nie sú totožné s novým auditom všetkých
údajov ani s odsúhlaseným cieľovým úložiskom PostgreSQL. Migrácia úložiska
a zosúladenie variantov modelov s aktuálnym plánom sa ešte nevykonali.

Príkaz `prepare` očakáva tri pôvodné súbory Elliptic a mapovanie vo formáte
`txId,transaction`. Mapovanie nie je vytvárané nástrojom
BitcoinTxSubgraphBuilder; ide o samostatný vstup experimentu.

```bash
bc-analysis prepare \
  --elliptic-dir /cesta/k/elliptic/raw \
  --mapping /cesta/k/Result.csv
```

Výber je pevne obmedzený na časové kroky 20, 29, 30, 32, 35 a 42.

## Postup experimentu

```bash
# Stiahne transakcie zo zvolených blokov a uloží odpovede do SQLite cache.
bc-analysis collect --mode blocks

# Skryje 20 % známych mapovaní a vyhodnotí spätné dohľadanie transakcií.
bc-analysis mapping-eval

# Vytvorí tri varianty príznakov/grafu a oba spôsoby rozdelenia.
bc-analysis build-datasets

# Spustí LogisticRegression, RandomForestClassifier a vlastnú dvojvrstvovú GCN.
bc-analysis train

# Vytvorí tabuľky a obrázky používané v práci.
bc-analysis report
```

Všetky príkazy majú prepínač `--help`. Predvolený Blockbook server možno
zmeniť argumentom `--base-url`. Zber je obnoviteľný: každá úspešná odpoveď
sa uloží do `analysis/data/cache/kap_04_blockchain/blockbook.sqlite3`.

Spracované transakcie a manifest sú v `data/processed/kap_04_blockchain/`.
Dáta pre modely sú v `data/processed/kap_05_experiment/`. Tabuľky metrík
a párovania sú v `results/kap_06_vysledky/`.

## Reprodukovateľnosť a interpretácia

- Pri každom behu sa vytvorí manifest s cestami, počtami záznamov, SHA-256
  odtlačkami vstupov, verziami zdrojov a knižníc, parametrami a
  počiatočnými hodnotami generátora náhodných čísel.
- Normalizácia sa vždy učí iba z trénovacej množiny.
- Neoznačené vrcholy sa pri GCN ponechávajú v grafe, ale nevstupujú do
  stratovej funkcie ani metrík.
- LogisticRegression používa vyvážené váhy tried a validáciou vybrané
  `C`; RandomForestClassifier používa 500 stromov; GCN má 128 skrytých
  hodnôt, ReLU, dropout 0,5 a optimalizátor Adam. Ide o referenčné modely,
  nie o nový algoritmus navrhnutý v práci.
- Report obsahuje aj prehľad skutočných 28 221 vrcholov a 31 653 hrán
  vo všetkých šiestich vybraných časových komponentoch.
- Nájdená zhoda je zhoda záznamu datasetu s verejnou transakciou. Nie je to
  identifikácia osoby ani dôkaz trestnej činnosti.
- Licencia Elliptic CC BY-NC-ND 4.0 neumožňuje publikovať upravený dataset.
  Preto sa publikujú iba postup, konfigurácia a súhrnné výsledky.
