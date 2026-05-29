# Classificació Multilabel d'Anuncis Oficials del BOPB en ODS

**Treball de Fi de Grau — Grau d'Enginyeria de Dades**

Escola d'Enginyeria, Universitat Autònoma de Barcelona (UAB)

- **Autor:** Jia Chun Comas Frigola
- **Tutor:** Antonio Lozano (Ciències de la Computació)
- **Curs:** 2025/2026

## Descripció

El [Butlletí Oficial de la Província de Barcelona (BOPB)](https://bop.diba.cat) publica anualment milers d'anuncis oficials procedents d'institucions públiques. Aquest projecte explora l'aplicació de tècniques d'**aprenentatge automàtic** i **aprenentatge profund** per predir automàticament les etiquetes dels [Objectius de Desenvolupament Sostenible (ODS)](https://bop.diba.cat/ca/ods) associades a cada anunci.

Com que un sol anunci pot associar-se a diversos ODS simultàniament, el problema es tracta com una tasca de **classificació multilabel** sobre text en català i castellà.

## Estructura del projecte

```
/
│
├── data/
│   ├── analysis/                   ← resultats de l'EDA (notebook 02)
│   ├── processed/                  ← output del notebook 01
│   └── raw/                        ← dades originals, mai es modifiquen
│       ├── Anuncis_2022_2024.xlsx
│       └── description/
│           ├── anuncis_BOPB_2022_contingut.csv
│           ├── anuncis_BOPB_2023_contingut.csv
│           └── anuncis_BOPB_2024_contingut.csv
│
├── notebooks/
│   ├── 01_data_preparation.ipynb   ← integració, neteja i splits
│   ├── 02_eda.ipynb                ← anàlisi exploratòria de dades
│   └── 03_modelling.ipynb          ← normalització ML, TF-IDF i splits
│
├── src/
│   ├── make_figures.py             ← genera gràfics finals a partir dels JSON de results/
│   ├── metrics.py                  ← Hamming Loss, F1, Jaccard, etc.
│   ├── schema.py                   ← constants de columnes
│   ├── utils.py                    ← paths i configuració global
│   └── models/
│       ├── ml_classic.py           ← Random Forest i XGBoost
│       └── deep_learning.py        ← BERTa i mmBERT
│
├── figures/                        ← gràfics generats per EDA i resultats
├── models/                         ← models entrenats guardats (.pkl, .pt)
├── results/                        ← summaries de W&B per model (.json)
│
├── .gitignore
├── .pre-commit-config.yaml
├── README.md
└── requirements.txt
```

## Instal·lació

### Prerequisits

- Python 3.11 o superior
- [Homebrew](https://brew.sh) (macOS)
- Git

### 1. Clona el repositori

```bash
git clone https://github.com/jiacomas/TFG-jiacomas.git
cd TFG-jiacomas
```

### 2. Crea i activa un entorn virtual

```bash
python -m venv tfg
source tfg/bin/activate        # macOS / Linux
# tfg\Scripts\activate         # Windows

conda create -n tfg python=3.11 -y # create miniconda environment
conda activate tfg
```

### 3. Dependències de sistema

**Només macOS** — XGBoost requereix `libomp` (OpenMP), que Apple Clang
no inclou per defecte. Instal·la-la via Homebrew **abans** de fer
`pip install`:

```bash
brew install libomp
```

> **Linux i Windows** — cap acció necessària. El wheel de PyPI ja inclou
> les dependències OpenMP en aquests sistemes.

> **Per què?** XGBoost utilitza paral·lelisme OpenMP per accelerar l'entrenament dels arbres. En macOS, Apple Clang no inclou `libomp` per defecte, de manera que el binari wheel de PyPI no pot trobar la llibreria en temps d'execució. Instal·lar-la via Homebrew resol l'error `Library not loaded: @rpath/libomp.dylib`.

### 4. Instal·la les dependències de Python

```bash
pip install -r requirements.txt
pre-commit install
```

## Ús

Executa els notebooks en ordre seqüencial des de JupyterLab:

```bash
jupyter lab
```

| Notebook              | Descripció                                          | Output principal     |
| --------------------- | --------------------------------------------------- | -------------------- |
| `01_data_preparation` | Integració XLSX + CSV, neteja, splits               | `data/processed/`    |
| `02_eda`              | Distribució ODS, co-ocurrències, longitud de text   | `figures/eda`        |
| `03_modelling`        | Normalització ML, TF-IDF, splits estratificats      | `data/processed/`    |

Entrenament dels models (des de l'arrel del projecte):

```bash
python -m src.models.ml_classic       # Random Forest + XGBoost
python -m src.models.deep_learning    # BERTa + mmBERT
python src/make_figures.py \
    --rf results/rf_summary.json --xgb results/xgb_summary.json \
    --berta results/berta_summary.json --mmbert results/mmbert_summary.json
```

> ⚠️ El conjunt de test (`data/processed/split_test.parquet`) **no s'avalua fins al final**, un cop la selecció de models és definitiva.

## Seguiment d'experiments

Tots els experiments es registren a [Weights & Biases](https://wandb.ai) sota el projecte `comparation-multilabel`. Cada notebook correspon a un run independent:

| Run W&B            |  Models                 |
| ------------------ | ------------------------|
| `ml_classic.py`    | Random forest & XGBoost |
| `deep_learning.py` | BERTa & mmBERT          |

Per accedir al dashboard:

```bash
wandb login    # només cal fer-ho un cop per màquina
```

## Mètriques d'avaluació

Totes les mètriques estan implementades a `src/metrics.py` i s'apliquen de forma consistent a tots els models:

| Mètrica             | Descripció                                                      |
| ------------------- | --------------------------------------------------------------- |
| **F1-micro**        | Mètrica principal; pes proporcional a la freqüència de cada ODS |
| **F1-macro**        | Mitjana no ponderada; més just amb ODS minoritaris              |
| **Hamming Loss**    | Proporció d'etiquetes incorrectes (↓ millor)                    |
| **Exact Match**     | Proporció de mostres amb totes les etiquetes correctes          |
| **Jaccard (micro)** | Intersecció sobre unió de conjunts d'etiquetes                  |

## Dades

Les dades originals provenen del [BOPB](https://bop.diba.cat) i cobreixen els anys 2022–2024 (35.520 anuncis únics). Per motius de confidencialitat i mida, **no s'inclouen al repositori**. El directori `data/raw/` està exclòs via `.gitignore`.

Per reproduir els experiments cal col·locar els fitxers originals a:

```
data/raw/Anuncis_2022_2024.xlsx
data/raw/description/anuncis_BOPB_2022_contingut.csv
data/raw/description/anuncis_BOPB_2023_contingut.csv
data/raw/description/anuncis_BOPB_2024_contingut.csv
```

## Llicència

Aquest projecte és un treball acadèmic de la Universitat Autònoma de Barcelona. Les dades del BOPB són propietat de la Diputació de Barcelona.
