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
│   ├── 03_ml_classic.ipynb         ← SVM, Random Forest, XGBoost
│   ├── 04_deep_learning.ipynb      ← CNN, BiLSTM
│   └── 05_transformers.ipynb       ← RoBERTa, fine-tuning
│
├── src/
│   ├── metrics.py                  ← Hamming Loss, F1, Jaccard, etc.
│   ├── schema.py                   ← constants de columnes
│   ├── utils.py                    ← paths i configuració global
│   └── models/
│       ├── ml_classic.py
│       ├── deep_learning.py
│       └── transformers.py
│
├── models/                         ← models entrenats guardats (.pkl, .pt)
├── results/                        ← mètriques i comparatives (.csv)
├── figures/                        ← gràfics generats per EDA i resultats
│
├── .gitignore
├── .pre-commit-config.yaml
├── Makefile
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
```

### 5. Instal·la els hooks de pre-commit

```bash
pre-commit install
```

A partir d'aquest moment, cada `git commit` executarà automàticament `black`, `ruff` i `isort` sobre el codi i les cel·les dels notebooks.

### Instal·lació en un sol pas (via Makefile)

```bash
make install
```

## Ús

Executa els notebooks en ordre seqüencial des de JupyterLab:

```bash
jupyter lab
```

| Notebook              | Descripció                                        | Output principal                    |
| --------------------- | ------------------------------------------------- | ----------------------------------- |
| `01_data_preparation` | Integració XLSX + CSV, neteja, splits             | `data/processed/`                   |
| `02_eda`              | Distribució ODS, co-ocurrències, longitud de text | `data/analysis/`, `figures/`        |
| `03_ml_classic`       | BR, CC, LP amb SVM / LR / RF / XGBoost            | `results/metrics_ml_classic.csv`    |
| `04_deep_learning`    | CNN i BiLSTM amb PyTorch                          | `results/metrics_deep_learning.csv` |
| `05_transformers`     | Fine-tuning RoBERTa (PlanTL-GOB-ES)               | `results/metrics_transformers.csv`  |

> ⚠️ El conjunt de test (`data/processed/split_test.parquet`) **no s'avalua fins al final**, un cop la selecció de models entre els notebooks 03–05 és definitiva.

## Seguiment d'experiments

Tots els experiments es registren a [Weights & Biases](https://wandb.ai) sota el projecte `bopb-ods-multilabel`. Cada notebook correspon a un run independent:

| Run W&B            | Notebook | Models                       |
| ------------------ | -------- | ---------------------------- |
| `03_ml_classic`    | 03       | BR/CC/LP × LR/SVM/RF/XGBoost |
| `04_deep_learning` | 04       | TextCNN, BiLSTM              |
| `05_transformers`  | 05       | RoBERTa-BNE, mBERT           |

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
