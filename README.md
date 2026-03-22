# Anàlisi i Comparació de Mètodes d'Aprendrege Automàtic i Aprenentatge Profund per a la Clasificació Multilabel d'Anuncis Oficials amb Objectius de Desenvolupament Sostenible (ODS)

TFG 66910

## Informació del TFG

- **Autor/a**: Jia Chun Comas Frigola
- **Tutor/a**: Dr. Antonio Lozano Bagen
- **Grau**: Grau en Enginyeria de Dades
- **Universitat**: Universitat Autònoma de Barcelona
- **Curs acadèmic**: 2025 – 2026

## Estructura del repositori

El repositori està estructurat de la següent manera:

```bash
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
│   ├── 01_data_preparation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_ml_classic.ipynb         ← SVM, Random Forest, XGBoost
│   ├── 04_deep_learning.ipynb      ← CNN, LSTM
│   └── 05_transformers.ipynb       ← BERT, fine-tuning
│
├── src/
│   ├── metrics.py                  ← Hamming Loss, F1, Jaccard, etc.
│   ├── schema.py                   ← schema de dades
│   ├── utils.py                    ← paths i funcions generals
│   └── models/
│       ├── ml_classic.py           ← classes/funcions dels models ML
│       ├── deep_learning.py        ← arquitectures CNN i LSTM
│       └── transformers.py         ← fine-tuning BERT
│
├── models/                         ← models entrenats guardats
│   ├── svm_br.pkl
│   ├── lstm_model.pt
│   └── bert_finetuned/
│
├── results/                        ← mètriques i comparatives
│   ├── metrics_ml_classic.csv
│   ├── metrics_deep_learning.csv
│   ├── metrics_transformers.csv
│   └── comparison_summary.csv
│
├── figures/                        ← gràfics generats (EDA + resultats)
│
├── .gitignore                      ← exclou data/raw, models pesants, etc.
├── .pre-commit-config.yaml         ← configuració del pre-commit
├── README.md                       ← descripció del projecte
└── requirements.txt                ← llibreries necessàries

```

## Instal·lació i execució

```bash
git clone https://github.com/jiacomas/TFG.git

# create virtual environment
python -m venv .venv

# activate virtual environment
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# run main script
python main.py
```
