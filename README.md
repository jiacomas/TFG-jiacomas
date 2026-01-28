# Anàlisi i Comparació de Mètodes d'Aprendrege Automàtic i Aprenentatge Profund per a la Clasificació Multilabel d'Anuncis Oficials amb Objectius de Desenvolupament Sostenible (ODS)
TFG 66910

## Informació del TFG
**Autor/a**: Jia Chun Comas Frigola     
**Tutor/a**: Dr. Antoni Lozano     
**Grau**: Grau en Enginyeria de Dades  
**Universitat**: Universitat Autònoma de Barcelona  
**Curs acadèmic**: 2025–2026

## Estructura del repositori
El repositori està estructurat de la següent manera:
```bash
/
├── data/                   # Dades del projecte
│   ├── pdfs/               # Dades originals (en format PDF)
│   ├── processed/          # Dades preprocessades
│   └── splits/             # Train / validation / test
│
├── src/                    # Codi font principal
│   ├── data/               # Càrrega i preprocessament de dades
│   │   ├── load_data.py
│   │   └── preprocess.py
│   │
│   ├── models/             # Implementació dels models
│   │   ├── model_a.py
│   │   ├── model_b.py
│   │   └── model_c.py
│   │
│   ├── training/           # Entrenament dels models
│   │   ├── train.py
│   │   └── hyperparams.py
│   │
│   ├── evaluation/         # Avaluació i comparació
│   │   ├── metrics.py
│   │   └── compare.py
│   │
│   └── utils/              # Funcions auxiliars
│       └── helpers.py
│
├── experiments/            # Resultats d’experiments
│   ├── exp_01/
│   │   ├── config.yaml
│   │   └── results.json
│   └── exp_02/
│
├── results/                # Resultats finals
│   ├── tables/             # Taules comparatives
│   ├── figures/            # Gràfiques
│   └── summary.md
│
├── tests/                  # Tests del codi
│
├── requirements.txt        # Dependències
├── README.md               # Descripció del projecte
└── LICENSE
```

## Instal·lació i execució
```bash
git clone https://github.com/jiacomas/TFG.git
cd TFG
pip install -r requirements.txt
# python main.py
```


