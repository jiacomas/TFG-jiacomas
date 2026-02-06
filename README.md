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
│   ├── raw/               # Dades originals (en format PDF)
│   ├── processed/          # Dades preprocessades
│   └── splits/             # Train / validation / test
│
├── src/                    # Codi font principal
│   ├── data/               # Càrrega i preprocessament de dades
│   │   ├── load_data.py
│   │   ├── merge_sources.py
│   │   └── preprocess.py
│   │
│   ├── models/             # Implementació dels models
│   │   ├── model_a.py
│   │   ├── model_b.py
│   │   └── model_c.py
│   │
│   ├── evaluation/         # Avaluació i comparació
│   │   ├── metrics.py
│   │   └── compare.py
│   │
│   └── utils/              # Funcions auxiliars
│       ├── config.py
│       └── seed.py
│
├── experiments/            # Resultats d’experiments
│   ├── exp_01/
│   └── exp_02/
│
├── results/                # Resultats finals
│   ├── plots/              # Gràfiques i taules
│   ├── metrics.csv
│   └── summary.md
│
├── requirements.txt        # Dependències
├── README.md               # Descripció del projecte
└── main.py                 # Script principal

```

## Instal·lació i execució
```bash
git clone https://github.com/jiacomas/TFG.git
pip install -r requirements.txt
python main.py
```


