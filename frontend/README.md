# BOPB × ODS - Frontend

Tauler interactiu del TFG *Classificació multilabel d'anuncis oficials del BOPB segons els 17 ODS*.

## Estructura

```text
frontend/
├── src/
│   ├── App.jsx       # Aplicació React en un únic fitxer (totes les pàgines i components)
│   ├── main.jsx      # Punt d'entrada de React
│   └── index.css     # Import de Tailwind CSS
├── index.html
├── vite.config.js
└── package.json
```

## Pàgines

| # | Pàgina | Descripció |
|---|---------|------------|
| 01 | Introducció | AFE (EDA): distribució dels ODS, mètriques multilabel i mapa de calor de coocurrència |
| 02 | Models | Comparació del rendiment entre RF, XGBoost, BERTa i mmBERT |
| 03 | Predir | Predicció en temps real mitjançant l'API de Claude (simula el comportament del model entrenat) |

## Flux de dades

Les dades de l'AFE (freqüències dels ODS, matriu de coocurrència i mètriques multilabel) es carreguen en temps d'execució des de `../results/eda_data.json`, fitxer generat pel quadern `notebooks/02_eda.ipynb`.

Vite serveix el directori `results/` com a directori públic (`publicDir: '../results'` a `vite.config.js`), de manera que el fitxer està disponible a `/eda_data.json` durant el desenvolupament i es copia automàticament a `dist/` durant la compilació.

Per regenerar el fitxer de dades, només cal tornar a executar el quadern `02_eda.ipynb`.

## Instal·lació

```bash
npm install
npm run dev     # http://localhost:5173
npm run build   # Compilació per a producció → dist/
```

## Requisits

- Node 18 o superior.
- Un fitxer `.env` o un servidor intermediari (*proxy*) que proporcioni `ANTHROPIC_API_KEY` per a la pàgina de predicció (la pàgina 03 crida directament l'API de Claude des del navegador; aquesta configuració només és adequada per al desenvolupament local).
