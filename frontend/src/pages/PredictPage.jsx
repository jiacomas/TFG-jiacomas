import React, { useState, useEffect, useMemo } from 'react';
import { ODS, METRICS, COMPUTE, MODEL_INFO, MODEL_ORDER, fmtDuration } from '../constants';
import { ODSBadge } from '../components';

async function callModelAPI(text, modelKey) {
  const modelBehaviors = {
    rf: `Estàs simulant un classificador Random Forest entrenat amb features TF-IDF per a un problema de classificació multilabel d'ODS en anuncis del BOPB.
Aquest model és MOLT CONSERVADOR: té alta precisió però molt baix recall (F1-macro 0.315).
- Només prediu els ODS més obvis i dominants del text
- Tendeix a no predir ODS minoritaris (14, 6, 5, 12, 2) tret que apareguin múltiples paraules clau directes
- Si el text és curt, ambigu o no clarament d'algun ODS, predius RES (llista buida)
- Acostuma a predir 1-2 ODS com a màxim
- Funciona només amb paraules clau, no entén context semàntic`,
    xgb: `Estàs simulant XGBoost amb gradient boosting sobre features TF-IDF per a classificació multilabel d'ODS al BOPB.
Bon equilibri precisió-recall (F1-macro 0.688). Comportament:
- Detecta ODS principals i alguns secundaris si hi ha senyals clares
- Conservador però menys que Random Forest
- Bo amb ODS majoritaris (8, 11, 9, 10, 3), més limitat amb minoritaris (14, 6, 12)
- Acostuma a predir 1-3 ODS
- Es basa en paraules clau però amb millor calibratge que RF`,
    berta: `Estàs simulant BERTa, un model RoBERTa preentrenat en català fine-tunejat per classificació multilabel ODS.
Bon recall en classes minoritàries (F1-macro 0.669, recall macro 0.666).
- Captura context semàntic del català
- Sensible a ODS minoritaris (6, 13, 16, 17)
- Pot tenir falsos positius en alguns casos
- Acostuma a predir 2-4 ODS
- A vegades sobreprediu però rarament oblida un ODS rellevant
- Treballa sobre els primers 256 tokens (pot perdre info de textos llargs)`,
    mmbert: `Estàs simulant mmBERT, ModernBERT multilingüe fine-tunejat, EL MILLOR MODEL del projecte.
F1-macro 0.742, F1-micro 0.833. Comportament:
- Excel·lent en context semàntic multilingüe
- Millor model per ODS minoritaris (14, 6, 2): aconsegueix detectar-los
- Equilibri òptim entre precisió i recall
- Acostuma a predir 2-4 ODS amb alta precisió
- Detecta dependències subtils entre ODS (3↔11, 9↔11, 8↔11)
- Treballa sobre els primers 256 tokens`,
  };

  const ods_descriptions = ODS.map(o => `${o.n}: ${o.short}`).join('; ');

  const prompt = `${modelBehaviors[modelKey]}

Llista dels 17 ODS:
${ods_descriptions}

Anunci del BOPB a classificar:
"""
${text.slice(0, 8000)}
"""

Comporta't EXACTAMENT com el model descrit. Retorna NOMÉS un JSON vàlid amb aquesta estructura, sense cap text addicional ni codi fences:
{
  "predicted": [llista d'enters dels ODS predits, p.ex. [8, 11]],
  "confidences": {"1": 0.05, "2": 0.03, ..., "17": 0.08},
  "reasoning": "Frase curta en català (max 2 frases) explicant per què aquest model en concret prediu això"
}

Les confidences han de tenir totes les 17 claus (de "1" a "17") amb valors entre 0 i 1.
Els valors d'ODS dins "predicted" han de coincidir amb els que tinguin confidence alta segons el llindar característic del model.
Si el model no prediu cap ODS, "predicted" ha de ser [].`;

  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "claude-sonnet-4-6",
      max_tokens: 1000,
      messages: [{ role: "user", content: prompt }],
    })
  });

  if (!response.ok) throw new Error('API error: ' + response.status);

  const data = await response.json();
  const textResponse = data.content
    .map(b => b.type === 'text' ? b.text : '')
    .filter(Boolean)
    .join('');

  const cleaned = textResponse.replace(/```json/g, '').replace(/```/g, '').trim();
  const parsed = JSON.parse(cleaned);

  const predicted = (parsed.predicted || []).filter(n => Number.isInteger(n) && n >= 1 && n <= 17);
  const confidences = {};
  for (let i = 1; i <= 17; i++) {
    const v = parsed.confidences?.[String(i)];
    confidences[i] = (typeof v === 'number' && v >= 0 && v <= 1) ? v : 0.05;
  }

  return {
    predicted,
    confidences,
    reasoning: parsed.reasoning || '',
  };
}

function LatestResult({ entry }) {
  const sorted = useMemo(() => {
    return Object.entries(entry.confidences || {})
      .map(([k, v]) => ({ n: parseInt(k), c: v }))
      .sort((a, b) => b.c - a.c);
  }, [entry]);

  return (
    <div className="mt-8 bg-white border border-stone-200 rounded-lg overflow-hidden">
      <div className="px-5 py-3 border-b border-stone-200 bg-stone-900 text-white flex items-center justify-between">
        <span className="text-xs uppercase tracking-wider font-medium">Predicció · {entry.modelName}</span>
        <span className="text-xs text-stone-400 font-mono">{entry.timestamp.toLocaleTimeString('ca-ES')}</span>
      </div>

      <div className="p-5">
        <div className="text-xs uppercase tracking-wider text-stone-500 mb-2">ODS predits</div>
        {entry.predicted.length === 0 ? (
          <p className="text-sm text-stone-500 italic">Cap ODS predit (el model és conservador o el text és curt).</p>
        ) : (
          <div className="flex flex-wrap gap-2">
            {entry.predicted.map(n => <ODSBadge key={n} n={n} withLabel />)}
          </div>
        )}

        {sorted.length > 0 && (
          <div className="mt-5">
            <div className="text-xs uppercase tracking-wider text-stone-500 mb-3">Probabilitats per ODS</div>
            <div className="space-y-1.5">
              {sorted.slice(0, 10).map(({ n, c }) => (
                <div key={n} className="flex items-center gap-3">
                  <div className="w-16 text-xs flex items-center gap-2">
                    <span
                      className="w-2 h-2 rounded-full shrink-0"
                      style={{ backgroundColor: ODS[n - 1].color }}
                    />
                    <span className="text-stone-600">ODS {n}</span>
                  </div>
                  <div className="flex-1 h-4 bg-stone-100 rounded-sm overflow-hidden relative">
                    <div
                      className="h-full transition-all duration-500"
                      style={{
                        width: `${c * 100}%`,
                        backgroundColor: entry.predicted.includes(n) ? ODS[n - 1].color : '#d6d3d1',
                      }}
                    />
                  </div>
                  <div className="w-12 text-xs font-mono tabular-nums text-right text-stone-700">
                    {(c * 100).toFixed(0)}%
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {entry.reasoning && (
          <div className="mt-5 pt-5 border-t border-stone-100">
            <div className="text-xs uppercase tracking-wider text-stone-500 mb-2">Raonament del model</div>
            <p className="text-sm text-stone-700 leading-relaxed italic">{entry.reasoning}</p>
          </div>
        )}

        <div className="mt-5 pt-5 border-t border-stone-100">
          <div className="text-xs uppercase tracking-wider text-stone-500 mb-3">Cost computacional</div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
            <div className="bg-stone-50 rounded p-3">
              <div className="text-stone-500 mb-1">Inferència (modelat)</div>
              <div className="font-mono text-stone-900 text-lg tabular-nums">{entry.compute.simulated_inference_ms.toFixed(1)} ms</div>
            </div>
            <div className="bg-stone-50 rounded p-3">
              <div className="text-stone-500 mb-1">RAM estimat</div>
              <div className="font-mono text-stone-900 text-lg tabular-nums">{(entry.compute.ram_mb / 1024).toFixed(2)} GB</div>
            </div>
            <div className="bg-stone-50 rounded p-3">
              <div className="text-stone-500 mb-1">Mida del model</div>
              <div className="font-mono text-stone-900 text-lg tabular-nums">{entry.compute.model_size_mb >= 1024 ? (entry.compute.model_size_mb / 1024).toFixed(2) + ' GB' : entry.compute.model_size_mb + ' MB'}</div>
            </div>
            <div className="bg-stone-50 rounded p-3">
              <div className="text-stone-500 mb-1">Latència API (real)</div>
              <div className="font-mono text-stone-900 text-lg tabular-nums">{entry.compute.api_elapsed_ms.toFixed(0)} ms</div>
            </div>
          </div>
          <p className="text-[10px] text-stone-400 mt-3 leading-relaxed">
            Els temps d'inferència i RAM són estimats a partir de les mesures reals del model durant l'avaluació del TFG.
            Aquesta interfície utilitza l'API de Claude per simular el comportament del model entrenat.
          </p>
        </div>
      </div>
    </div>
  );
}

function HistoryEntry({ entry }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="bg-white border border-stone-200 rounded-lg">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full text-left p-4 hover:bg-stone-50 transition"
      >
        <div className="flex items-baseline justify-between mb-2">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: entry.modelColor }} />
            <span className="text-sm font-medium text-stone-900">{entry.modelName}</span>
          </div>
          <span className="text-[10px] text-stone-500 font-mono">{entry.timestamp.toLocaleTimeString('ca-ES')}</span>
        </div>
        <p className="text-xs text-stone-500 leading-relaxed line-clamp-2">{entry.textPreview}</p>
        <div className="flex flex-wrap gap-1 mt-3">
          {entry.predicted.length === 0 ? (
            <span className="text-[10px] text-stone-400 italic">cap predicció</span>
          ) : (
            entry.predicted.map(n => <ODSBadge key={n} n={n} />)
          )}
        </div>
        <div className="flex gap-4 mt-3 text-[10px] text-stone-500 font-mono">
          <span>{entry.compute.simulated_inference_ms.toFixed(1)} ms</span>
          <span>{(entry.compute.ram_mb / 1024).toFixed(1)} GB</span>
          <span className="ml-auto">{expanded ? '▲' : '▼'}</span>
        </div>
      </button>
      {expanded && (
        <div className="border-t border-stone-100 p-4 bg-stone-50">
          <div className="text-[10px] uppercase tracking-wider text-stone-500 mb-2">Probabilitats</div>
          <div className="space-y-1">
            {Object.entries(entry.confidences || {})
              .map(([k, v]) => ({ n: parseInt(k), c: v }))
              .sort((a, b) => b.c - a.c)
              .slice(0, 6)
              .map(({ n, c }) => (
                <div key={n} className="flex items-center gap-2 text-[11px]">
                  <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: ODS[n - 1].color }} />
                  <span className="text-stone-600 w-12">ODS {n}</span>
                  <div className="flex-1 h-1.5 bg-stone-200 rounded-full overflow-hidden">
                    <div className="h-full" style={{ width: `${c * 100}%`, backgroundColor: ODS[n - 1].color }} />
                  </div>
                  <span className="text-stone-500 font-mono w-8 text-right">{(c * 100).toFixed(0)}%</span>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ComparisonView({ entries }) {
  const allOds = new Set();
  entries.forEach(e => e.predicted.forEach(n => allOds.add(n)));
  const odsList = Array.from(allOds).sort((a, b) => a - b);

  if (odsList.length === 0) {
    return <p className="text-xs text-stone-600">Cap model ha predit cap ODS per aquest text.</p>;
  }

  return (
    <div className="text-xs">
      <p className="text-stone-700 mb-3">
        Has provat <strong>{entries.length} models</strong> amb el mateix text.
        Comparativa d'ODS predits:
      </p>
      <table className="w-full">
        <thead>
          <tr className="text-stone-500 border-b border-amber-200">
            <th className="text-left pb-1 font-medium">Model</th>
            {odsList.map(n => (
              <th key={n} className="pb-1 px-1">
                <span
                  className="inline-block w-5 h-5 rounded text-white text-[10px] font-bold tabular-nums flex items-center justify-center"
                  style={{ backgroundColor: ODS[n - 1].color }}
                >
                  {n}
                </span>
              </th>
            ))}
            <th className="text-right pb-1 font-medium">Temps</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(e => (
            <tr key={e.id} className="border-b border-amber-100 last:border-0">
              <td className="py-1.5">
                <span className="flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: e.modelColor }} />
                  <span className="text-stone-700 font-medium">{e.modelName}</span>
                </span>
              </td>
              {odsList.map(n => (
                <td key={n} className="text-center py-1.5">
                  {e.predicted.includes(n) ? (
                    <span className="text-emerald-600 font-bold">●</span>
                  ) : (
                    <span className="text-stone-300">○</span>
                  )}
                </td>
              ))}
              <td className="text-right text-stone-600 font-mono py-1.5">{e.compute.simulated_inference_ms.toFixed(0)}ms</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function PredictPage() {
  const [text, setText] = useState('');
  const [model, setModel] = useState('mmbert');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [showExamples, setShowExamples] = useState(false);

  const examples = [
    {
      title: "Concurs d'obres de renovació urbana",
      text: "Anunci de licitació per a la contractació de les obres de renovació de la pavimentació i mobiliari urbà del carrer Major del municipi. L'objectiu és millorar l'accessibilitat i la qualitat de l'espai públic. Pressupost: 425.000 euros. Termini: 6 mesos.",
    },
    {
      title: 'Subvencions per a entitats socials',
      text: "Convocatòria de subvencions destinades a entitats sense ànim de lucre que treballen amb col·lectius vulnerables. Es prioritzen projectes de lluita contra la pobresa, inserció laboral de persones en situació de risc i atenció a la infància. Dotació total: 150.000 euros.",
    },
    {
      title: "Aprovació pla d'energia renovable",
      text: "Aprovació definitiva del Pla d'Acció per a l'Energia Sostenible i el Clima (PAESC) del municipi. El document estableix mesures per reduir un 40% les emissions de CO2 abans del 2030, instal·lar plaques fotovoltaiques en equipaments municipals i fomentar la mobilitat elèctrica.",
    },
  ];

  async function predict() {
    if (!text.trim()) {
      setError("Cal introduir el text d'un anunci.");
      return;
    }
    setLoading(true);
    setError(null);

    const startTime = performance.now();

    try {
      const result = await callModelAPI(text, model);
      const elapsedMs = performance.now() - startTime;

      const baseInfMs = COMPUTE[model].inf_per_sample_ms;
      const simulatedMs = baseInfMs * (0.85 + Math.random() * 0.3);

      const entry = {
        id: Date.now(),
        text: text.trim(),
        textPreview: text.trim().slice(0, 140) + (text.trim().length > 140 ? '…' : ''),
        model,
        modelName: MODEL_INFO[model].name,
        modelColor: MODEL_INFO[model].color,
        predicted: result.predicted,
        confidences: result.confidences,
        reasoning: result.reasoning,
        compute: {
          api_elapsed_ms: elapsedMs,
          simulated_inference_ms: simulatedMs,
          ram_mb: COMPUTE[model].ram_inf_mb,
          model_size_mb: COMPUTE[model].size_mb,
        },
        timestamp: new Date(),
      };
      setHistory(h => [entry, ...h]);
    } catch (e) {
      console.error(e);
      setError('Hi ha hagut un error en la predicció. Torna-ho a provar.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-6xl mx-auto px-6 md:px-10 py-12 md:py-16">
      <div className="mb-10">
        <div className="text-xs uppercase tracking-[0.25em] text-stone-500 mb-4">
          <span className="font-mono">03</span> &nbsp;·&nbsp; Predicció en viu
        </div>
        <h1 className="font-serif text-4xl md:text-5xl leading-tight text-stone-900 mb-6">
          Enganxa un anunci,
          <span className="text-stone-400"> tria un model,</span>
          <br />
          mira quins ODS prediu.
        </h1>
        <p className="text-stone-600 text-lg leading-relaxed">
          Obre el <a className="underline hover:text-stone-900" href="https://bop.diba.cat/cercador-butlletins" target="_blank" rel="noreferrer">cercador del BOPB</a>, copia el text d'un anunci real i prova-hi els quatre
          models. L'historial de la sessió guarda les prediccions perquè puguis comparar-les.
        </p>
      </div>

      <div className="grid lg:grid-cols-12 gap-8">

        {/* Input column */}
        <div className="lg:col-span-7">
          <div className="bg-white border border-stone-200 rounded-lg overflow-hidden">
            <div className="px-5 py-3 border-b border-stone-200 flex items-center justify-between bg-stone-50">
              <span className="text-xs uppercase tracking-wider text-stone-600 font-medium">Text de l'anunci</span>
              <button
                onClick={() => setShowExamples(!showExamples)}
                className="text-xs text-stone-500 hover:text-stone-900 underline"
              >
                {showExamples ? 'Amagar' : 'Mostrar'} exemples
              </button>
            </div>
            {showExamples && (
              <div className="px-5 py-3 border-b border-stone-200 bg-amber-50/50 space-y-2">
                {examples.map((ex, i) => (
                  <button
                    key={i}
                    onClick={() => { setText(ex.text); setShowExamples(false); }}
                    className="block w-full text-left text-sm text-stone-700 hover:text-stone-900 py-1.5 px-2 hover:bg-white rounded transition"
                  >
                    <span className="font-medium text-stone-900">{ex.title}</span>
                    <span className="text-stone-500"> — {ex.text.slice(0, 80)}…</span>
                  </button>
                ))}
              </div>
            )}
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Enganxa aquí el text d'un anunci del BOPB…"
              className="w-full p-5 h-72 text-sm text-stone-800 font-mono leading-relaxed resize-none focus:outline-none"
              spellCheck={false}
            />
            <div className="px-5 py-2 border-t border-stone-200 bg-stone-50 text-xs text-stone-500 flex justify-between">
              <span>{text.trim().split(/\s+/).filter(Boolean).length} paraules · {text.length} caràcters</span>
              {text.length > 0 && (
                <button onClick={() => setText('')} className="hover:text-stone-900">Esborrar</button>
              )}
            </div>
          </div>

          {/* Model picker */}
          <div className="mt-6">
            <div className="text-xs uppercase tracking-wider text-stone-600 font-medium mb-3">Model de predicció</div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {MODEL_ORDER.map(k => {
                const m = MODEL_INFO[k];
                const isActive = model === k;
                return (
                  <button
                    key={k}
                    onClick={() => setModel(k)}
                    className={`text-left p-3 rounded-lg border transition ${isActive ? 'border-stone-900 bg-stone-50' : 'border-stone-200 hover:border-stone-400'}`}
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <span className="w-2 h-2 rounded-full" style={{ backgroundColor: m.color }} />
                      <span className="font-medium text-sm text-stone-900">{m.name}</span>
                    </div>
                    <div className="text-[10px] text-stone-500 uppercase tracking-wider">{m.family}</div>
                    <div className="text-xs text-stone-600 mt-1 tabular-nums">
                      F1-macro {METRICS[k].f1_macro.toFixed(3)}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Predict button */}
          <button
            onClick={predict}
            disabled={loading || !text.trim()}
            className="mt-6 w-full bg-stone-900 text-white py-4 rounded-lg font-medium hover:bg-stone-800 disabled:bg-stone-300 disabled:cursor-not-allowed transition flex items-center justify-center gap-2"
          >
            {loading ? (
              <>
                <span className="animate-pulse">●</span>
                <span>{MODEL_INFO[model].name} està prediant…</span>
              </>
            ) : (
              <span>Predir ODS amb {MODEL_INFO[model].name}</span>
            )}
          </button>

          {error && (
            <div className="mt-4 bg-rose-50 border border-rose-200 text-rose-900 px-4 py-3 rounded-lg text-sm">
              {error}
            </div>
          )}

          {history.length > 0 && (
            <LatestResult entry={history[0]} />
          )}
        </div>

        {/* History column */}
        <div className="lg:col-span-5">
          <div className="sticky top-24">
            <div className="flex items-center justify-between mb-4">
              <h2 className="font-serif text-2xl text-stone-900">Historial de sessió</h2>
              {history.length > 0 && (
                <button onClick={() => setHistory([])} className="text-xs text-stone-500 hover:text-stone-900 underline">
                  Esborrar
                </button>
              )}
            </div>
            {history.length === 0 ? (
              <div className="bg-stone-50 border border-dashed border-stone-300 rounded-lg p-8 text-center">
                <p className="text-sm text-stone-500">
                  Les prediccions apareixeran aquí.
                  <br />
                  L'historial es manté durant la sessió actual.
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                {history.map(h => <HistoryEntry key={h.id} entry={h} />)}
              </div>
            )}

            {history.filter(h => h.text === history[0]?.text).length > 1 && (
              <div className="mt-6 bg-amber-50 border border-amber-200 rounded-lg p-4">
                <div className="text-xs uppercase tracking-wider text-amber-700 mb-2 font-medium">Comparativa multi-model</div>
                <ComparisonView entries={history.filter(h => h.text === history[0]?.text)} />
              </div>
            )}
          </div>
        </div>

      </div>
    </div>
  );
}
