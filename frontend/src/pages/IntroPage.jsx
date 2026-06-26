import React, { useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts';
import { ODS, ODS_DETAILS, ODS_FREQ_DEFAULT } from '../constants';
import { Stat, SectionTitle, Fmt } from '../components';

// Fallback co-occurrence matrix used before eda_data.json loads
const M_DEFAULT = [
  [0, 255, 474, 599, 243, 32, 75, 477, 59, 1398, 795, 71, 137, 5, 66, 234, 416],
  [255, 0, 244, 289, 102, 37, 35, 358, 57, 360, 243, 89, 118, 7, 79, 123, 153],
  [474, 244, 0, 654, 296, 101, 107, 721, 1455, 887, 2318, 120, 378, 12, 197, 402, 578],
  [599, 289, 654, 0, 303, 50, 67, 1080, 118, 1034, 751, 119, 232, 10, 129, 479, 623],
  [243, 102, 296, 303, 0, 28, 51, 374, 113, 642, 334, 79, 184, 7, 51, 244, 238],
  [32, 37, 101, 50, 28, 0, 58, 155, 500, 49, 562, 44, 140, 15, 110, 67, 102],
  [75, 35, 107, 67, 51, 58, 0, 126, 613, 95, 773, 71, 736, 7, 45, 56, 130],
  [477, 358, 721, 1080, 374, 155, 126, 0, 369, 1074, 1893, 237, 585, 16, 360, 1312, 1109],
  [59, 57, 1455, 118, 113, 500, 613, 369, 0, 110, 2438, 83, 763, 5, 110, 122, 277],
  [1398, 360, 887, 1034, 642, 49, 95, 1074, 110, 0, 1257, 123, 266, 14, 124, 628, 787],
  [795, 243, 2318, 751, 334, 562, 773, 1893, 2438, 1257, 0, 320, 1550, 24, 726, 900, 924],
  [71, 89, 120, 119, 79, 44, 71, 237, 83, 123, 320, 0, 242, 11, 68, 91, 165],
  [137, 118, 378, 232, 184, 140, 736, 585, 763, 266, 1550, 242, 0, 21, 751, 228, 356],
  [5, 7, 12, 10, 7, 15, 7, 16, 5, 14, 24, 11, 21, 0, 20, 13, 10],
  [66, 79, 197, 129, 51, 110, 45, 360, 110, 124, 726, 68, 751, 20, 0, 122, 170],
  [234, 123, 402, 479, 244, 67, 56, 1312, 122, 628, 900, 91, 228, 13, 122, 0, 619],
  [416, 153, 578, 623, 238, 102, 130, 1109, 277, 787, 924, 165, 356, 10, 170, 619, 0],
];

function CooccurrenceHeatmap({ matrix }) {
  const M = matrix ?? M_DEFAULT;
  const [hovered, setHovered] = React.useState({ row: null, col: null });

  const max = Math.max(...M.flatMap((row, i) => row.filter((_, j) => i !== j)));

  const color = (v) => {
    if (v === 0) return '#f5f5f4';

    const t = Math.min(1, Math.sqrt(v / max));
    const r = Math.round(240 - 210 * t);
    const g = Math.round(249 - 191 * t);
    const b = Math.round(255 - 117 * t);

    return `rgb(${r},${g},${b})`;
  };

  return (
    <div className="bg-white border border-stone-200 rounded-lg p-4 overflow-x-auto">
      <div className="inline-block min-w-full">
        <div
          className="grid gap-0.5"
          style={{
            gridTemplateColumns: "40px repeat(17, minmax(36px, 1fr))",
          }}
        >
          <div />

          {/* Column headers */}
          {ODS.map((o, j) => (
            <div
              key={`h${o.n}`}
              className="text-[10px] tabular-nums text-center pb-1 rounded transition-all duration-150 font-medium"
              style={{
                color: hovered.col === j ? "#264653" : "#78716c",
                backgroundColor:
                  hovered.col === j ? "rgba(38,70,83,0.08)" : "transparent",
              }}
            >
              {o.n}
            </div>
          ))}

          {M.map((row, i) => (
            <React.Fragment key={`r${i}`}>
              {/* Row header */}
              <div
                className="text-[10px] tabular-nums flex items-center justify-end pr-1.5 rounded transition-all duration-150 font-medium"
                style={{
                  color: hovered.row === i ? "#264653" : "#78716c",
                  backgroundColor:
                    hovered.row === i ? "rgba(38,70,83,0.08)" : "transparent",
                }}
              >
                ODS {i + 1}
              </div>

              {/* Cells */}
              {row.map((v, j) => {
                const active =
                  hovered.row === null ||
                  hovered.row === i ||
                  hovered.col === j;

                return (
                  <div
                    key={`c${i}-${j}`}
                    onMouseEnter={() => setHovered({ row: i, col: j })}
                    onMouseLeave={() =>
                      setHovered({ row: null, col: null })
                    }
                    className="aspect-square flex items-center justify-center text-[9px] tabular-nums rounded-sm transition-all duration-150"
                    style={{
                      backgroundColor:
                        i === j ? "#f5f5f4" : color(v),

                      color:
                        i === j
                          ? "#a8a29e"
                          : v > 800
                            ? "#fff"
                            : "#44403c",

                      opacity: active ? 1 : 0.3,

                      boxShadow:
                        hovered.row === i || hovered.col === j
                          ? "inset 0 0 0 1.5px rgba(38,70,83,.35)"
                          : undefined,

                      transform:
                        hovered.row === i && hovered.col === j
                          ? "scale(1.05)"
                          : "scale(1)",
                    }}
                    title={`ODS ${i + 1} ↔ ODS ${j + 1}: ${v}`}
                  >
                    {i === j ? "·" : v >= 100 ? v : ""}
                  </div>
                );
              })}
            </React.Fragment>
          ))}
        </div>
      </div>

      <div className="text-xs text-stone-500 mt-3">
        Co-ocurrències absolutes. Passeu el cursor sobre una cel·la per
        ressaltar la fila i la columna corresponents. Cel·les blanques:
        parelles amb &lt;100 co-ocurrències. Diagonal omesa.
      </div>
    </div>
  );
}

function ODSGrid() {
  const [selected, setSelected] = useState(null);

  return (
    <div>
      <div className="grid grid-cols-4 sm:grid-cols-6 gap-1.5">
        {ODS.map(o => {
          const isSelected = selected === o.n;
          const Icon = o.Icon;
          return (
            <button
              key={o.n}
              onClick={() => setSelected(isSelected ? null : o.n)}
              className={`group relative rounded overflow-hidden transition-all duration-150 cursor-pointer ${isSelected
                ? 'ring-2 ring-offset-2 ring-stone-900 scale-95'
                : 'hover:brightness-110 hover:scale-105'
                }`}
              style={{
                backgroundColor: o.color,
                aspectRatio: '1',
              }}
              title={`ODS ${o.n}: ${o.short}`}
            >
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-1 p-1.5">
                <div className="flex items-center gap-1 text-white">
                  <span className="font-bold tabular-nums text-sm leading-none">{o.n}</span>
                  {Icon && <Icon size={16} strokeWidth={1.75} className="opacity-90 shrink-0" />}
                </div>
                <span className="text-white/70 text-[8px] leading-tight text-center line-clamp-2 hidden sm:block px-0.5">{o.short}</span>
              </div>
            </button>
          );
        })}
        <div className="rounded bg-stone-100 border border-dashed border-stone-300" style={{ aspectRatio: '1' }} />
      </div>

      {selected && (() => {
        const o = ODS[selected - 1];
        const d = ODS_DETAILS[selected - 1];
        return (
          <div className="mt-3 bg-white border border-stone-200 rounded-lg p-4 text-sm transition-all duration-200">
            <div className="flex items-start gap-3">
              <div
                className="shrink-0 w-12 h-12 rounded-lg flex items-center justify-center text-white shadow-sm gap-0.5"
                style={{ backgroundColor: o.color }}
              >
                <span className="font-bold text-lg tabular-nums leading-none">{selected}</span>
              </div>
              <div className="min-w-0">
                <div className="flex items-baseline gap-2 flex-wrap mb-1.5">
                  <span className="font-semibold text-stone-600">{o.short}</span>
                </div>
                <p className="text-stone-600 text-xs leading-relaxed mb-2"><Fmt>{d.desc}</Fmt></p>
                <div className="flex flex-wrap gap-1">
                  {d.keywords.map(kw => (
                    <span
                      key={kw}
                      className="text-[10px] px-2 py-0.5 rounded-full text-white"
                      style={{ backgroundColor: o.color + 'cc' }}
                    >
                      {kw}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        );
      })()}

      {!selected && (
        <p className="text-xs text-stone-400 mt-3 text-center">
          Fes clic en un ODS per veure&apos;n la descripció
        </p>
      )}
    </div>
  );
}

export default function IntroPage({ edaData }) {
  const odsFreq = edaData?.label_counts ?? ODS_FREQ_DEFAULT;
  const freqData = odsFreq.map((v, i) => ({ ods: `ODS ${i + 1}`, count: v, color: ODS[i].color, n: i + 1 }));

  return (
    <div className="max-w-6xl mx-auto px-6 md:px-10 py-12 md:py-20">

      {/* Hero */}
      <div className="grid md:grid-cols-12 gap-8 mb-20">
        <div className="md:col-span-7">
          <div className="text-xs uppercase tracking-[0.25em] text-stone-500 mb-4">
            <span className="font-mono">01</span> &nbsp;·&nbsp; Introducció i context
          </div>
          <h1 className="font-serif text-4xl md:text-6xl leading-[1.05] text-stone-900 mb-6">
            Classificació multilabel
            d&apos;anuncis oficials del BOPB
            <span className="text-stone-400"> segons els </span>
            <span style={{ color: '#264653' }}>17 ODS</span>.
          </h1>
          <p className="text-stone-600 text-lg leading-relaxed">
            Cada any es publiquen milers d&apos;anuncis al Butlletí Oficial de la Província de Barcelona (BOPB), que aborden àmbits molt diversos de l&apos;administració pública. Aquest treball de fi de grau estudia l&apos;aplicació de tècniques d&apos;aprenentatge automàtic per classificar automàticament aquests anuncis segons els Objectius de Desenvolupament Sostenible (ODS), comparant diversos models clàssics i d&apos;aprenentatge profund.
          </p>
        </div>
        <div className="md:col-span-5">
          <div className="text-xs uppercase tracking-wider text-stone-500 mb-3 font-medium">
            Els 17 Objectius de Desenvolupament Sostenible (ODS)
          </div>
          <ODSGrid />
        </div>
      </div>

      {/* Project stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-20 py-8 border-y border-stone-200">
        <Stat label="Anuncis totals" value="35.520" sub="2022–2024" />
        <Stat label="Amb almenys 1 ODS" value="19.284" sub="54,3% del corpus" />
        <Stat label="Etiquetes ODS" value="17" sub="problema multilabel" />
        <Stat label="Models avaluats" value="4" sub={<>2 clàssics + 2 <em>Transformers</em></>} />
      </div>

      {/* Context */}
      <section className="mb-20">
        <SectionTitle kicker="Context del problema" num="1.1">
          Un repte de NLP <em>multilabel</em> sobre text administratiu en català
        </SectionTitle>
        <div className="grid md:grid-cols-2 gap-10 text-stone-700 leading-relaxed">
          <div className="space-y-4">
            <p>
              Els anuncis oficials del BOPB són actualment categoritzats per part dels organismes emissors,
              un procés que requereix intervenció manual significativa. L&apos;objectiu d&apos;aquest projecte
              és comprovar si tècniques de <em>Machine Learning</em> i <em>Deep Learning</em> poden automatitzar aquesta tasca
              amb un rendiment competitiu.
            </p>
            <p>
              El problema s&apos;emmarca en l&apos;àmbit del <strong>Processament del Llenguatge Natural (NLP)</strong>:
              s&apos;ha d&apos;analitzar contingut textual en català i predir etiquetes en un escenari amb tres
              dificultats afegides - multietiquetatge, fort desbalanceig de classes i complexitat semàntica del
              llenguatge administratiu.
            </p>
          </div>
          <div className="bg-stone-50 border border-stone-200 rounded-lg p-6">
            <h4 className="font-serif text-lg mb-3 text-stone-900">Objectius específics</h4>
            <ol className="space-y-2 text-sm text-stone-600">
              <li className="flex gap-3">
                <span className="font-mono text-stone-400">01</span>
                <span>Preparar, integrar i netejar les dades històriques (XLSX + CSV de text).</span>
              </li>
              <li className="flex gap-3">
                <span className="font-mono text-stone-400">02</span>
                <span>Implementar models clàssics (<i>Random Forest</i>, <i>XGBoost</i>) sobre TF-IDF.</span>
              </li>
              <li className="flex gap-3">
                <span className="font-mono text-stone-400">03</span>
                <span><i>Fine-tunejar</i> models <i>Transformer</i> en català (<i>BERTa</i>, <i>mmBERT</i>).</span>
              </li>
              <li className="flex gap-3">
                <span className="font-mono text-stone-400">04</span>
                <span>Comparar rendiment, velocitat i cost computacional.</span>
              </li>
            </ol>
          </div>
        </div>
      </section>

      {/* Pipeline */}
      <section className="mb-20">
        <SectionTitle kicker="Metodologia" num="1.2">
          <em>Pipeline</em> en tres fases
        </SectionTitle>
        <div className="grid md:grid-cols-3 gap-6">
          {[
            {
              title: 'Preparació de dades',
              num: '1',
              color: '#264653',
              items: [
                "Integració XLSX (metadades) + CSV (textos extrets de PDF)",
                "Reconstrucció de l'estructura multilabel per identificador",
                'Neteja base: normalització Unicode NFC, URLs, caràcters de control',
                'Normalització ML: lematització spaCy (ca_core_news_lg) + stopwords',
                'Divisió estratificada multietiqueta 70 / 15 / 15%',
              ]
            },
            {
              title: 'Modelatge',
              num: '2',
              color: '#2d6a4f',
              items: [
                'Pipeline ML: TF-IDF (5.000 termes, 1–2 grams) + RF/XGBoost OvR',
                'Pipeline DL: Tokenització Transformer (256 tokens), fine-tuning',
                'Compensació de desbalanceig: pesos per classe + MLSMOTE',
                'BCEWithLogitsLoss amb pos_weight per ODS',
                'Tuning de llindars per ODS sobre validació',
              ]
            },
            {
              title: 'Avaluació',
              num: '3',
              color: '#e76f51',
              items: [
                'F1-micro, F1-macro, F1-samples',
                'ROC-AUC i Average Precision',
                'Hamming Loss i Subset Accuracy',
                "Temps d'entrenament i d'inferència",
                'RAM pic i mida del model',
              ]
            },
          ].map((phase, i) => (
            <div key={i} className="bg-white border border-stone-200 rounded-lg p-6 relative overflow-hidden">
              <div
                className="absolute top-0 left-0 right-0 h-1"
                style={{ backgroundColor: phase.color }}
              />
              <div className="flex items-baseline gap-3 mb-4">
                <span className="font-serif text-5xl text-stone-200">{phase.num}</span>
                <h3 className="font-serif text-xl text-stone-900">{phase.title}</h3>
              </div>
              <ul className="space-y-2 text-sm text-stone-600">
                {phase.items.map((item, j) => (
                  <li key={j} className="flex gap-2">
                    <span className="text-stone-300 mt-1">›</span>
                    <span className="leading-relaxed"><Fmt>{item}</Fmt></span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </section>

      {/* EDA: label distribution */}
      <section className="mb-20">
        <SectionTitle kicker="Anàlisi exploratòria" num="1.3">
          Distribució per ODS - un fort desbalanceig
        </SectionTitle>
        <p className="text-stone-700 leading-relaxed mb-6">
          La distribució d&apos;anuncis per ODS és molt desigual. ODS 11 (ciutats sostenibles) i ODS 8 (treball
          digne) concentren més de 9.000 anuncis cadascun, mentre que ODS 14 (vida submarina) en té
          només 39 - un <em>ratio</em> de desbalanceig superior a 200×. Aquest fet té implicacions directes en
          l&apos;estratègia d&apos;entrenament.
        </p>
        <div className="bg-white border border-stone-200 rounded-lg p-6">
          <div style={{ width: '100%', height: 380 }}>
            <ResponsiveContainer>
              <BarChart data={freqData} margin={{ top: 20, right: 20, bottom: 30, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e7e5e4" />
                <XAxis dataKey="ods" tick={{ fontSize: 11 }} angle={-30} textAnchor="end" height={50} stroke="#78716c" />
                <YAxis tick={{ fontSize: 11 }} stroke="#78716c" />
                <Tooltip
                  contentStyle={{ background: '#fff', border: '1px solid #e7e5e4', fontSize: 12, borderRadius: 6 }}
                  formatter={(v) => [v.toLocaleString(), 'Anuncis']}
                />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {freqData.map((d, i) => <Cell key={i} fill={d.color} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="grid md:grid-cols-3 gap-6 mt-6">
          <div className="bg-emerald-50 border border-emerald-100 rounded p-4">
            <div className="text-xs uppercase tracking-wider text-emerald-700 mb-1">Majoritàris</div>
            <div className="text-sm text-stone-700">ODS 11, 8, 3, 10, 9 -&gt; 2.800 anuncis cadascun.</div>
          </div>
          <div className="bg-amber-50 border border-amber-100 rounded p-4">
            <div className="text-xs uppercase tracking-wider text-amber-700 mb-1">Intermedis</div>
            <div className="text-sm text-stone-700">ODS 16, 17, 4, 13, 1 -&gt; entre 1.500 i 2.500 anuncis.</div>
          </div>
          <div className="bg-rose-50 border border-rose-100 rounded p-4">
            <div className="text-xs uppercase tracking-wider text-rose-700 mb-1">Minoritaris</div>
            <div className="text-sm text-stone-700">ODS 14, 12, 6, 2, 5 -&gt; menys de 700 anuncis (ODS 14 només 39).</div>
          </div>
        </div>
      </section>

      {/* Multilabel complexity */}
      <section className="mb-20">
        <SectionTitle kicker="Complexitat multilabel" num="1.4">
          Mètriques característiques d&apos;un problema <em>multilabel</em>
        </SectionTitle>
        <div className="grid md:grid-cols-4 gap-6">
          {[
            { label: 'Label Cardinality', value: '2,15', sub: "mitjana d'ODS per anunci" },
            { label: 'Label Density', value: '0,126', sub: 'cardinalitat / 17 etiquetes' },
            { label: 'Conjunts únics', value: '1.042', sub: "combinacions d'ODS observades" },
            { label: 'Anuncis amb ≥ 2 etiquetes', value: '50,8%', sub: '9.800 anuncis multi-etiquetats' },
          ].map((s, i) => (
            <div key={i} className="bg-white border border-stone-200 rounded-lg p-5">
              <div className="text-xs uppercase tracking-widest text-stone-500 font-medium"><Fmt>{s.label}</Fmt></div>
              <div className="text-3xl font-serif text-stone-900 tabular-nums mt-1">{s.value}</div>
              <div className="text-xs text-stone-500 mt-1">{s.sub}</div>
            </div>
          ))}
        </div>
        <p className="text-stone-700 leading-relaxed mt-6">
          La densitat baixa (0,126) confirma que l&apos;espai d&apos;etiquetes és <em>esparcit</em>: cada anunci té
          només una petita fracció dels 17 ODS possibles. Combinada amb el fort desbalanceig, aquesta
          característica fa que els models <em>OvR</em> (<em>One-vs-Rest</em>) hagin d&apos;aprendre a no predir
          gairebé res la major part del temps - una trampa per al <em>recall</em> si no es compensa.
        </p>
      </section>

      {/* Co-occurrence */}
      <section className="mb-20">
        <SectionTitle kicker="Correlació entre etiquetes" num="1.5">
          Algunes parelles d&apos;ODS apareixen sistemàticament juntes
        </SectionTitle>
        <p className="text-stone-700 leading-relaxed mb-6">
          La matriu de co-ocurrència mostra dependències fortes. Les parelles més freqüents:
          <strong> ODS 3 ↔ 11</strong> (salut + ciutats: 2.318 vegades),
          <strong> ODS 9 ↔ 11</strong> (infraestructures + ciutats: 2.438),
          i <strong>ODS 11 ↔ 8</strong> (ciutats + treball: 1.893).
          Aquestes dependències evidencien que tractar les etiquetes de manera independent és subòptim.
        </p>
        <CooccurrenceHeatmap matrix={edaData?.cooccurrence_matrix} />
      </section>

      {/* Text quality */}
      <section className="mb-20">
        <SectionTitle kicker="Qualitat del text" num="1.6">
          Soroll d&apos;OCR i longitud de seqüència
        </SectionTitle>
        <div className="grid md:grid-cols-2 gap-8">
          <div className="bg-white border border-stone-200 rounded-lg p-6">
            <h4 className="font-serif text-lg mb-3 text-stone-900">Longitud del text</h4>
            <dl className="text-sm space-y-2">
              <div className="flex justify-between border-b border-stone-100 py-1">
                <dt className="text-stone-500">Mitjana de caràcters</dt>
                <dd className="tabular-nums text-stone-900">17.277</dd>
              </div>
              <div className="flex justify-between border-b border-stone-100 py-1">
                <dt className="text-stone-500">Mediana</dt>
                <dd className="tabular-nums text-stone-900">3.511</dd>
              </div>
              <div className="flex justify-between border-b border-stone-100 py-1">
                <dt className="text-stone-500">Màxim (truncat a)</dt>
                <dd className="tabular-nums text-stone-900">200.000</dd>
              </div>
              <div className="flex justify-between py-1">
                <dt className="text-stone-500">Anuncis &lt; 10 paraules</dt>
                <dd className="tabular-nums text-stone-900">5 (0,03%)</dd>
              </div>
            </dl>
            <p className="text-sm text-stone-600 mt-4 leading-relaxed">
              La variabilitat de longitud és enorme. Els models <em>Transformer</em> truncats a 256 <em>tokens</em> perdran informació en els anuncis més llargs - una limitació coneguda del projecte.
            </p>
          </div>
          <div className="bg-white border border-stone-200 rounded-lg p-6">
            <h4 className="font-serif text-lg mb-3 text-stone-900">Soroll d&apos;OCR</h4>
            <p className="text-sm text-stone-600 leading-relaxed">
              El text dels anuncis prové d&apos;extracció de PDFs, que sovint inclouen artefactes d&apos;OCR.
              Mesurant la fracció de caràcters no ASCII com a indicador indirecte de qualitat:
            </p>
            <dl className="text-sm space-y-2 mt-4">
              <div className="flex justify-between border-b border-stone-100 py-1">
                <dt className="text-stone-500">Mediana de soroll</dt>
                <dd className="tabular-nums text-stone-900">~2,2%</dd>
              </div>
              <div className="flex justify-between py-1">
                <dt className="text-stone-500">P95 (5% més sorollós)</dt>
                <dd className="tabular-nums text-stone-900">2,8%</dd>
              </div>
            </dl>
            <p className="text-sm text-stone-600 mt-4 leading-relaxed">
              El soroll es mitiga amb normalització Unicode NFC, eliminació de caràcters de control
              i URLs durant la fase de neteja.
            </p>
          </div>
        </div>
      </section>

      {/* References */}
      <section>
        <SectionTitle kicker="Fonts" num="1.7">
          Referències i recursos
        </SectionTitle>
        <ul className="text-sm text-stone-600 space-y-2">
          <li>BOPB - <a className="underline hover:text-stone-900" href="https://bop.diba.cat/ca/ods" target="_blank" rel="noreferrer">Objectius de Desenvolupament Sostenible</a></li>
          <li><a className="underline hover:text-stone-900" href="https://arxiv.org/abs/1810.04805" target="_blank" rel="noreferrer">Devlin et al. (2019), <em>BERT: Pre-training of Deep Bidirectional Transformers</em> - arXiv:1810.04805</a></li>
          <li><a className="underline hover:text-stone-900" href="https://arxiv.org/abs/2401.16549" target="_blank" rel="noreferrer">Tarekegn, Ullah &amp; Cheikh (2024), <em>Deep Learning for Multi-Label Learning: A Survey</em> - arXiv:2401.16549</a></li>
          <li><a className="underline hover:text-stone-900" href="https://arxiv.org/abs/2309.01666" target="_blank" rel="noreferrer">Alfaro, Allende-Cid &amp; Allende (2023), <em>Multilabel Text Classification with Label-Dependent Representation</em></a></li>
          <li><a className="underline hover:text-stone-900" href="https://doi.org/10.1145/1217299.1217300" target="_blank" rel="noreferrer">Tsoumakas &amp; Katakis (2007), <em>Multi-Label Classification: An Overview</em></a></li>
          <li><a className="underline hover:text-stone-900" href="https://doi.org/10.1023/A:1010933404324" target="_blank" rel="noreferrer">Breiman (2001), <em>Random Forests</em></a></li>
          <li><a className="underline hover:text-stone-900" href="https://arxiv.org/abs/1603.02754" target="_blank" rel="noreferrer">Chen &amp; Guestrin (2016), <em>XGBoost: A Scalable Tree Boosting System</em></a></li>
          <li><a className="underline hover:text-stone-900" href="https://arxiv.org/abs/2107.07253" target="_blank" rel="noreferrer">Armengol-Estapé et al. (2021), <em>Are Multilingual Models the Best Choice for Moderately Under-resourced Languages? BERTa</em></a></li>
        </ul>
      </section>
    </div>
  );
}
