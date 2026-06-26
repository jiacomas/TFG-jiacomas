import React, { useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ScatterChart, Scatter, ZAxis, Cell,
} from 'recharts';
import { ODS, F1_PER_ODS, METRICS, COMPUTE, MODEL_INFO, MODEL_ORDER, fmtDuration } from '../constants';
import { Stat, SectionTitle, Fmt } from '../components';

export default function ModelsPage() {
  const [selected, setSelected] = useState(null);

  if (selected) return <ModelDetail modelKey={selected} onBack={() => setSelected(null)} />;

  return <ModelsOverview onSelect={setSelected} />;
}

function ModelsOverview({ onSelect }) {
  const radarData = ODS.map((o, i) => ({
    ods: `ODS ${o.n}`,
    'Random Forest': F1_PER_ODS.rf[i],
    'XGBoost': F1_PER_ODS.xgb[i],
    'BERTa': F1_PER_ODS.berta[i],
    'mmBERT': F1_PER_ODS.mmbert[i],
  }));

  const f1OdsData = ODS.map((o, i) => ({
    ods: `${o.n}`,
    'Random Forest': F1_PER_ODS.rf[i],
    'XGBoost': F1_PER_ODS.xgb[i],
    'BERTa': F1_PER_ODS.berta[i],
    'mmBERT': F1_PER_ODS.mmbert[i],
    color: o.color,
  }));

  const metricNames = [
    ['f1_macro', 'F1-macro'],
    ['f1_micro', 'F1-micro'],
    ['f1_samples', 'F1-samples'],
    ['rec_macro', 'Recall macro'],
  ];
  const headlineData = metricNames.map(([k, label]) => ({
    metric: label,
    'Random Forest': METRICS.rf[k],
    'XGBoost': METRICS.xgb[k],
    'BERTa': METRICS.berta[k],
    'mmBERT': METRICS.mmbert[k],
  }));

  const tradeoff = MODEL_ORDER.map(k => ({
    name: MODEL_INFO[k].name,
    color: MODEL_INFO[k].color,
    train_min: COMPUTE[k].train_s / 60,
    f1: METRICS[k].f1_macro,
    size: COMPUTE[k].size_mb,
    key: k,
  }));

  return (
    <div className="max-w-6xl mx-auto px-6 md:px-10 py-12 md:py-20">

      {/* Hero */}
      <div className="mb-16">
        <div className="text-xs uppercase tracking-[0.25em] text-stone-500 mb-4">
          <span className="font-mono">02</span> &nbsp;·&nbsp; Models i comparativa
        </div>
        <h1 className="font-serif text-4xl md:text-5xl leading-tight text-stone-900 mb-6 max-w-4xl">
          Quatre models, dues famílies,
          <span className="text-stone-400"> un guanyador clar.</span>
        </h1>
        <p className="text-stone-600 text-lg leading-relaxed">
          S&apos;han entrenat dos models clàssics (<em>Random Forest</em>, <em>XGBoost</em>) sobre TF-IDF i dos models
          <em> Transformer</em> (<em>BERTa</em>, <em>mmBERT</em>) amb <em>fine-tuning</em>. Aquí pots comparar-los i, si en cliques un,
          aprofundir en la seva arquitectura i mètriques.
        </p>
      </div>

      {/* Model cards */}
      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4 mb-16">
        {MODEL_ORDER.map(k => {
          const m = MODEL_INFO[k];
          const metric = METRICS[k];
          return (
            <button
              key={k}
              onClick={() => onSelect(k)}
              className="text-left bg-white border border-stone-200 rounded-lg p-5 hover:border-stone-900 hover:shadow-sm transition group"
            >
              <div className="flex items-baseline justify-between mb-2">
                <span
                  className="text-xs uppercase tracking-wider px-2 py-0.5 rounded text-white font-medium"
                  style={{ backgroundColor: m.color }}
                >
                  {m.family}
                </span>
                <span className="text-xs text-stone-400 group-hover:text-stone-900 transition">&rarr;</span>
              </div>
              <h3 className="font-serif text-2xl text-stone-900 mb-2">{m.name}</h3>
              <p className="text-sm text-stone-600 leading-relaxed mb-4"><Fmt>{m.short}</Fmt></p>
              <div className="grid grid-cols-2 gap-2 pt-3 border-t border-stone-100">
                <div>
                  <div className="text-[10px] uppercase tracking-wider text-stone-500">F1-macro</div>
                  <div className="font-mono text-lg tabular-nums">{metric.f1_macro.toFixed(3)}</div>
                </div>
                <div>
                  <div className="text-[10px] uppercase tracking-wider text-stone-500">Entrenament</div>
                  <div className="font-mono text-lg tabular-nums">{fmtDuration(COMPUTE[k].train_s)}</div>
                </div>
              </div>
            </button>
          );
        })}
      </div>
      <p className="text-xs text-stone-500 -mt-12 mb-16">
        Clica una targeta per veure els detalls del model.
      </p>

      {/* Aggregated comparison */}
      <section className="mb-16">
        <SectionTitle kicker="Comparativa global" num="2.1">
          Mètriques agregades sobre el <em>test set</em>
        </SectionTitle>
        <div className="bg-white border border-stone-200 rounded-lg p-6 mb-6">
          <div style={{ width: '100%', height: 320 }}>
            <ResponsiveContainer>
              <BarChart data={headlineData} margin={{ top: 20, right: 20, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e7e5e4" />
                <XAxis dataKey="metric" tick={{ fontSize: 12 }} stroke="#78716c" />
                <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} stroke="#78716c" />
                <Tooltip
                  contentStyle={{ background: '#fff', border: '1px solid #e7e5e4', fontSize: 12, borderRadius: 6 }}
                  formatter={(v) => v.toFixed(3)}
                />
                <Legend wrapperStyle={{ fontSize: 12, paddingTop: 10 }} />
                <Bar dataKey="Random Forest" fill={MODEL_INFO.rf.color} radius={[3, 3, 0, 0]} />
                <Bar dataKey="XGBoost" fill={MODEL_INFO.xgb.color} radius={[3, 3, 0, 0]} />
                <Bar dataKey="BERTa" fill={MODEL_INFO.berta.color} radius={[3, 3, 0, 0]} />
                <Bar dataKey="mmBERT" fill={MODEL_INFO.mmbert.color} radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Detailed table */}
        <div className="bg-white border border-stone-200 rounded-lg overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-stone-50 text-stone-600 text-xs uppercase tracking-wider">
              <tr>
                <th className="text-left px-4 py-3 font-medium">Mètrica</th>
                {MODEL_ORDER.map(k => (
                  <th key={k} className="text-right px-4 py-3 font-medium">
                    <span style={{ color: MODEL_INFO[k].color }}>&#9679; </span>{MODEL_INFO[k].name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[
                ['F1-micro', 'f1_micro', 3, 'higher'],
                ['F1-macro', 'f1_macro', 3, 'higher'],
                ['F1-samples', 'f1_samples', 3, 'higher'],
                ['Precisió macro', 'prec_macro', 3, 'higher'],
                ['Recall macro', 'rec_macro', 3, 'higher'],
                ['ROC-AUC micro', 'roc_auc', 3, 'higher'],
                ['Hamming loss', 'hamming', 3, 'lower'],
                ['Subset accuracy', 'subset_acc', 3, 'higher'],
              ].map(([label, key, dec, dir]) => {
                const values = MODEL_ORDER.map(k => METRICS[k][key]);
                const best = dir === 'higher' ? Math.max(...values) : Math.min(...values);
                return (
                  <tr key={key} className="border-t border-stone-100">
                    <td className="px-4 py-2.5 text-stone-700"><Fmt>{label}</Fmt></td>
                    {MODEL_ORDER.map((k, i) => {
                      const v = values[i];
                      const isBest = v === best;
                      return (
                        <td
                          key={k}
                          className={`px-4 py-2.5 text-right font-mono tabular-nums ${isBest ? 'font-bold text-stone-900' : 'text-stone-600'}`}
                        >
                          {v.toFixed(dec)}
                          {isBest && <span className="text-emerald-600 ml-1">&#9733;</span>}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-stone-500 mt-2">
          &#9733; millor valor de la fila. Per a <em>Hamming loss</em>, més baix és millor.
        </p>
      </section>

      {/* F1 per ODS comparison */}
      <section className="mb-16">
        <SectionTitle kicker="Detall per categoria" num="2.2">
          F1-score per ODS
        </SectionTitle>
        <p className="text-stone-700 leading-relaxed mb-6">
          La diferència entre models es fa més evident en ODS minoritaris.
          <em> Random Forest</em> pràcticament no detecta ODS 14, 6 i 5. Els <em>Transformers</em> &mdash; sobretot <em>mmBERT</em> &mdash;
          aconsegueixen <em>recall</em> significatiu fins i tot en categories amb molt pocs exemples.
        </p>
        <div className="bg-white border border-stone-200 rounded-lg p-6 mb-6">
          <div style={{ width: '100%', height: 380 }}>
            <ResponsiveContainer>
              <BarChart data={f1OdsData} margin={{ top: 10, right: 10, bottom: 10, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e7e5e4" />
                <XAxis dataKey="ods" tick={{ fontSize: 11 }} stroke="#78716c" />
                <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} stroke="#78716c" />
                <Tooltip
                  contentStyle={{ background: '#fff', border: '1px solid #e7e5e4', fontSize: 12, borderRadius: 6 }}
                  formatter={(v) => v.toFixed(3)}
                  labelFormatter={(l) => `ODS ${l}`}
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="Random Forest" fill={MODEL_INFO.rf.color} />
                <Bar dataKey="XGBoost" fill={MODEL_INFO.xgb.color} />
                <Bar dataKey="BERTa" fill={MODEL_INFO.berta.color} />
                <Bar dataKey="mmBERT" fill={MODEL_INFO.mmbert.color} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Radar */}
        <div className="bg-white border border-stone-200 rounded-lg p-6">
          <h4 className="font-serif text-lg text-stone-900 mb-3">Radar &mdash; cobertura per ODS</h4>
          <div style={{ width: '100%', height: 480 }}>
            <ResponsiveContainer>
              <RadarChart data={radarData} margin={{ top: 20, right: 30, bottom: 20, left: 30 }}>
                <PolarGrid stroke="#e7e5e4" />
                <PolarAngleAxis dataKey="ods" tick={{ fontSize: 11, fill: '#57534e' }} />
                <PolarRadiusAxis angle={90} domain={[0, 1]} tick={false} />
                <Radar name="Random Forest" dataKey="Random Forest" stroke={MODEL_INFO.rf.color} fill={MODEL_INFO.rf.color} fillOpacity={0.05} strokeWidth={2} />
                <Radar name="XGBoost" dataKey="XGBoost" stroke={MODEL_INFO.xgb.color} fill={MODEL_INFO.xgb.color} fillOpacity={0.10} strokeWidth={2} />
                <Radar name="BERTa" dataKey="BERTa" stroke={MODEL_INFO.berta.color} fill={MODEL_INFO.berta.color} fillOpacity={0.10} strokeWidth={2} />
                <Radar name="mmBERT" dataKey="mmBERT" stroke={MODEL_INFO.mmbert.color} fill={MODEL_INFO.mmbert.color} fillOpacity={0.10} strokeWidth={2} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Tooltip contentStyle={{ background: '#fff', border: '1px solid #e7e5e4', fontSize: 12, borderRadius: 6 }} formatter={(v) => v.toFixed(3)} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>

      {/* Compute */}
      <section className="mb-16">
        <SectionTitle kicker="Cost computacional" num="2.3">
          Rendiment vs. cost &mdash; el compromís clau
        </SectionTitle>
        <p className="text-stone-700 leading-relaxed mb-6">
          La diferència en cost entre models clàssics i <em>Transformer</em> és d&apos;ordres de magnitud.
          <em> mmBERT</em> triga 300&times; més a entrenar-se que <em>XGBoost</em> per guanyar 0,054 punts en F1-macro.
          La decisió depèn de si el cas d&apos;ús prioritza precisió absoluta o eficiència.
        </p>

        <div className="grid md:grid-cols-2 gap-6 mb-6">
          {/* Trade-off scatter */}
          <div className="bg-white border border-stone-200 rounded-lg p-6">
            <h4 className="font-serif text-lg text-stone-900 mb-1">Frontera rendiment / cost</h4>
            <p className="text-xs text-stone-500 mb-3">Quart superior esquerra = ideal (alt F1, baix cost)</p>
            <div style={{ width: '100%', height: 320 }}>
              <ResponsiveContainer>
                <ScatterChart margin={{ top: 20, right: 20, bottom: 40, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e7e5e4" />
                  <XAxis
                    type="number"
                    dataKey="train_min"
                    name="Entrenament"
                    unit=" min"
                    scale="log"
                    domain={[0.5, 500]}
                    tick={{ fontSize: 11 }}
                    stroke="#78716c"
                    label={{ value: "Temps d'entrenament (min, log)", position: 'insideBottom', offset: -10, fontSize: 12, fill: '#78716c' }}
                  />
                  <YAxis
                    type="number"
                    dataKey="f1"
                    name="F1-macro"
                    domain={[0.2, 0.8]}
                    tick={{ fontSize: 11 }}
                    stroke="#78716c"
                    label={{ value: 'F1-macro', angle: -90, position: 'insideLeft', fontSize: 12, fill: '#78716c' }}
                  />
                  <Tooltip
                    contentStyle={{ background: '#fff', border: '1px solid #e7e5e4', fontSize: 12, borderRadius: 6 }}
                    cursor={{ strokeDasharray: '3 3' }}
                    formatter={(v, name) => name === 'F1-macro' ? v.toFixed(3) : v.toFixed(1)}
                    labelFormatter={() => ''}
                  />
                  <Scatter data={tradeoff}>
                    {tradeoff.map((d, i) => <Cell key={i} fill={d.color} />)}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
            </div>
            <div className="flex flex-wrap gap-3 mt-3 text-xs">
              {tradeoff.map(d => (
                <span key={d.key} className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: d.color }} />
                  <span className="text-stone-600">{d.name}</span>
                </span>
              ))}
            </div>
          </div>

          {/* RAM + Size */}
          <div className="bg-white border border-stone-200 rounded-lg p-6">
            <h4 className="font-serif text-lg text-stone-900 mb-1">Recursos</h4>
            <p className="text-xs text-stone-500 mb-3">RAM pic durant entrenament i mida del model en disc</p>
            <div style={{ width: '100%', height: 320 }}>
              <ResponsiveContainer>
                <BarChart data={MODEL_ORDER.map(k => ({
                  name: MODEL_INFO[k].name,
                  'RAM train (MB)': COMPUTE[k].ram_train_mb,
                  'Mida (MB)': COMPUTE[k].size_mb,
                  color: MODEL_INFO[k].color,
                }))} margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e7e5e4" />
                  <XAxis dataKey="name" tick={{ fontSize: 11 }} stroke="#78716c" />
                  <YAxis tick={{ fontSize: 11 }} stroke="#78716c" />
                  <Tooltip
                    contentStyle={{ background: '#fff', border: '1px solid #e7e5e4', fontSize: 12, borderRadius: 6 }}
                    formatter={(v) => `${v.toLocaleString()} MB`}
                  />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Bar dataKey="RAM train (MB)" fill="#264653" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="Mida (MB)" fill="#e76f51" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Compute table */}
        <div className="bg-white border border-stone-200 rounded-lg overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-stone-50 text-stone-600 text-xs uppercase tracking-wider">
              <tr>
                <th className="text-left px-4 py-3 font-medium">Recurs</th>
                {MODEL_ORDER.map(k => (
                  <th key={k} className="text-right px-4 py-3 font-medium">
                    <span style={{ color: MODEL_INFO[k].color }}>&#9679; </span>{MODEL_INFO[k].name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="text-stone-700">
              <tr className="border-t border-stone-100">
                <td className="px-4 py-2.5">Paràmetres</td>
                {MODEL_ORDER.map(k => <td key={k} className="px-4 py-2.5 text-right font-mono tabular-nums">{COMPUTE[k].params}</td>)}
              </tr>
              <tr className="border-t border-stone-100">
                <td className="px-4 py-2.5">Temps d&apos;entrenament</td>
                {MODEL_ORDER.map(k => <td key={k} className="px-4 py-2.5 text-right font-mono tabular-nums">{fmtDuration(COMPUTE[k].train_s)}</td>)}
              </tr>
              <tr className="border-t border-stone-100">
                <td className="px-4 py-2.5">Temps inferència (test sencer)</td>
                {MODEL_ORDER.map(k => <td key={k} className="px-4 py-2.5 text-right font-mono tabular-nums">{COMPUTE[k].inf_total_s.toFixed(1)} s</td>)}
              </tr>
              <tr className="border-t border-stone-100">
                <td className="px-4 py-2.5">Inferència per anunci</td>
                {MODEL_ORDER.map(k => <td key={k} className="px-4 py-2.5 text-right font-mono tabular-nums">{COMPUTE[k].inf_per_sample_ms.toFixed(2)} ms</td>)}
              </tr>
              <tr className="border-t border-stone-100">
                <td className="px-4 py-2.5">RAM entrenament</td>
                {MODEL_ORDER.map(k => <td key={k} className="px-4 py-2.5 text-right font-mono tabular-nums">{COMPUTE[k].ram_train_mb.toLocaleString()} MB</td>)}
              </tr>
              <tr className="border-t border-stone-100">
                <td className="px-4 py-2.5">RAM inferència</td>
                {MODEL_ORDER.map(k => <td key={k} className="px-4 py-2.5 text-right font-mono tabular-nums">{COMPUTE[k].ram_inf_mb.toLocaleString()} MB</td>)}
              </tr>
              <tr className="border-t border-stone-100">
                <td className="px-4 py-2.5">Mida en disc</td>
                {MODEL_ORDER.map(k => <td key={k} className="px-4 py-2.5 text-right font-mono tabular-nums">{COMPUTE[k].size_mb.toLocaleString()} MB</td>)}
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* Conclusions */}
      <section>
        <SectionTitle kicker="Conclusions" num="2.4">
          Quin model triar?
        </SectionTitle>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-stone-900 text-stone-100 rounded-lg p-6">
            <div className="text-xs uppercase tracking-wider text-stone-400 mb-2">Si prioritzes precisió</div>
            <h4 className="font-serif text-2xl mb-3">mmBERT</h4>
            <p className="text-sm text-stone-300 leading-relaxed">
              Millor en tots els F1, millor <em>recall</em> en classes minoritàries (ODS 14, 6, 2). La inversió
              de cost computacional està justificada per a sistemes de producció on la cobertura
              d&apos;ODS minoritaris és crítica.
            </p>
          </div>
          <div className="bg-stone-50 border border-stone-200 rounded-lg p-6">
            <div className="text-xs uppercase tracking-wider text-stone-500 mb-2">Si prioritzes eficiència</div>
            <h4 className="font-serif text-2xl mb-3 text-stone-900">XGBoost</h4>
            <p className="text-sm text-stone-700 leading-relaxed">
              500&times; més ràpid que <em>mmBERT</em> a l&apos;entrenament i pràcticament instantani en inferència,
              amb un F1-micro a només 2,5 punts del millor. Òptim per a pilots o sistemes amb
              restriccions de maquinari.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}

function ModelDetail({ modelKey, onBack }) {
  const m = MODEL_INFO[modelKey];
  const metric = METRICS[modelKey];
  const compute = COMPUTE[modelKey];
  const f1ods = ODS.map((o, i) => ({
    ods: `ODS ${o.n}`,
    F1: F1_PER_ODS[modelKey][i],
    color: o.color,
  }));

  return (
    <div className="max-w-5xl mx-auto px-6 md:px-10 py-12 md:py-16">
      <button
        onClick={onBack}
        className="text-sm text-stone-500 hover:text-stone-900 mb-8 inline-flex items-center gap-2"
      >
        &larr; Tornar a la comparativa
      </button>

      <div className="flex flex-wrap items-baseline gap-4 mb-2">
        <span
          className="text-xs uppercase tracking-wider px-2 py-1 rounded text-white font-medium"
          style={{ backgroundColor: m.color }}
        >
          {m.family}
        </span>
        <a
          href={m.paperUrl}
          target="_blank"
          rel="noreferrer"
          className="text-xs text-stone-500 font-mono underline hover:text-stone-900"
        >
          {m.paper}
        </a>
      </div>
      <h1 className="font-serif text-5xl md:text-6xl text-stone-900 mb-4">{m.name}</h1>
      <p className="text-xl text-stone-600 leading-relaxed mb-10"><Fmt>{m.short}</Fmt></p>

      {/* Headline metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-12 py-8 border-y border-stone-200">
        <Stat label="F1-macro" value={metric.f1_macro.toFixed(3)} sub={<em>test set</em>} />
        <Stat label="F1-micro" value={metric.f1_micro.toFixed(3)} sub={<em>test set</em>} />
        <Stat label="ROC-AUC micro" value={metric.roc_auc.toFixed(3)} sub={<em>test set</em>} />
        <Stat label={<em>Hamming loss</em>} value={metric.hamming.toFixed(3)} sub="(menys = millor)" />
      </div>

      {/* Background */}
      <section className="mb-12">
        <h2 className="text-xs uppercase tracking-[0.2em] text-stone-500 mb-3"><em>Background</em></h2>
        <p className="text-stone-700 leading-relaxed text-lg"><Fmt>{m.desc}</Fmt></p>
      </section>

      {/* Config */}
      <section className="mb-12">
        <h2 className="text-xs uppercase tracking-[0.2em] text-stone-500 mb-3">Configuració d&apos;entrenament</h2>
        <div className="bg-stone-50 border border-stone-200 rounded-lg p-5">
          <dl className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-2 text-sm">
            {Object.entries(m.config).map(([k, v]) => (
              <div key={k} className="flex justify-between py-1 border-b border-stone-200 last:border-0">
                <dt className="text-stone-500"><Fmt>{k}</Fmt></dt>
                <dd className="font-mono text-stone-900 text-right">{v}</dd>
              </div>
            ))}
          </dl>
        </div>
      </section>

      {/* Pros / Cons */}
      <section className="grid md:grid-cols-2 gap-6 mb-12">
        <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-6">
          <h3 className="text-xs uppercase tracking-wider text-emerald-700 mb-3">Punts forts</h3>
          <ul className="space-y-2 text-sm text-stone-700">
            {m.pros.map((p, i) => <li key={i} className="flex gap-2"><span className="text-emerald-600">&#10003;</span><span><Fmt>{p}</Fmt></span></li>)}
          </ul>
        </div>
        <div className="bg-rose-50 border border-rose-200 rounded-lg p-6">
          <h3 className="text-xs uppercase tracking-wider text-rose-700 mb-3">Limitacions</h3>
          <ul className="space-y-2 text-sm text-stone-700">
            {m.cons.map((p, i) => <li key={i} className="flex gap-2"><span className="text-rose-600">&#10007;</span><span><Fmt>{p}</Fmt></span></li>)}
          </ul>
        </div>
      </section>

      {/* Detailed metrics */}
      <section className="mb-12">
        <h2 className="text-xs uppercase tracking-[0.2em] text-stone-500 mb-3">Mètriques detallades &mdash; <em>test set</em></h2>
        <div className="bg-white border border-stone-200 rounded-lg overflow-hidden">
          <table className="w-full text-sm">
            <tbody>
              {[
                ['F1-micro', metric.f1_micro, 3],
                ['F1-macro', metric.f1_macro, 3],
                ['F1-samples', metric.f1_samples, 3],
                ['Precisió macro', metric.prec_macro, 3],
                ['Recall macro', metric.rec_macro, 3],
                ['ROC-AUC micro', metric.roc_auc, 3],
                ['Hamming loss', metric.hamming, 3],
                ['Subset accuracy', metric.subset_acc, 3],
              ].map(([label, v, dec]) => (
                <tr key={label} className="border-b border-stone-100 last:border-0">
                  <td className="px-4 py-2.5 text-stone-600"><Fmt>{label}</Fmt></td>
                  <td className="px-4 py-2.5 text-right font-mono tabular-nums text-stone-900">{v.toFixed(dec)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* F1 per ODS for this model */}
      <section className="mb-12">
        <h2 className="text-xs uppercase tracking-[0.2em] text-stone-500 mb-3">F1 per ODS</h2>
        <div className="bg-white border border-stone-200 rounded-lg p-6">
          <div style={{ width: '100%', height: 320 }}>
            <ResponsiveContainer>
              <BarChart data={f1ods} margin={{ top: 10, right: 10, bottom: 30, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e7e5e4" />
                <XAxis dataKey="ods" tick={{ fontSize: 10 }} angle={-35} textAnchor="end" height={60} stroke="#78716c" />
                <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} stroke="#78716c" />
                <Tooltip
                  contentStyle={{ background: '#fff', border: '1px solid #e7e5e4', fontSize: 12, borderRadius: 6 }}
                  formatter={(v) => v.toFixed(3)}
                />
                <Bar dataKey="F1" radius={[3, 3, 0, 0]}>
                  {f1ods.map((d, i) => <Cell key={i} fill={d.color} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>

      {/* Compute */}
      <section>
        <h2 className="text-xs uppercase tracking-[0.2em] text-stone-500 mb-3">Cost computacional</h2>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-6 bg-stone-900 text-stone-100 rounded-lg p-8">
          <div>
            <div className="text-xs uppercase tracking-wider text-stone-400">Paràmetres</div>
            <div className="font-serif text-3xl mt-1 tabular-nums">{compute.params}</div>
          </div>
          <div>
            <div className="text-xs uppercase tracking-wider text-stone-400">Entrenament</div>
            <div className="font-serif text-3xl mt-1 tabular-nums">{fmtDuration(compute.train_s)}</div>
          </div>
          <div>
            <div className="text-xs uppercase tracking-wider text-stone-400">Inferència / anunci</div>
            <div className="font-serif text-3xl mt-1 tabular-nums">{compute.inf_per_sample_ms.toFixed(1)} ms</div>
          </div>
          <div>
            <div className="text-xs uppercase tracking-wider text-stone-400">RAM (train)</div>
            <div className="font-serif text-3xl mt-1 tabular-nums">{(compute.ram_train_mb / 1024).toFixed(1)} GB</div>
          </div>
          <div>
            <div className="text-xs uppercase tracking-wider text-stone-400">RAM (inferència)</div>
            <div className="font-serif text-3xl mt-1 tabular-nums">{(compute.ram_inf_mb / 1024).toFixed(1)} GB</div>
          </div>
          <div>
            <div className="text-xs uppercase tracking-wider text-stone-400">Mida en disc</div>
            <div className="font-serif text-3xl mt-1 tabular-nums">{compute.size_mb >= 1024 ? (compute.size_mb / 1024).toFixed(2) + ' GB' : compute.size_mb + ' MB'}</div>
          </div>
        </div>
      </section>
    </div>
  );
}
