import React from 'react';
import { ODS } from './constants';

/*
 * Fmt - typographic helper.
 * Wraps foreign / English terms and model & library names in <em> (italic),
 * and code identifiers in <code> (monospace). Match is
 * case-sensitive and boundary-protected so it never breaks inside a word.
 */
const ITALIC_TERMS = [
  'rotary positional embeddings', 'Average Precision', 'Subset Accuracy', 'Subset accuracy',
  'Hamming Loss', 'Hamming loss', 'Label Cardinality', 'Label Density', 'Gradient boosting',
  'One-vs-Rest', 'Random Forest', 'Fine-tunejar', 'fine-tuning', 'fine-tuned', 'ModernBERT',
  'RoBERTa', 'Transformers', 'Transformer', 'Ensemble', 'baseline', 'Pipeline', 'pipeline',
  'stopwords', 'multilabel', 'XGBoost', 'mmBERT', 'BERTa', 'spaCy', 'Recall', 'recall', 'Tuning',
  'tuning', 'tokens', 'Flash', 'ratio', 'test set', 'Background', 'embeddings', 'batch', 'epochs',
];
const MONO_TERMS = [
  'BCEWithLogitsLoss', 'RandomForestClassifier', 'XGBClassifier', 'ca_core_news_lg', 'pos_weight',
  'class_weight', 'learning_rate', 'n_estimators', 'tree_method', 'eval_metric', 'max_length',
];

const _terms = [
  ...MONO_TERMS.map(t => ({ t, mono: true })),
  ...ITALIC_TERMS.map(t => ({ t, mono: false })),
].sort((a, b) => b.t.length - a.t.length);
const _isMono = new Map(_terms.map(o => [o.t, o.mono]));
const _esc = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const _re = new RegExp(
  '(?<![A-Za-zÀ-ÿ0-9])(' + _terms.map(o => _esc(o.t)).join('|') + ')(?![A-Za-zÀ-ÿ0-9])'
);

export function Fmt({ children }) {
  if (typeof children !== 'string') return children;
  return children.split(_re).map((part, i) =>
    _isMono.has(part)
      ? (_isMono.get(part)
        ? <code key={i} className="font-mono text-[0.92em]">{part}</code>
        : <em key={i}>{part}</em>)
      : part
  );
}

export function ODSBadge({ n, withLabel = false }) {
  const ods = ODS[n - 1];
  return (
    <span
      title={`ODS ${n}: ${ods.short}`}
      className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-white text-xs font-medium tracking-wide"
      style={{ backgroundColor: ods.color }}
    >
      <span className="tabular-nums">{n}</span>
      {withLabel && <span className="hidden md:inline">{ods.short}</span>}
    </span>
  );
}

export function Stat({ label, value, sub }) {
  return (
    <div className="border-l-2 border-stone-300 pl-4">
      <div className="text-xs uppercase tracking-widest text-stone-500 font-medium">{label}</div>
      <div className="text-3xl font-serif text-stone-900 tabular-nums mt-1">{value}</div>
      {sub && <div className="text-xs text-stone-500 mt-1">{sub}</div>}
    </div>
  );
}

export function SectionTitle({ kicker, children, num }) {
  return (
    <div className="mb-6">
      {kicker && (
        <div className="flex items-center gap-3 text-xs uppercase tracking-[0.2em] text-stone-500 mb-2">
          {num && <span className="font-mono text-stone-400">{num}</span>}
          <span>{kicker}</span>
        </div>
      )}
      <h2 className="font-serif text-3xl md:text-4xl text-stone-900 leading-tight">{children}</h2>
    </div>
  );
}
