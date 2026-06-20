import React from 'react';
import { ODS } from './constants';

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
