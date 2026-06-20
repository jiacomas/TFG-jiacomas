import React, { useState, useEffect } from 'react';
import IntroPage from './pages/IntroPage';
import ModelsPage from './pages/ModelsPage';
import PredictPage from './pages/PredictPage';
import { ODS } from './constants';

export default function App() {
  const [page, setPage] = useState('intro');
  const [edaData, setEdaData] = useState(null);

  useEffect(() => {
    fetch('/eda_data.json')
      .then(r => r.ok ? r.json() : null)
      .then(data => { if (data) setEdaData(data); })
      .catch(() => { });
  }, []);

  // Inject Google Fonts
  useEffect(() => {
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = 'https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap';
    document.head.appendChild(link);
    return () => { document.head.removeChild(link); };
  }, []);

  const navItems = [
    { key: 'intro', label: 'Introducció', num: '01' },
    { key: 'models', label: 'Models', num: '02' },
    { key: 'predict', label: 'Predir', num: '03' },
  ];

  return (
    <div className="min-h-screen bg-stone-50 text-stone-900" style={{ fontFamily: "'Inter', system-ui, -apple-system, sans-serif" }}>
      <style>{`
        body { font-family: 'Inter', system-ui, sans-serif; }
        .font-serif { font-family: 'Fraunces', Georgia, serif !important; font-variation-settings: "opsz" 144; }
        .font-mono { font-family: 'JetBrains Mono', ui-monospace, monospace !important; }
      `}</style>

      {/* Top Nav */}
      <header className="sticky top-0 z-50 bg-stone-50/85 backdrop-blur-md border-b border-stone-200">
        <div className="max-w-6xl mx-auto px-6 md:px-10 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3 min-w-0">
            <div className="grid grid-cols-3 gap-0.5 shrink-0">
              {[8, 11, 13, 3, 16, 7].map((n, i) => (
                <div key={i} className="w-1.5 h-1.5 rounded-sm" style={{ backgroundColor: ODS[n - 1].color }} />
              ))}
            </div>
            <div className="min-w-0">
              <div className="text-xs uppercase tracking-widest text-stone-500 font-medium leading-none">TFG · Enginyeria de Dades · UAB</div>
              <div className="font-serif text-base md:text-lg text-stone-900 leading-tight truncate">BOPB × ODS — Classificació multilabel</div>
            </div>
          </div>
          <nav className="flex gap-1 md:gap-2 shrink-0">
            {navItems.map(item => (
              <button
                key={item.key}
                onClick={() => setPage(item.key)}
                className={`group relative text-xs md:text-sm px-3 md:px-4 py-2 rounded font-medium transition ${page === item.key ? 'text-stone-900' : 'text-stone-500 hover:text-stone-900'}`}
              >
                <span className="font-mono text-[10px] text-stone-400 mr-1.5 hidden md:inline">{item.num}</span>
                {item.label}
                {page === item.key && (
                  <span className="absolute -bottom-[17px] left-3 right-3 h-0.5 bg-stone-900" />
                )}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main>
        {page === 'intro' && <IntroPage edaData={edaData} />}
        {page === 'models' && <ModelsPage />}
        {page === 'predict' && <PredictPage />}
      </main>

      {/* Footer */}
      <footer className="border-t border-stone-200 mt-20">
        <div className="max-w-6xl mx-auto px-6 md:px-10 py-10 grid md:grid-cols-3 gap-8 text-sm">
          <div>
            <div className="text-xs uppercase tracking-widest text-stone-500 mb-2">Autor</div>
            <div className="font-serif text-lg text-stone-900">Jia Chun Comas Frigola</div>
            <div className="text-stone-500 text-xs mt-1">Tutor: Antonio Lozano · Curs 2025/2026</div>
          </div>
          <div>
            <div className="text-xs uppercase tracking-widest text-stone-500 mb-2">Institució</div>
            <div className="text-stone-700">Grau en Enginyeria de Dades</div>
            <div className="text-stone-500 text-xs">Escola d'Enginyeria · Universitat Autònoma de Barcelona</div>
          </div>
          <div>
            <div className="text-xs uppercase tracking-widest text-stone-500 mb-2">Dades</div>
            <a href="https://bop.diba.cat/cercador-butlletins" target="_blank" rel="noreferrer" className="text-stone-700 hover:text-stone-900 underline block">Cercador del BOPB</a>
            <a href="https://bop.diba.cat/ca/ods" target="_blank" rel="noreferrer" className="text-stone-700 hover:text-stone-900 underline block">ODS al BOPB</a>
          </div>
        </div>
        <div className="border-t border-stone-200 py-4">
          <div className="max-w-6xl mx-auto px-6 md:px-10 flex justify-between text-[10px] text-stone-400 font-mono">
            <span>Comparació models ML vs. DL</span>
            <span>Juny 2026</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
