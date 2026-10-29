/**
 * Entry point. Applies the persisted theme and reading mode before the first
 * render (no flash when the stored choice differs from the OS), loads the
 * styles and KaTeX CSS, then mounts the app.
 */
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import 'katex/dist/katex.min.css';
import './styles/index.css';
import App from './App';
import { LOGO_TARGET } from './components/shell/Logo';
import './search/expose';
import { useSim } from './state/store';
import { applyReading, applyTheme } from './styles/theme';

// zustand's persist middleware hydrates synchronously from localStorage (safe adapter).
applyTheme(useSim.getState().theme);
applyReading(useSim.getState().reading);

// Site icon (the Logo mark), injected here because index.html declares none.
if (!document.querySelector('link[rel="icon"]')) {
  const t = LOGO_TARGET;
  const svg =
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>" +
    "<rect x='1.5' y='1.5' width='9.5' height='9.5' rx='2.6' fill='%232a78d6'/>" +
    "<rect x='13' y='1.5' width='9.5' height='9.5' rx='2.6' fill='%231baf7a'/>" +
    "<rect x='1.5' y='13' width='9.5' height='9.5' rx='2.6' fill='%23e87ba4'/>" +
    "<g fill='none' stroke='%236250d6' stroke-width='1.3'>" +
    `<rect x='13.6' y='13.6' width='8.3' height='8.3' rx='2.2' stroke-dasharray='${t.dash.join(' ')}' stroke-dashoffset='${t.dashOffset}'/>` +
    `<path d='${t.question}' stroke-linecap='round' stroke-linejoin='round'/></g>` +
    `<circle cx='${t.dot.cx}' cy='${t.dot.cy}' r='${t.dot.r}' fill='%236250d6'/>` +
    '</svg>';
  const link = document.createElement('link');
  link.rel = 'icon';
  link.type = 'image/svg+xml';
  link.href = `data:image/svg+xml,${svg}`;
  document.head.appendChild(link);
}

const el = document.getElementById('root');
if (!el) throw new Error('#root element missing from index.html');

createRoot(el).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
