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
import './search/expose';
import { useSim } from './state/store';
import { applyReading, applyTheme } from './styles/theme';

// zustand's persist middleware hydrates synchronously from localStorage (safe adapter).
applyTheme(useSim.getState().theme);
applyReading(useSim.getState().reading);

const el = document.getElementById('root');
if (!el) throw new Error('#root element missing from index.html');

createRoot(el).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
