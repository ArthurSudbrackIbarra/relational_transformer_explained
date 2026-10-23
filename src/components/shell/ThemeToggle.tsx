/**
 * Light/dark toggle. The first choice defaults to the OS (prefers-color-scheme);
 * clicking stamps the opposite of what is on screen and persists it (store →
 * localStorage in try/catch). Shift-click returns to "follow the system".
 */
import { BookOpenText, Moon, Sun } from 'lucide-react';
import { useSim } from '@/state/store';
import { useResolvedTheme } from '@/styles/theme';
import { IconButton } from '../ui/controls';

export function ThemeToggle() {
  const setTheme = useSim((s) => s.setTheme);
  const resolved = useResolvedTheme();
  const next = resolved === 'dark' ? 'light' : 'dark';
  return (
    <IconButton
      label={`Switch to ${next} theme (shift-click: follow system)`}
      onClick={(e) => setTheme(e.shiftKey ? null : next)}
    >
      {resolved === 'dark' ? <Sun size={17} /> : <Moon size={17} />}
    </IconButton>
  );
}

/** Reading mode: Libron serif on a paper-toned page (index.css). Off by default; works in both themes. */
export function ReadingToggle() {
  const reading = useSim((s) => s.reading);
  const setReading = useSim((s) => s.setReading);
  return (
    <IconButton
      label={reading ? 'Turn off reading mode' : 'Turn on reading mode (book page)'}
      aria-pressed={reading}
      active={reading}
      onClick={() => setReading(!reading)}
    >
      <BookOpenText size={17} />
    </IconButton>
  );
}
