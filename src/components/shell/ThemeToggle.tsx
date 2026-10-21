/**
 * Light/dark toggle. The first choice defaults to the OS (prefers-color-scheme);
 * clicking stamps the opposite of what is on screen and persists it (store →
 * localStorage in try/catch). Shift-click returns to "follow the system".
 */
import { Moon, Sun } from 'lucide-react';
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
