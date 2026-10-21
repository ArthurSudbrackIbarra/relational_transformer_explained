/**
 * The zero vector, typeset with KaTeX as a bold 0. A combining arrow (U+20D7) over a monospace 0
 * draws as a hook in the site's mono font, so the chapter never writes it as text.
 */
import { TeX } from '@/components';

export function ZeroVec({ className }: { className?: string }) {
  return (
    <span className={className} aria-label="the zero vector">
      <TeX tex={String.raw`\mathbf{0}`} />
    </span>
  );
}
