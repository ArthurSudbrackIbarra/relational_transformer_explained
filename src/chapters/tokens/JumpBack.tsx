/**
 * "Back to Step N" after an "inspect in Step 1" link: the links scroll thousands of pixels up, so
 * this floating button returns the reader to where they were. It goes away when used, when
 * dismissed, or when the reader scrolls back near that place on their own.
 */
import { ArrowDown, X } from 'lucide-react';
import { AnimatePresence, motion, useReducedMotion } from 'motion/react';
import { useEffect } from 'react';
import type { TokensLive } from './live';

export function JumpBack({ live }: { live: TokensLive }) {
  const { jump, jumpBack, dismissJump } = live;
  const reduce = useReducedMotion();

  useEffect(() => {
    if (!jump) return;
    let left = false; // the outbound scroll starts at jump.y; arm only once the page has moved away
    const onScroll = () => {
      const dist = Math.abs(window.scrollY - jump.y);
      if (dist > 600) left = true;
      else if (left && dist < 240) dismissJump();
    };
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && dismissJump();
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('keydown', onKey);
    };
  }, [jump, dismissJump]);

  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-5 z-40 flex justify-center px-4">
      <AnimatePresence>
        {jump && (
        <motion.div
          key={jump.seq}
          initial={reduce ? false : { opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          exit={reduce ? undefined : { opacity: 0, y: 16 }}
          transition={{ duration: 0.2 }}
          className="pointer-events-auto flex items-center gap-1 rounded-full border border-border bg-surface p-1 shadow-[var(--shadow-pop)]"
          role="region"
          aria-label="Return to where you were reading"
        >
          <button
            type="button"
            onClick={jumpBack}
            className="inline-flex h-9 items-center gap-1.5 rounded-full bg-accent px-3.5 text-[15.5px] font-medium whitespace-nowrap text-accent-ink hover:opacity-90"
          >
            <ArrowDown aria-hidden="true" size={16} /> {jump.label}
          </button>
          <button
            type="button"
            onClick={dismissJump}
            aria-label="Dismiss"
            className="inline-flex h-9 w-9 items-center justify-center rounded-full text-ink-2 hover:bg-surface-2 hover:text-ink"
          >
            <X aria-hidden="true" size={16} />
          </button>
        </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
