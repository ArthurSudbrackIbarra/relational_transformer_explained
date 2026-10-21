/**
 * App shell (BUILD_PLAN §5): sticky top bar (title, simulation indicator,
 * theme toggle, paper & code links), left chapter sidebar ≥ 1024px that
 * becomes a drawer below, and the content column (prose ~70ch, figures up to
 * 1200px). The landing page hides the fixed sidebar (the drawer stays available).
 */
import { BookOpen, Code2, Menu, X } from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import { useEffect, useRef, useState, type ReactNode } from 'react';
import { Link, Outlet, useLocation } from 'react-router-dom';
import { getChapter } from '@/chapters/registry';
import { IconButton } from '../ui/controls';
import { cx } from '../ui/cx';
import { CODE, PAPER } from '../prose/sources';
import { Logo } from './Logo';
import { ChapterList } from './Sidebar';
import { SimIndicator } from './SimIndicator';
import { ThemeToggle } from './ThemeToggle';

function currentSlug(pathname: string): string | undefined {
  const slug = pathname.replace(/^\/+/, '').split('/')[0];
  return getChapter(slug) ? slug : undefined;
}

function TopBar({ onMenu, menuOpen, showMenuAlways }: { onMenu: () => void; menuOpen: boolean; showMenuAlways: boolean }) {
  return (
    <header className="sticky top-0 z-40 h-[var(--topbar-h)] border-b border-border bg-[color-mix(in_srgb,var(--page)_86%,transparent)] backdrop-blur-md">
      <div className="flex h-full items-center gap-2 px-3 sm:px-4">
        <IconButton
          label={menuOpen ? 'Close chapter menu' : 'Open chapter menu'}
          onClick={onMenu}
          aria-expanded={menuOpen}
          aria-controls="chapter-drawer"
          className={cx(!showMenuAlways && 'lg:hidden')}
        >
          {menuOpen ? <X size={18} /> : <Menu size={18} />}
        </IconButton>
        <Link to="/" data-plain className="flex min-w-0 items-center gap-2.5 rounded-md pr-1 text-ink">
          <Logo />
          <span className="truncate text-[17.5px] font-semibold tracking-[-0.01em]">
            <span className="hidden sm:inline">Relational Transformer, Step by Step</span>
            <span className="sm:hidden">RT, Step by Step</span>
          </span>
        </Link>
        <div className="ml-auto flex min-w-0 items-center gap-1 sm:gap-1.5">
          <SimIndicator className="shrink-0 lg:max-w-[440px]" />
          <ThemeToggle />
          <a
            href={PAPER.url}
            target="_blank"
            rel="noreferrer noopener"
            data-plain
            className="hidden h-9 items-center gap-1.5 rounded-lg px-2 text-[15.5px] font-medium text-ink-2 hover:bg-surface-2 hover:text-ink sm:inline-flex"
            title={`${PAPER.authors}, ${PAPER.title} (${PAPER.venue})`}
          >
            <BookOpen aria-hidden="true" size={16} />
            <span className="hidden xl:inline">Paper</span>
            <span className="sr-only xl:hidden">Paper</span>
          </a>
          <a
            href={`${CODE.url}/tree/${CODE.commitFull}`}
            target="_blank"
            rel="noreferrer noopener"
            data-plain
            className="hidden h-9 items-center gap-1.5 rounded-lg px-2 text-[15.5px] font-medium text-ink-2 hover:bg-surface-2 hover:text-ink sm:inline-flex"
            title={`${CODE.repo} @${CODE.commit} (paper-era code)`}
          >
            <Code2 aria-hidden="true" size={16} />
            <span className="hidden xl:inline">Code</span>
            <span className="sr-only xl:hidden">Code</span>
          </a>
        </div>
      </div>
    </header>
  );
}

function Drawer({ open, onClose, slug }: { open: boolean; onClose: () => void; slug?: string }) {
  const panel = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKey);
    const t = window.setTimeout(() => panel.current?.querySelector<HTMLElement>('a[aria-current="page"], a')?.focus(), 30);
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.removeEventListener('keydown', onKey);
      window.clearTimeout(t);
      document.body.style.overflow = prev;
    };
  }, [open, onClose]);
  return (
    <AnimatePresence>
      {open && (
        <div className="fixed inset-0 z-50" id="chapter-drawer">
          <motion.div
            className="absolute inset-0 bg-black/30"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.18 }}
            onClick={onClose}
            aria-hidden="true"
          />
          <motion.div
            ref={panel}
            role="dialog"
            aria-modal="true"
            aria-label="Chapters"
            className="scroll-quiet absolute top-0 bottom-0 left-0 w-[min(320px,88vw)] overflow-y-auto border-r border-border bg-page px-2 pt-3 pb-8 shadow-[var(--shadow-pop)]"
            initial={{ x: -24, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: -24, opacity: 0 }}
            transition={{ duration: 0.2, ease: [0.2, 0.8, 0.2, 1] }}
          >
            <div className="mb-3 flex items-center justify-between px-2">
              <Link to="/" onClick={onClose} data-plain className="flex items-center gap-2 text-[16.5px] font-semibold text-ink">
                <Logo size={20} /> Home
              </Link>
              <IconButton label="Close chapter menu" onClick={onClose}>
                <X size={18} />
              </IconButton>
            </div>
            <ChapterList onNavigate={onClose} currentSlug={slug} />
            <div className="mt-4 flex flex-col gap-1 border-t border-grid px-1 pt-4">
              <a
                href={PAPER.url}
                target="_blank"
                rel="noreferrer noopener"
                data-plain
                className="flex items-center gap-2 rounded-lg px-2 py-2 text-[16.5px] text-ink-2 hover:bg-surface-2 hover:text-ink"
              >
                <BookOpen aria-hidden="true" size={16} /> Paper (arXiv {PAPER.arxiv})
              </a>
              <a
                href={`${CODE.url}/tree/${CODE.commitFull}`}
                target="_blank"
                rel="noreferrer noopener"
                data-plain
                className="flex items-center gap-2 rounded-lg px-2 py-2 text-[16.5px] text-ink-2 hover:bg-surface-2 hover:text-ink"
              >
                <Code2 aria-hidden="true" size={16} /> Code @{CODE.commit}
              </a>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}

export interface LayoutProps {
  /** Override the routed content (defaults to <Outlet/>). */
  children?: ReactNode;
}

/** The app shell. Use as a layout route element. */
export function Layout({ children }: LayoutProps) {
  const { pathname } = useLocation();
  const slug = currentSlug(pathname);
  const isLanding = pathname === '/' || pathname === '';
  const [drawer, setDrawer] = useState(false);

  // close the drawer on navigation
  useEffect(() => setDrawer(false), [pathname]);

  const withSidebar = !isLanding;

  return (
    <div className="min-h-screen bg-page">
      <a
        href="#main"
        onClick={(e) => {
          e.preventDefault();
          document.getElementById('main')?.focus();
        }}
        className="sr-only-focusable fixed top-2 left-2 z-[60] rounded-md bg-accent px-3 py-2 text-[15.5px] font-medium text-accent-ink"
      >
        Skip to content
      </a>
      <TopBar onMenu={() => setDrawer((d) => !d)} menuOpen={drawer} showMenuAlways={!withSidebar} />
      <Drawer open={drawer} onClose={() => setDrawer(false)} slug={slug} />
      <div className="flex">
        {withSidebar && (
          <div className="hidden w-[var(--sidebar-w)] shrink-0 border-r border-border lg:block">
            <aside
              className="scroll-quiet sticky top-[var(--topbar-h)] h-[calc(100vh-var(--topbar-h))] overflow-y-auto px-2 pt-6 pb-10"
              aria-label="Course navigation"
            >
              <ChapterList currentSlug={slug} />
            </aside>
          </div>
        )}
        <main id="main" tabIndex={-1} className="min-w-0 flex-1 outline-none">
          {children ?? <Outlet />}
        </main>
      </div>
    </div>
  );
}

/**
 * Content column wrapper for routed pages: centres content up to 1200px with a
 * 16px gutter on phones. Chapters use it via <ChapterPage>.
 */
export function ContentColumn({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cx('mx-auto w-full max-w-[var(--content-max)] px-4 sm:px-6 lg:px-10', className)}>{children}</div>;
}
