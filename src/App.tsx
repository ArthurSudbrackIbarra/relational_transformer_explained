/**
 * Router + shell (BUILD_PLAN §1, §6). HashRouter so the static build works
 * from any path:
 *   /#/                 Landing
 *   /#/<slug>           chapters (lazy, via src/chapters/registry.ts)
 *   /#/qa, /#/qa/<id>   questions & answers (lazy, src/qa)
 *   /#/dev/components   component gallery (not in the navigation)
 *   anything else       NotFound
 */
import { MotionConfig } from 'motion/react';
import { lazy, Suspense, useEffect, useLayoutEffect, type ReactNode } from 'react';
import { HashRouter, Route, Routes, useLocation, useParams } from 'react-router-dom';
import { getChapterComponent } from '@/chapters/registry';
import { Layout } from '@/components/shell/Layout';
import { ErrorBoundary } from '@/components/ui/ErrorBoundary';
import { Landing } from '@/pages/Landing';
import { NotFound } from '@/pages/NotFound';
import { useSim } from '@/state/store';
import { applyReading, applyTheme } from '@/styles/theme';

const DevGallery = lazy(() => import('@/pages/DevGallery').then((m) => ({ default: m.DevGallery })));
const QAIndex = lazy(() => import('@/qa/QAPages').then((m) => ({ default: m.QAIndex })));
const QAPage = lazy(() => import('@/qa/QAPages').then((m) => ({ default: m.QAPage })));

function Lazy({ children, resetKey }: { children: ReactNode; resetKey?: string }) {
  return (
    <ErrorBoundary resetKey={resetKey}>
      <Suspense fallback={<PageLoading />}>{children}</Suspense>
    </ErrorBoundary>
  );
}

/** Keeps <html data-theme> and <html data-reading> in sync with the store. */
function ThemeSync() {
  const theme = useSim((s) => s.theme);
  const reading = useSim((s) => s.reading);
  useLayoutEffect(() => applyTheme(theme), [theme]);
  useLayoutEffect(() => applyReading(reading), [reading]);
  return null;
}

/** Scroll to the top on route changes (not on ?section= changes within a chapter). */
function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: 'instant' as ScrollBehavior });
  }, [pathname]);
  return null;
}

function PageLoading() {
  return (
    <div className="mx-auto max-w-[var(--content-max)] px-4 pt-16 sm:px-6 lg:px-10" aria-busy="true" aria-live="polite">
      <div className="mx-auto max-w-[var(--prose-max)]">
        <div className="h-3 w-28 animate-pulse rounded bg-surface-3" />
        <div className="mt-4 h-9 w-3/4 animate-pulse rounded bg-surface-3" />
        <div className="mt-5 h-4 w-full animate-pulse rounded bg-surface-2" />
        <div className="mt-2 h-4 w-5/6 animate-pulse rounded bg-surface-2" />
        <span className="sr-only">Loading…</span>
      </div>
    </div>
  );
}

function ChapterRoute() {
  const { slug = '' } = useParams();
  const markVisited = useSim((s) => s.markVisited);
  const Chapter = getChapterComponent(slug);
  useEffect(() => {
    if (Chapter) markVisited(slug);
  }, [slug, Chapter, markVisited]);
  if (!Chapter) return <NotFound />;
  return (
    <ErrorBoundary resetKey={slug}>
      <Suspense fallback={<PageLoading />}>
        <Chapter />
      </Suspense>
    </ErrorBoundary>
  );
}

function QARoute() {
  const { qid = '' } = useParams();
  return (
    <Lazy resetKey={qid}>
      <QAPage />
    </Lazy>
  );
}

export default function App() {
  return (
    <HashRouter>
      <MotionConfig reducedMotion="user">
        <ThemeSync />
        <ScrollToTop />
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<Landing />} />
            <Route
              path="dev/components"
              element={
                <Lazy>
                  <DevGallery />
                </Lazy>
              }
            />
            <Route
              path="qa"
              element={
                <Lazy>
                  <QAIndex />
                </Lazy>
              }
            />
            <Route path="qa/:qid" element={<QARoute />} />
            <Route path=":slug" element={<ChapterRoute />} />
            <Route path="*" element={<NotFound />} />
          </Route>
        </Routes>
      </MotionConfig>
    </HashRouter>
  );
}
