/**
 * Certificate of completion (/#/certificate) and its verification page (/#/certificate/verify).
 *
 * The certificate page walks the reader through three steps: tick every chapter, pass the final quiz
 * (quiz.ts), then type the name to print. The issued certificate is drawn by render.ts and downloads
 * as PDF or PNG; its ID (seal.ts) lets anyone confirm the printed details on the verify page. All of it
 * runs in the browser and is stored only there (the store's `certificate`); both pages say plainly what
 * a browser-computed ID can and cannot prove.
 */
import { Award, Check, CircleCheck, CircleX, Copy, Download, PenLine, ShieldCheck } from 'lucide-react';
import { useEffect, useState, type FormEvent, type ReactNode } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { CHAPTERS } from '@/chapters/registry';
import { Callout } from '@/components/prose/Callout';
import { SITE_TITLE, SITE_URL } from '@/components/prose/sources';
import { ChapterPage } from '@/components/shell/Chapter';
import { CompleteToggle } from '@/components/shell/Sidebar';
import { Button } from '@/components/ui/controls';
import { cx } from '@/components/ui/cx';
import { useSim, type CertificateRecord } from '@/state/store';
import { FinalQuiz, QuizReview } from './FinalQuiz';
import { FINAL_QUIZ, passScore } from './quiz';
import { canvasBlob, certificateFileName, certificatePdf, renderCertificate, saveBlob, type CertificateView } from './render';
import {
  certificateId,
  cleanName,
  formatCertDate,
  NAME_MAX,
  nameProblem,
  normalizeId,
  todayIso,
  verifyCertificate,
  verifyPath,
  verifyUrl,
  type Verdict,
} from './seal';

export function CertificatePage() {
  useEffect(() => {
    document.title = `Certificate · ${SITE_TITLE}`;
  }, []);
  const completed = useSim((s) => s.completed);
  const record = useSim((s) => s.certificate);
  const setRecord = useSim((s) => s.setCertificate);
  // the passing attempt's answers, for the review (this visit only)
  const [review, setReview] = useState<number[] | null>(null);
  const [renaming, setRenaming] = useState(false);
  const nDone = CHAPTERS.filter((c) => completed[c.slug]).length;
  const allDone = nDone === CHAPTERS.length;
  const stage = record?.name && !renaming ? 'issued' : record ? 'name' : allDone ? 'quiz' : 'chapters';
  const total = FINAL_QUIZ.length;

  return (
    <ChapterPage>
      <PageHeader title="Certificate of completion">
        Work through all {CHAPTERS.length} chapters and pass a short final quiz, and this site issues a certificate with your name, the
        date and an ID that anyone can check.
      </PageHeader>
      <div className="page-grid">
        <Steps
          steps={[
            {
              title: `Complete all ${CHAPTERS.length} chapters`,
              detail: record ? `All ${record.chapters} completed` : `${nDone} of ${CHAPTERS.length} ticked`,
              state: record || allDone ? 'done' : 'current',
            },
            {
              title: 'Pass the final quiz',
              detail: record ? `${record.score} of ${record.total} right` : `${total} questions, ${passScore(total)} right answers to pass`,
              state: record ? 'done' : allDone ? 'current' : 'upcoming',
            },
            {
              title: 'Get your certificate',
              detail: 'PDF or PNG, with an ID anyone can verify',
              state: record?.name ? 'done' : record ? 'current' : 'upcoming',
            },
          ]}
        />

        {stage === 'chapters' && <RemainingChapters />}

        {stage === 'quiz' && (
          <>
            <h2 className="mt-12 text-[26px] font-semibold tracking-[-0.015em] text-ink">Final quiz</h2>
            <p className="mt-3 mb-8 text-[18px] leading-[1.65] text-ink-2">
              One question for each stage of the course, {total} in all. Answer every question, then submit; {passScore(total)} right
              answers pass. If you miss the mark, the page names the chapters to review, and you can try again with the choices in a new
              order.
            </p>
            <FinalQuiz
              onPass={(score, n, picked) => {
                setReview(picked);
                setRecord({ name: '', date: todayIso(), score, total: n, chapters: CHAPTERS.length });
              }}
            />
          </>
        )}

        {stage === 'name' && record && (
          <NameForm
            record={record}
            renaming={renaming}
            onCancel={renaming ? () => setRenaming(false) : undefined}
            onIssue={(name) => {
              setRecord({ ...record, name });
              setRenaming(false);
            }}
          />
        )}

        {stage === 'issued' && record && <Issued record={record} onRename={() => setRenaming(true)} />}

        {review && (
          <details className="mt-12 rounded-2xl border border-border bg-surface px-5 py-4">
            <summary className="cursor-pointer text-[18px] font-semibold text-ink">Review your answers</summary>
            <div className="mt-5">
              <QuizReview picked={review} />
            </div>
          </details>
        )}

        <HowVerified />
      </div>
    </ChapterPage>
  );
}

export function VerifyPage() {
  useEffect(() => {
    document.title = `Verify a certificate · ${SITE_TITLE}`;
  }, []);
  const [params, setParams] = useSearchParams();
  const name = params.get('name') ?? '';
  const id = params.get('id') ?? '';
  const asked = name !== '' || id !== '';
  const key = `${name}\n${id}`;
  // keyed by the query, so a result never outlives the question it answers
  const [result, setResult] = useState<{ key: string; verdict: Verdict } | null>(null);
  useEffect(() => {
    if (!asked) return;
    let live = true;
    verifyCertificate(name, id).then((verdict) => live && setResult({ key, verdict }));
    return () => {
      live = false;
    };
  }, [key, asked, name, id]);
  const verdict = asked && result?.key === key ? result.verdict : null;

  return (
    <ChapterPage>
      <PageHeader title="Verify a certificate">
        Type the name and the certificate ID exactly as printed, or open the link in a certificate’s QR code.
      </PageHeader>
      <div className="page-grid">
        <VerifyForm key={key} name={name} id={id} onSubmit={(n, i) => setParams({ name: cleanName(n), id: normalizeId(i) })} />
        <div aria-live="polite">
          {verdict?.ok && <ValidResult verdict={verdict} />}
          {verdict && !verdict.ok && <InvalidResult reason={verdict.reason} />}
        </div>
        <HowVerified />
      </div>
    </ChapterPage>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Certificate page parts
// ─────────────────────────────────────────────────────────────────────────────

function PageHeader({ title, children }: { title: string; children: ReactNode }) {
  return (
    <header className="page-grid mb-10">
      <div className="eyebrow flex items-center gap-2">
        <Award aria-hidden="true" size={15} /> Beyond the course
      </div>
      <h1 className="mt-3 text-[34px] leading-[1.12] font-semibold tracking-[-0.022em] text-balance text-ink sm:text-[42px]">{title}</h1>
      <p className="mt-4 text-[21px] leading-[1.55] text-pretty text-ink-2 sm:text-[22px]">{children}</p>
    </header>
  );
}

type StepState = 'done' | 'current' | 'upcoming';

function Steps({ steps }: { steps: { title: string; detail: string; state: StepState }[] }) {
  return (
    <ol className="!m-0 grid gap-3 !p-0 sm:grid-cols-3" style={{ listStyle: 'none' }}>
      {steps.map((s, i) => (
        <li
          key={s.title}
          aria-current={s.state === 'current' ? 'step' : undefined}
          className={cx(
            '!m-0 flex gap-3 rounded-xl border px-4 py-3.5',
            s.state === 'current' ? 'border-accent bg-accent-soft' : 'border-border bg-surface',
          )}
        >
          <span
            aria-hidden="true"
            className={cx(
              'mt-[1px] flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-[15px] font-semibold tabular-nums',
              s.state === 'done' ? 'bg-accent text-accent-ink' : s.state === 'current' ? 'border-[1.5px] border-accent text-accent-text' : 'border-[1.5px] border-axis text-muted',
            )}
          >
            {s.state === 'done' ? <Check size={15} strokeWidth={3} /> : i + 1}
          </span>
          <span className="min-w-0">
            <span className="block text-[17px] leading-snug font-semibold text-ink">
              {s.title}
              <span className="sr-only">{s.state === 'done' ? ' (done)' : s.state === 'current' ? ' (current step)' : ''}</span>
            </span>
            <span className="mt-0.5 block text-[15.5px] leading-snug text-ink-2">{s.detail}</span>
          </span>
        </li>
      ))}
    </ol>
  );
}

function RemainingChapters() {
  const completed = useSim((s) => s.completed);
  const left = CHAPTERS.filter((c) => !completed[c.slug]);
  return (
    <section className="mt-12">
      <h2 className="text-[26px] font-semibold tracking-[-0.015em] text-ink">
        {left.length === 1 ? 'One chapter to go' : `${left.length} chapters to go`}
      </h2>
      <p className="mt-3 text-[18px] leading-[1.65] text-ink-2">
        Tick a chapter once you have worked through it, with the button at its end or the circle beside it here and in the sidebar.
        The final quiz opens when every chapter is ticked.
      </p>
      <ul className="!mt-5 !mb-0 grid gap-2 !p-0" style={{ listStyle: 'none' }}>
        {left.map((c) => (
          <li key={c.slug} className="!m-0 flex items-center gap-2 rounded-xl border border-border bg-surface py-1.5 pr-1.5 pl-4">
            <Link to={`/${c.slug}`} data-plain className="flex min-w-0 flex-1 items-baseline gap-3 py-1 text-[17px] text-ink hover:text-accent-text">
              <span className="w-6 shrink-0 text-right text-[15.5px] text-muted tabular-nums">{c.number}</span>
              <span className="min-w-0">{c.title}</span>
            </Link>
            <CompleteToggle chapter={c} />
          </li>
        ))}
      </ul>
    </section>
  );
}

function NameForm({
  record,
  renaming,
  onIssue,
  onCancel,
}: {
  record: CertificateRecord;
  renaming: boolean;
  onIssue: (name: string) => void;
  onCancel?: () => void;
}) {
  const [value, setValue] = useState(record.name);
  const [tried, setTried] = useState(false);
  const problem = nameProblem(value);
  const submit = (e: FormEvent) => {
    e.preventDefault();
    setTried(true);
    if (!problem) onIssue(cleanName(value));
  };
  return (
    <section className="mt-12">
      {!renaming && (
        <p className="!m-0 flex items-center gap-2 text-[19px] font-semibold text-ink">
          <CircleCheck aria-hidden="true" size={20} style={{ color: 'var(--good)' }} />
          You passed the final quiz with {record.score} of {record.total} right answers.
        </p>
      )}
      <h2 className="mt-6 text-[26px] font-semibold tracking-[-0.015em] text-ink">{renaming ? 'Change the name' : 'Your name on the certificate'}</h2>
      <p className="mt-3 text-[18px] leading-[1.65] text-ink-2">
        Type it as it should be printed. The name is part of the certificate ID, so {renaming ? 'a new name gives a new ID' : 'changing it later gives a new ID'};
        a certificate already shared stays valid under its own name. Your name stays in this browser: the site has no server and sends it
        nowhere. It travels only inside the verification link, when you share that.
      </p>
      <form onSubmit={submit} noValidate className="mt-5 flex flex-col gap-3">
        <label className="flex flex-col gap-1.5">
          <span className="text-[16px] font-medium text-ink">Full name</span>
          <input
            type="text"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            autoComplete="name"
            maxLength={NAME_MAX + 20}
            aria-invalid={tried && !!problem}
            aria-describedby="cert-name-hint"
            className="h-12 w-full max-w-[34rem] rounded-xl border border-border-strong bg-surface px-4 text-[18px] text-ink placeholder:text-muted focus:border-accent focus:outline-none"
            placeholder="Ada Lovelace"
          />
        </label>
        <span id="cert-name-hint" className={cx('text-[15.5px]', tried && problem ? 'text-[var(--critical)]' : 'text-muted')}>
          {tried && problem ? problem : `Up to ${NAME_MAX} characters. Dated ${formatCertDate(record.date)}, the day you passed.`}
        </span>
        <div className="flex flex-wrap gap-3">
          <Button type="submit" variant="primary" icon={<Award aria-hidden="true" size={17} />}>
            {renaming ? 'Issue with this name' : 'Issue the certificate'}
          </Button>
          {onCancel && (
            <Button variant="ghost" onClick={onCancel}>
              Keep the current name
            </Button>
          )}
        </div>
      </form>
    </section>
  );
}

/** The ID of a record, computed asynchronously (null while it is being computed). */
function useCertificateView(facts: CertificateRecord): CertificateView | null {
  const key = `${facts.name}\n${facts.date}\n${facts.score}\n${facts.total}\n${facts.chapters}`;
  const [view, setView] = useState<{ key: string; view: CertificateView } | null>(null);
  useEffect(() => {
    let live = true;
    certificateId(facts).then((id) => live && setView({ key, view: { ...facts, id, url: verifyUrl(facts.name, id) } }));
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- the key covers every field of facts
  }, [key]);
  return view?.key === key ? view.view : null;
}

function Issued({ record, onRename }: { record: CertificateRecord; onRename: () => void }) {
  const view = useCertificateView(record);
  const [busy, setBusy] = useState<'pdf' | 'png' | null>(null);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string | null>(null);
  if (!view) return <div className="mt-12 aspect-[1600/1131] w-full animate-pulse rounded-xl bg-surface-2" />;

  const download = async (kind: 'pdf' | 'png') => {
    setBusy(kind);
    setError(null);
    try {
      const canvas = await renderCertificate(view, 2400);
      const blob = kind === 'pdf' ? await certificatePdf(canvas, view) : await canvasBlob(canvas, 'image/png');
      saveBlob(blob, `${certificateFileName(view.name)}.${kind}`);
    } catch {
      setError('This browser could not draw the certificate. Try another browser.');
    } finally {
      setBusy(null);
    }
  };
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(view.url);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      setError('Copying is blocked here. The link is printed below; select and copy it by hand.');
    }
  };

  return (
    <section className="mt-12">
      <h2 className="text-[26px] font-semibold tracking-[-0.015em] text-ink">Your certificate</h2>
      <CertificatePreview view={view} className="mt-5" />
      <div className="mt-5 flex flex-wrap gap-3">
        <Button variant="primary" icon={<Download aria-hidden="true" size={17} />} onClick={() => download('pdf')} disabled={busy !== null}>
          {busy === 'pdf' ? 'Preparing…' : 'Download PDF'}
        </Button>
        <Button icon={<Download aria-hidden="true" size={17} />} onClick={() => download('png')} disabled={busy !== null}>
          {busy === 'png' ? 'Preparing…' : 'Download PNG'}
        </Button>
        <Button icon={copied ? <Check aria-hidden="true" size={17} /> : <Copy aria-hidden="true" size={17} />} onClick={copy}>
          {copied ? 'Link copied' : 'Copy verification link'}
        </Button>
      </div>
      {error && <p className="!mt-3 !mb-0 text-[16px] text-[var(--critical)]">{error}</p>}
      <Details view={view} />
      <div className="mt-5 flex flex-wrap gap-x-5 gap-y-2 text-[16.5px]">
        <Link to={verifyPath(view.name, view.id)} className="inline-flex items-center gap-1.5">
          <ShieldCheck aria-hidden="true" size={17} /> Open its verification page
        </Link>
        <button type="button" onClick={onRename} className="inline-flex items-center gap-1.5 text-accent-text hover:underline">
          <PenLine aria-hidden="true" size={17} /> Change the name
        </button>
      </div>
    </section>
  );
}

function Details({ view }: { view: CertificateView }) {
  const rows: [string, ReactNode][] = [
    ['Name', view.name],
    ['Completed on', formatCertDate(view.date)],
    ['Final quiz', `${view.score} of ${view.total} right`],
    ['Chapters', `all ${view.chapters}`],
    ['Certificate ID', <span className="tracking-[0.04em]">{view.id}</span>],
    [
      'Verification link',
      <a href={view.url} target="_blank" rel="noreferrer noopener" className="break-all">
        {view.url}
      </a>,
    ],
  ];
  return (
    <dl className="!mt-6 !mb-0 grid gap-x-6 gap-y-2.5 rounded-xl border border-border bg-surface px-5 py-4 text-[17px] sm:grid-cols-[auto_minmax(0,1fr)]">
      {rows.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="text-[15.5px] font-medium text-muted sm:pt-[2px]">{k}</dt>
          <dd className="!m-0 min-w-0 text-ink">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

/** The certificate as drawn for download, shown as an image (the same canvas renderer). */
function CertificatePreview({ view, className }: { view: CertificateView; className?: string }) {
  const key = `${view.name}\n${view.id}`;
  // url null: drawing failed
  const [img, setImg] = useState<{ key: string; url: string | null } | null>(null);
  useEffect(() => {
    let live = true;
    let url = '';
    renderCertificate(view, 1800)
      .then((c) => canvasBlob(c, 'image/png'))
      .then((b) => {
        if (!live) return;
        url = URL.createObjectURL(b);
        setImg({ key, url });
      })
      .catch(() => live && setImg({ key, url: null }));
    return () => {
      live = false;
      if (url) URL.revokeObjectURL(url);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- the ID covers every field the picture shows
  }, [key]);
  const alt = `Certificate of completion for ${view.name}: completed all ${view.chapters} chapters of ${SITE_TITLE} and passed the final quiz with ${view.score} of ${view.total} answers correct, on ${formatCertDate(view.date)}. Certificate ID ${view.id}.`;
  return (
    <div className={cx('aspect-[1600/1131] w-full overflow-hidden rounded-xl border border-border bg-[#fffdf8] shadow-[var(--shadow-pop)]', className)}>
      {img?.key === key && img.url ? (
        <img src={img.url} alt={alt} className="block h-full w-full" />
      ) : img?.key === key ? (
        <p className="!m-0 flex h-full items-center justify-center px-6 text-center text-[16px] text-[#4b4842]">
          This browser could not draw the certificate. The details below are what it states.
        </p>
      ) : (
        <div className="h-full w-full animate-pulse bg-surface-2" role="img" aria-label={alt} />
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Verify page parts
// ─────────────────────────────────────────────────────────────────────────────

function VerifyForm({ name, id, onSubmit }: { name: string; id: string; onSubmit: (name: string, id: string) => void }) {
  const [n, setN] = useState(name);
  const [i, setI] = useState(id);
  const input =
    'h-12 w-full rounded-xl border border-border-strong bg-surface px-4 text-[18px] text-ink placeholder:text-muted focus:border-accent focus:outline-none';
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit(n, i);
      }}
      className="grid gap-4 rounded-2xl border border-border bg-surface px-5 py-5 sm:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)_auto] sm:items-end"
    >
      <label className="flex min-w-0 flex-col gap-1.5">
        <span className="text-[16px] font-medium text-ink">Name</span>
        <input type="text" value={n} onChange={(e) => setN(e.target.value)} className={input} placeholder="As printed" />
      </label>
      <label className="flex min-w-0 flex-col gap-1.5">
        <span className="text-[16px] font-medium text-ink">Certificate ID</span>
        <input
          type="text"
          value={i}
          onChange={(e) => setI(e.target.value)}
          className={cx(input, 'tracking-[0.04em] uppercase')}
          placeholder="XXXXX-XXXXX-XXXXX-XXXXX"
          spellCheck={false}
          autoCapitalize="characters"
        />
      </label>
      <Button type="submit" variant="primary" className="h-12" icon={<ShieldCheck aria-hidden="true" size={17} />}>
        Verify
      </Button>
    </form>
  );
}

function ValidResult({ verdict }: { verdict: Extract<Verdict, { ok: true }> }) {
  const f = verdict.facts;
  const view: CertificateView = { ...f, id: verdict.id, url: verifyUrl(f.name, verdict.id) };
  return (
    <section className="mt-8">
      <div className="rounded-2xl border border-[var(--good)] bg-surface px-5 py-4">
        <p className="!m-0 flex items-center gap-2 text-[20px] font-semibold text-ink">
          <CircleCheck aria-hidden="true" size={22} style={{ color: 'var(--good)' }} /> The name and ID match
        </p>
        <p className="!mt-2 !mb-0 text-[18px] leading-[1.65] text-ink-2">
          This ID was issued for <strong className="text-ink">{f.name}</strong>, who completed all {f.chapters} chapters of {SITE_TITLE} and
          passed its final quiz with {f.score} of {f.total} right answers on {formatCertDate(f.date)}. Compare these details with the
          certificate you were shown: the ID only matches them.
        </p>
      </div>
      <CertificatePreview view={view} className="mt-6" />
      <p className="!mt-3 !mb-0 text-[15.5px] text-muted">The certificate as this site draws it from the details above.</p>
    </section>
  );
}

function InvalidResult({ reason }: { reason: 'name' | 'id-format' | 'mismatch' }) {
  const text = {
    name: 'Type the name as printed on the certificate.',
    'id-format': 'A certificate ID has 20 letters and digits, in four groups of five. Check it for a missing or extra character.',
    mismatch:
      'No certificate of this site has this name with this ID. Check both for typos (case and extra spaces do not matter). If they are typed exactly as printed, the certificate was altered or not issued by this site.',
  }[reason];
  return (
    <div className="mt-8 rounded-2xl border border-[var(--critical)] bg-surface px-5 py-4">
      <p className="!m-0 flex items-center gap-2 text-[20px] font-semibold text-ink">
        <CircleX aria-hidden="true" size={22} style={{ color: 'var(--critical)' }} />
        {reason === 'mismatch' ? 'The name and ID do not match' : 'Not a valid entry'}
      </p>
      <p className="!mt-2 !mb-0 text-[18px] leading-[1.65] text-ink-2">{text}</p>
    </div>
  );
}

/** What the ID proves and what it does not: shown on both pages. */
function HowVerified() {
  return (
    <Callout title="What the certificate ID proves" className="mt-14">
      <p>
        The ID is computed from the name, the completion date, the quiz score and the number of chapters. Change any of them on a
        certificate and the ID stops matching, so a{' '}
        <Link to="/certificate/verify">verification</Link> catches an edited certificate.
      </p>
      <p>
        It cannot prove more than that. {SITE_TITLE} is a static site with no server: the quiz is graded and the ID is computed in your
        browser, and someone who reads the site’s code could compute a matching ID for any name. Read the certificate as a record of
        completion on the honor system. It is issued by an independent educational site, not affiliated with the paper’s authors or their
        institutions, and it is not an academic credential.
      </p>
      <p className="!mb-0 text-[16px] text-ink-2">
        Verification links point to{' '}
        <a href={SITE_URL} className="[overflow-wrap:anywhere]">
          {SITE_URL.replace(/^https:\/\//, '').replace(/\/$/, '')}
        </a>
        .
      </p>
    </Callout>
  );
}
