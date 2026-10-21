/**
 * ForumDB (`forum`): a programming Q&A forum. Pretraining DB (BUILD_PLAN §3.4).
 *
 *   users(user_id; display_name, location, reputation, joined)
 *   posts(post_id; owner_id→users; created [time], title, tag, body_length)
 *   votes(vote_id; user_id→users, post_id→posts; created [time], is_upvote)
 *
 * Generative process (persistent signal):
 *  - Users follow a monthly Markov activity regime (stay 0.80, re-activate 0.20), starting
 *    active in the month they join. While active they post Poisson(post_rate) times a month
 *    and cast votes in proportion to a persistent per-user vote rate.
 *  - A latent expertise drives post quality (with a small body-length effect) and the profile
 *    reputation. Each post has an appeal (quality plus a heavy-tailed post-specific factor) and
 *    receives votes at a rate appeal × attention, where attention decays slowly with the post's
 *    age down to a long-tail floor. A post's vote rate does not depend on how many other posts
 *    exist, so popular posts stay popular from month to month (self-label persistence, spec §11.2).
 *    Each vote comes from an active user (more often one who follows the post's tag); a user
 *    votes at most once per post and never on their own posts. P(upvote) rises with post quality
 *    and the voter's generosity.
 *  - `reputation` is a profile attribute driven by expertise, not computed from the votes in this
 *    DB (so it carries no future information).
 */
import type { RawTable } from '../../types';
import {
  DATA_END,
  DAY_MS,
  MINUTE_MS,
  NUM_MONTHS,
  categorical,
  clamp,
  col,
  detExp,
  detLog,
  gamma,
  logNormal,
  makeTable,
  monthIndexOf,
  monthStart,
  normal,
  poisson,
  randomTime,
  sigmoid,
  sortByTime,
  subRng,
  utc,
} from '../util';
import type { GeneratedDb } from './types';

export const FORUM_SEED = 20240115;

const N_USERS = 180;
/** Gamma scale of a user's monthly posting rate (shape 1.3) and vote-rate weight (shape 2). */
const POST_SCALE = 0.12;
const VOTE_SCALE = 2.1;
/** Post appeal = exp(APPEAL_Q·quality + N(0, APPEAL_SD)). */
const APPEAL_Q = 0.8;
const APPEAL_SD = 0.8;
/** Expected votes per day of a fresh post of appeal 1. */
const VOTES_PER_DAY = 0.04;
/** Attention decay of a post (e-folding, days) and its long-tail floor. */
const ATTN_TAU = 250;
const ATTN_FLOOR = 0.4;

const NAME_A = ['pixel', 'byte', 'lambda', 'async', 'rusty', 'curly', 'tensor', 'query', 'kernel', 'vector', 'quiet', 'lazy', 'brave', 'cosmic', 'tiny', 'nimble', 'sleepy', 'clever', 'binary', 'agile'];
const NAME_B = ['fox', 'owl', 'panda', 'otter', 'wolf', 'lynx', 'hawk', 'koala', 'badger', 'heron', 'gecko', 'yak', 'raven', 'moose', 'tiger', 'falcon'];

const LOCATIONS = [
  'Berlin, Germany',
  'São Paulo, Brazil',
  'Bangalore, India',
  'Toronto, Canada',
  'Seattle, USA',
  'London, UK',
  'Tokyo, Japan',
  'Paris, France',
  'Sydney, Australia',
  'Lagos, Nigeria',
  'Mexico City, Mexico',
  'Warsaw, Poland',
  'Porto Alegre, Brazil',
  'Austin, USA',
];

const TAGS = ['python', 'javascript', 'sql', 'rust', 'css', 'statistics'] as const;
const TAG_LANG: Record<(typeof TAGS)[number], string> = {
  python: 'Python',
  javascript: 'JavaScript',
  sql: 'SQL',
  rust: 'Rust',
  css: 'CSS',
  statistics: 'R',
};
/** Ten question topics per tag; titles are topic × one of four question framings. */
const TOPICS: Record<(typeof TAGS)[number], string[]> = {
  python: [
    'read a large CSV file in chunks',
    'merge two dictionaries',
    'speed up a pandas groupby',
    'type-hint a decorator',
    'parse dates with time zones',
    'test async code',
    'profile memory usage',
    'flatten a nested list',
    'run a subprocess with a timeout',
    'serialize a dataclass to JSON',
  ],
  javascript: [
    'debounce an input handler',
    'deep clone an object',
    'cancel a fetch request',
    'sort an array of objects by date',
    'avoid a memory leak in event listeners',
    'format a number as currency',
    'read a file in the browser',
    'wait for several promises',
    'detect dark mode',
    'lazy load images',
  ],
  sql: [
    'select the latest row per group',
    'pivot rows into columns',
    'compute a running total',
    'delete duplicate rows',
    'index a column used in a join',
    'find gaps in a sequence',
    'paginate results efficiently',
    'compare dates across time zones',
    'write a recursive query',
    'count distinct values per month',
  ],
  rust: [
    'share state between threads',
    'return an iterator from a function',
    'fix a borrow checker error in a loop',
    'parse command-line arguments',
    'convert a String to a &str',
    'handle errors without unwrap',
    'read a file line by line',
    'implement Display for a struct',
    'call C code via FFI',
    'use async with tokio',
  ],
  css: [
    'center a div vertically',
    'make a sticky header',
    'build a responsive grid',
    'truncate text with an ellipsis',
    'animate a height change',
    'style a file input',
    'create a dark theme with variables',
    'stop a flex item from shrinking',
    'fix z-index stacking',
    'make an element fill the viewport',
  ],
  statistics: [
    'interpret a p-value',
    'compute a bootstrap confidence interval',
    'fit a mixed-effects model',
    'check normality of residuals',
    'handle class imbalance',
    'choose the number of clusters',
    'run a chi-squared test',
    'adjust for multiple comparisons',
    'estimate a sample size',
    'plot a ROC curve',
  ],
};
const FRAMINGS: ((topic: string, lang: string) => string)[] = [
  (t, l) => `How do I ${t} in ${l}?`,
  (t, l) => `Best way to ${t} in ${l}`,
  (t, l) => `Why is it so slow to ${t} in ${l}?`,
  (t, l) => `Error when trying to ${t} (${l})`,
];

export function generateForum(seed = FORUM_SEED): GeneratedDb {

  // ── users ────────────────────────────────────────────────────────────────
  const ru = subRng(seed, 1);
  interface User {
    displayName: string;
    location: string | null;
    reputation: number;
    joined: number;
    expertise: number;
    postRate: number;
    voteRate: number;
    generosity: number;
    verbosity: number;
    tagW: number[];
  }
  const used = new Set<string>();
  const users: User[] = [];
  for (let i = 0; i < N_USERS; i++) {
    let displayName = '';
    do {
      const a = ru.choice(NAME_A);
      const b = ru.choice(NAME_B);
      const sep = ru.choice(['', '_', '']);
      displayName = ru.bool(0.55) ? `${a}${sep}${b}${10 + ru.int(90)}` : `${a}${sep}${b}`;
    } while (used.has(displayName));
    used.add(displayName);
    const expertise = normal(ru);
    const joined = ru.bool(0.35) ? randomTime(ru, utc(2024, 1, 1), utc(2024, 3, 1)) : randomTime(ru, utc(2024, 3, 1), utc(2025, 5, 1));
    const fav = ru.int(TAGS.length);
    const fav2 = ru.int(TAGS.length);
    users.push({
      displayName,
      location: ru.bool(0.15) ? null : ru.choice(LOCATIONS),
      reputation: Math.round(detExp(4.4 + 1.1 * expertise + normal(ru, 0, 0.45))),
      joined,
      expertise,
      postRate: gamma(ru, 1.3, POST_SCALE) * clamp(1 + 0.3 * expertise, 0.4, 2),
      voteRate: gamma(ru, 2, VOTE_SCALE),
      generosity: normal(ru, 0, 0.7),
      verbosity: normal(ru, 0, 0.3),
      tagW: TAGS.map((_, t) => (t === fav ? 4 : t === fav2 ? 2 : 0.3)),
    });
  }

  // ── monthly activity regime ──────────────────────────────────────────────
  const ra = subRng(seed, 2);
  const active: boolean[][] = users.map((u) => {
    const m0 = monthIndexOf(u.joined);
    const seq: boolean[] = [];
    let state = true;
    for (let m = 0; m < NUM_MONTHS; m++) {
      if (m < m0) {
        seq.push(false);
        continue;
      }
      if (m > m0) state = state ? ra.bool(0.8) : ra.bool(0.2);
      seq.push(state);
    }
    return seq;
  });
  const monthWindow = (u: User, m: number): [number, number] => [
    Math.max(monthStart(m), u.joined + MINUTE_MS),
    Math.min(monthStart(m + 1), DATA_END),
  ];

  // ── posts ────────────────────────────────────────────────────────────────
  const rp = subRng(seed, 3);
  interface PostRec {
    owner: number;
    t: number;
    title: string;
    tag: number;
    body: number;
    quality: number;
  }
  const posts: PostRec[] = [];
  for (let u = 0; u < N_USERS; u++) {
    const us = users[u];
    for (let m = 0; m < NUM_MONTHS; m++) {
      if (!active[u][m]) continue;
      const [a, b] = monthWindow(us, m);
      if (b <= a) continue;
      const n = poisson(rp, (us.postRate * (b - a)) / (monthStart(m + 1) - monthStart(m)));
      for (let k = 0; k < n; k++) {
        const tag = categorical(rp, us.tagW);
        const tg = TAGS[tag];
        const title = rp.choice(FRAMINGS)(rp.choice(TOPICS[tg]), TAG_LANG[tg]);
        const body = clamp(Math.round(logNormal(rp, 6.2 + us.verbosity + 0.1 * us.expertise, 0.45)), 40, 6000);
        const quality = 0.7 * us.expertise + 0.25 * detLog(body / 500) + normal(rp, 0, 0.6);
        posts.push({ owner: u, t: randomTime(rp, a, b), title, tag, body, quality });
      }
    }
  }
  sortByTime(posts, (p) => p.t, (p) => p.owner);

  // ── votes ────────────────────────────────────────────────────────────────
  const rv = subRng(seed, 4);
  interface VoteRec {
    voter: number;
    post: number;
    t: number;
    up: boolean;
  }
  const votes: VoteRec[] = [];
  // Appeal of a post: quality plus a post-specific factor (some questions just hit a nerve),
  // heavy-tailed, so a few posts keep collecting votes for months.
  const rq = subRng(seed, 5);
  const appeal = posts.map((p) => detExp(APPEAL_Q * p.quality + normal(rq, 0, APPEAL_SD)));
  // Votes are generated per post: in each month, post p receives Poisson(Λ) votes with
  //   Λ = VOTES_PER_DAY · appeal_p · ∫ (exp(−age/τ) + floor) d(age)   over the post's part of the month,
  // i.e. attention decays with the post's age (e-folding ATTN_TAU days) down to a long-tail floor,
  // independently of how many other posts exist. So a post's vote count carries over from one month
  // to the next (self-label persistence, spec §11.2). Each vote goes to an active user who joined
  // before it (weighted by the user's vote rate, doubled for the user's favourite tags); a user votes
  // at most once per post and never on their own posts.
  const tauMs = ATTN_TAU * DAY_MS;
  for (let pi = 0; pi < posts.length; pi++) {
    const p = posts[pi];
    const voters = new Set<number>([p.owner]);
    for (let m = monthIndexOf(p.t); m < NUM_MONTHS; m++) {
      const a = Math.max(monthStart(m), p.t + 60 * MINUTE_MS); // votes come at least an hour after posting
      const b = Math.min(monthStart(m + 1), DATA_END);
      if (b <= a) continue;
      const exposureDays = ATTN_TAU * (detExp(-(a - p.t) / tauMs) - detExp(-(b - p.t) / tauMs)) + (ATTN_FLOOR * (b - a)) / DAY_MS;
      const n = poisson(rv, VOTES_PER_DAY * appeal[pi] * exposureDays);
      for (let k = 0; k < n; k++) {
        const t = randomTime(rv, a, b);
        let any = false;
        const w = users.map((us, u) => {
          if (voters.has(u) || !active[u][m] || us.joined >= t) return 0;
          any = true;
          return us.voteRate * (us.tagW[p.tag] >= 2 ? 2 : 1);
        });
        if (!any) break; // nobody left to vote on this post this month
        const u = categorical(rv, w);
        voters.add(u);
        votes.push({ voter: u, post: pi, t, up: rv.bool(sigmoid(1.0 + 1.3 * p.quality + 0.6 * users[u].generosity)) });
      }
    }
  }
  sortByTime(votes, (v) => v.t, (v) => v.voter * 10000 + v.post);

  const tables: RawTable[] = [
    makeTable(
      {
        name: 'users',
        pkey: 'user_id',
        fkeys: {},
        timeCol: null,
        description: 'One row per registered user. A dimension table: no foreign keys and no time column.',
        columns: [
          col('user_id', 'int', 'Primary key (= row position).'),
          col('display_name', 'str', 'Public user name.'),
          col('location', 'str', 'Self-reported location; missing for about 15% of users.'),
          col('reputation', 'float', 'Reputation points shown on the profile.'),
          col('joined', 'datetime', 'When the account was created (a feature, not the time column).'),
        ],
      },
      users.map((u, i) => ({
        user_id: i,
        display_name: u.displayName,
        location: u.location,
        reputation: u.reputation,
        joined: u.joined,
      })),
    ),
    makeTable(
      {
        name: 'posts',
        pkey: 'post_id',
        fkeys: { owner_id: 'users' },
        timeCol: 'created',
        description: 'Questions asked on the forum.',
        columns: [
          col('post_id', 'int', 'Primary key (= row position; posts are sorted by time).'),
          col('owner_id', 'int', 'Foreign key → users: who asked the question.'),
          col('created', 'datetime', 'When the question was posted (the time column of this table).'),
          col('title', 'str', 'Question title.'),
          col('tag', 'str', 'Main topic tag.'),
          col('body_length', 'int', 'Length of the question body in characters.'),
        ],
      },
      posts.map((p, i) => ({
        post_id: i,
        owner_id: p.owner,
        created: p.t,
        title: p.title,
        tag: TAGS[p.tag],
        body_length: p.body,
      })),
    ),
    makeTable(
      {
        name: 'votes',
        pkey: 'vote_id',
        fkeys: { user_id: 'users', post_id: 'posts' },
        timeCol: 'created',
        description: 'Up- and down-votes cast by users on posts.',
        columns: [
          col('vote_id', 'int', 'Primary key (= row position; votes are sorted by time).'),
          col('user_id', 'int', 'Foreign key → users: who voted.'),
          col('post_id', 'int', 'Foreign key → posts: the post voted on.'),
          col('created', 'datetime', 'When the vote was cast (the time column of this table).'),
          col('is_upvote', 'bool', 'True for an up-vote, false for a down-vote.'),
        ],
      },
      votes.map((v, i) => ({ vote_id: i, user_id: v.voter, post_id: v.post, created: v.t, is_upvote: v.up })),
    ),
  ];

  return { name: 'forum', tables };
}

