/**
 * RacingDB (`racing`): a fictional motor-racing championship. Pretraining DB (BUILD_PLAN §3.3).
 *
 *   drivers(driver_id; name, nationality, birth_year)
 *   teams  (team_id;   name, country)
 *   races  (race_id;   circuit, race_date [time], laps)
 *   results(result_id; driver_id→drivers, team_id→teams, race_id→races; date [time], grid, position, points, dnf)
 *
 * `races` has a time column but NO foreign key, so pre.rs never sets its node timestamp
 * (node ts = None, spec §1.4) and the sampler's temporal filter always lets race rows through.
 * All results of one race share the race timestamp, so for the `results.position` autocomplete
 * task the leakage columns (points, dnf) are dropped for every row of the seed's race
 * (same-timestamp rule, spec §4.9).
 *
 * Generative process (persistent signal): driver skill and reliability propensity (how often a
 * driver's races end early: crashes, a car-breaking style) are fixed traits; each team's car pace
 * and unreliability drift slowly from race to race (AR(1)) and jump a little between seasons.
 * Grid = qualifying order of skill + car + noise. Each car retires (DNF) with a probability driven
 * mostly by the driver's propensity and partly by team unreliability, so DNFs cluster on the same
 * drivers window after window (self-label persistence, spec §11.2); finishers are ranked by race
 * pace; points 25-18-15-12-10-8-6-4-2-1 for the top ten finishers.
 * Two veterans retire after 2024; two rookies join for 2025 (entities enter/leave the tasks).
 *
 * All names (drivers, teams) are fictional. Circuit names are real venues, used as places only.
 */
import type { RawTable } from '../../types';
import { DAY_MS, col, makeTable, normal, sigmoid, sortByTime, subRng, utc } from '../util';
import type { GeneratedDb } from './types';

export const RACING_SEED = 20240107;

const TEAMS: { name: string; country: string }[] = [
  { name: 'Aurora Racing', country: 'United Kingdom' },
  { name: 'Scuderia Volta', country: 'Italy' },
  { name: 'Kestrel Grand Prix', country: 'Germany' },
  { name: 'Meridian Motorsport', country: 'United States' },
  { name: 'Sakura Speed', country: 'Japan' },
  { name: 'Bleu Horizon Racing', country: 'France' },
  { name: 'Iberia Racing Team', country: 'Spain' },
  { name: 'Southern Cross Motorsport', country: 'Australia' },
  { name: 'Lowland Racing', country: 'Netherlands' },
  { name: 'Polaris Racing', country: 'Switzerland' },
];

/** 22 fictional drivers. Indices 20 and 21 are 2025 rookies; 16 and 10 retire after 2024. */
const DRIVERS: { name: string; nationality: string; birthYear: number }[] = [
  { name: 'Marco Bellini', nationality: 'Italian', birthYear: 1994 },
  { name: 'Lena Hoffmann', nationality: 'German', birthYear: 1999 },
  { name: 'Oliver Grant', nationality: 'British', birthYear: 1996 },
  { name: 'Kenji Watanabe', nationality: 'Japanese', birthYear: 1998 },
  { name: 'Rafael Moreira', nationality: 'Brazilian', birthYear: 1993 },
  { name: 'Hugo Lambert', nationality: 'French', birthYear: 2000 },
  { name: 'Diego Navarro', nationality: 'Spanish', birthYear: 1991 },
  { name: 'Jack Whitfield', nationality: 'Australian', birthYear: 1997 },
  { name: 'Sven de Vries', nationality: 'Dutch', birthYear: 2001 },
  { name: 'Mateo Ruiz', nationality: 'Mexican', birthYear: 1995 },
  { name: 'Eetu Virtanen', nationality: 'Finnish', birthYear: 1990 },
  { name: 'Noah Tremblay', nationality: 'Canadian', birthYear: 2002 },
  { name: 'Cole Hartley', nationality: 'American', birthYear: 1999 },
  { name: 'Magnus Holm', nationality: 'Danish', birthYear: 1992 },
  { name: 'Wei Zhang', nationality: 'Chinese', birthYear: 2000 },
  { name: 'Arun Chaiyaporn', nationality: 'Thai', birthYear: 2003 },
  { name: 'Felix Brandt', nationality: 'German', birthYear: 1989 },
  { name: 'Charlotte Evans', nationality: 'British', birthYear: 2001 },
  { name: 'Luca Ferraro', nationality: 'Italian', birthYear: 2004 },
  { name: 'Pierre Gasnier', nationality: 'French', birthYear: 1997 },
  { name: 'Thiago Barros', nationality: 'Brazilian', birthYear: 2005 },
  { name: 'Yuki Morimoto', nationality: 'Japanese', birthYear: 2006 },
];

/** Calendar order of circuits with their race distance in laps. */
const CIRCUITS: [string, number][] = [
  ['Sakhir', 57],
  ['Jeddah Corniche', 50],
  ['Albert Park', 58],
  ['Suzuka', 53],
  ['Shanghai', 56],
  ['Miami', 57],
  ['Imola', 63],
  ['Monaco', 78],
  ['Montreal', 70],
  ['Barcelona', 66],
  ['Spielberg', 71],
  ['Silverstone', 52],
  ['Hungaroring', 70],
  ['Spa-Francorchamps', 44],
  ['Zandvoort', 72],
  ['Monza', 53],
  ['Baku', 51],
  ['Marina Bay', 62],
  ['Austin', 56],
  ['Mexico City', 71],
  ['Interlagos', 71],
  ['Las Vegas', 50],
  ['Lusail', 57],
  ['Yas Marina', 58],
];

const POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1];

/**
 * Retirement model: logit P(DNF) = DNF_A + DNF_TEAM·(team unreliability) + DNF_DRIVER·(driver
 * propensity) + 0.25 from grid slot 11 back. The driver term dominates, so a driver's DNF history
 * carries over from one 30-day window to the next (window-to-window label correlation ≈ 0.4).
 */
const DNF_A = -2.8;
const DNF_DRIVER = 2.0;
const DNF_TEAM = 1.2;
/** Race-day noise of qualifying and race pace (smaller → results follow skill + car more closely). */
const QUALI_NOISE = 0.35;
const RACE_NOISE = 0.4;
/** Per-race innovation of the teams' unreliability (AR(1) around the team's base level). */
const UNREL_NOISE = 0.1;

export function generateRacing(seed = RACING_SEED): GeneratedDb {
  // ── calendar: a race every 14 days (Sundays, 13:00 UTC); 24 races in 2024 (Jan 14 – Dec 1),
  //    a winter break, then 12 races in 2025 (Jan 12 – Jun 15). Every 30-day window of the task
  //    timeline holds one to three races (one only in December 2024). ──
  const races: { circuit: string; date: number; laps: number; season: number }[] = [];
  for (let k = 0; k < 24; k++) {
    const [circuit, laps] = CIRCUITS[k];
    races.push({ circuit, laps, date: utc(2024, 1, 14, 13) + k * 14 * DAY_MS, season: 2024 });
  }
  for (let k = 0; k < 12; k++) {
    const [circuit, laps] = CIRCUITS[k];
    races.push({ circuit, laps, date: utc(2025, 1, 12, 13) + k * 14 * DAY_MS, season: 2025 });
  }

  // ── latent traits ─────────────────────────────────────────────────────────
  const rl = subRng(seed, 1);
  const skill = DRIVERS.map((_, i) => normal(rl, i >= 20 ? -0.2 : 0, 0.55));
  // reliability propensity: how often a driver's races end early (crashes, car-breaking style).
  // A strong, persistent trait, so a driver's DNF history predicts the next window (spec §11.2).
  const propensity = DRIVERS.map(() => normal(rl, 0, 1));
  const carBase = TEAMS.map(() => normal(rl, 0, 0.6));
  const unrelBase = TEAMS.map(() => normal(rl, 0, 0.6));

  // ── line-ups: 2 seats per team ────────────────────────────────────────────
  const lineup2024 = Array.from({ length: 20 }, (_, i) => i); // seat s → driver
  const lineup2025 = lineup2024.slice();
  lineup2025[16] = 20; // rookie takes the seat of retiring Felix Brandt
  lineup2025[10] = 21; // rookie takes the seat of retiring Eetu Virtanen
  // two transfers for 2025: swap seats 3 ↔ 8 and 5 ↔ 14
  [lineup2025[3], lineup2025[8]] = [lineup2025[8], lineup2025[3]];
  [lineup2025[5], lineup2025[14]] = [lineup2025[14], lineup2025[5]];

  // ── simulate races ────────────────────────────────────────────────────────
  const rr = subRng(seed, 2);
  const car = carBase.slice();
  const unrel = unrelBase.slice();
  interface ResRec {
    d: number;
    team: number;
    race: number;
    t: number;
    grid: number;
    position: number | null;
    points: number;
    dnf: boolean;
  }
  const results: ResRec[] = [];
  races.forEach((race, ri) => {
    if (ri > 0 && race.season !== races[ri - 1].season) {
      // new season: cars change more between seasons
      for (let tm = 0; tm < TEAMS.length; tm++) {
        carBase[tm] += normal(rr, 0, 0.3);
        car[tm] = carBase[tm];
      }
    }
    for (let tm = 0; tm < TEAMS.length; tm++) {
      car[tm] = carBase[tm] + 0.92 * (car[tm] - carBase[tm]) + normal(rr, 0, 0.12);
      unrel[tm] = unrelBase[tm] + 0.9 * (unrel[tm] - unrelBase[tm]) + normal(rr, 0, UNREL_NOISE);
    }
    const lineup = race.season === 2024 ? lineup2024 : lineup2025;
    const entries = lineup.map((d, seat) => ({ d, team: Math.floor(seat / 2) }));
    // qualifying → grid 1..20
    const qual = entries.map((e) => skill[e.d] + car[e.team] + normal(rr, 0, QUALI_NOISE));
    const gridOrder = entries.map((_, i) => i).sort((a, b) => qual[b] - qual[a] || a - b);
    const grid: number[] = new Array(entries.length);
    gridOrder.forEach((ei, pos) => (grid[ei] = pos + 1));
    // race: retirements, then finishing order by race pace
    const pace = entries.map((e, i) => skill[e.d] + car[e.team] + normal(rr, 0, RACE_NOISE) - 0.04 * grid[i]);
    const dnf = entries.map((e, i) =>
      rr.bool(sigmoid(DNF_A + DNF_TEAM * unrel[e.team] + DNF_DRIVER * propensity[e.d] + (grid[i] > 10 ? 0.25 : 0))),
    );
    const finishers = entries.map((_, i) => i).filter((i) => !dnf[i]).sort((a, b) => pace[b] - pace[a] || a - b);
    const position: (number | null)[] = entries.map(() => null);
    finishers.forEach((ei, p) => (position[ei] = p + 1));
    // result rows of a race are listed in finishing order, retirements last (by grid)
    const order = finishers.concat(entries.map((_, i) => i).filter((i) => dnf[i]).sort((a, b) => grid[a] - grid[b]));
    for (const i of order) {
      const pos = position[i];
      results.push({
        d: entries[i].d,
        team: entries[i].team,
        race: ri,
        t: race.date,
        grid: grid[i],
        position: pos,
        points: pos !== null && pos <= 10 ? POINTS[pos - 1] : 0,
        dnf: dnf[i],
      });
    }
  });
  // already chronological; stable sort keeps the within-race order
  sortByTime(results, (x) => x.t, () => 0);

  const tables: RawTable[] = [
    makeTable(
      {
        name: 'drivers',
        pkey: 'driver_id',
        fkeys: {},
        timeCol: null,
        description: 'One row per driver. A dimension table: no foreign keys and no time column.',
        columns: [
          col('driver_id', 'int', 'Primary key (= row position).'),
          col('name', 'str', 'Driver name (fictional).'),
          col('nationality', 'str', 'Nationality of the driver.'),
          col('birth_year', 'int', 'Year of birth.'),
        ],
      },
      DRIVERS.map((d, i) => ({ driver_id: i, name: d.name, nationality: d.nationality, birth_year: d.birthYear })),
    ),
    makeTable(
      {
        name: 'teams',
        pkey: 'team_id',
        fkeys: {},
        timeCol: null,
        description: 'One row per team (constructor). A dimension table: no foreign keys and no time column.',
        columns: [
          col('team_id', 'int', 'Primary key (= row position).'),
          col('name', 'str', 'Team name (fictional).'),
          col('country', 'str', 'Country where the team is based.'),
        ],
      },
      TEAMS.map((t, i) => ({ team_id: i, name: t.name, country: t.country })),
    ),
    makeTable(
      {
        name: 'races',
        pkey: 'race_id',
        fkeys: {},
        timeCol: 'race_date',
        description:
          'One row per race. It has a time column but no foreign key, so preprocessing gives its rows no timestamp.',
        columns: [
          col('race_id', 'int', 'Primary key (= row position; races are in calendar order).'),
          col('circuit', 'str', 'Circuit where the race is held.'),
          col('race_date', 'datetime', 'Race start time (the time column of this table).'),
          col('laps', 'int', 'Scheduled race distance in laps.'),
        ],
      },
      races.map((r, i) => ({ race_id: i, circuit: r.circuit, race_date: r.date, laps: r.laps })),
    ),
    makeTable(
      {
        name: 'results',
        pkey: 'result_id',
        fkeys: { driver_id: 'drivers', team_id: 'teams', race_id: 'races' },
        timeCol: 'date',
        description: 'One row per driver per race: starting grid slot, finishing position, points and whether the car retired.',
        columns: [
          col('result_id', 'int', 'Primary key (= row position).'),
          col('driver_id', 'int', 'Foreign key → drivers.'),
          col('team_id', 'int', 'Foreign key → teams: the team the driver raced for.'),
          col('race_id', 'int', 'Foreign key → races.'),
          col('date', 'datetime', 'Race start time (the time column of this table).'),
          col('grid', 'int', 'Starting position from qualifying (1 = pole).'),
          col('position', 'int', 'Finishing position; missing when the driver did not finish.'),
          col('points', 'float', 'Championship points scored (25-18-15-12-10-8-6-4-2-1).'),
          col('dnf', 'bool', 'Whether the driver did not finish (retired from the race).'),
        ],
      },
      results.map((x, i) => ({
        result_id: i,
        driver_id: x.d,
        team_id: x.team,
        race_id: x.race,
        date: x.t,
        grid: x.grid,
        position: x.position,
        points: x.points,
        dnf: x.dnf,
      })),
    ),
  ];

  return { name: 'racing', tables };
}

