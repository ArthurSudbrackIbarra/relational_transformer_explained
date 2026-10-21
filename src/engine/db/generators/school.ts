/**
 * SchoolDB (`school`): a university course-work database. Pretraining DB (BUILD_PLAN §3.2).
 *
 *   students   (student_id; program, age, scholarship, enrolled_on)
 *   courses    (course_id;  title, department, credits, difficulty)
 *   submissions(submission_id; student_id→students, course_id→courses; submitted_at [time], grade, late)
 *
 * Generative process (persistent signal):
 *  - Students join in one of three intakes (most in January 2024) and enrol each regular term in
 *    3–4 courses (every course runs in every regular term), chosen by the course's popularity and
 *    mostly from their own department; about half take one of the six summer courses.
 *  - Course popularity is a persistent per-course level (log-normal spread) that drifts slowly from
 *    term to term (log-AR(1)), so a course's submission volume carries over from one month to the
 *    next (self-label persistence, spec §11.2), apart from the term breaks that affect every course.
 *  - A course has two assignments per credit in a regular term (one per credit in summer), due at
 *    evenly spaced dates. A student submits an assignment only while *engaged* (monthly Markov
 *    regime: stay 0.90, re-engage 0.12, so disengagement is sticky, like a drop-out) and then with a
 *    personal diligence probability.
 *  - `late` comes from a persistent procrastination trait (+ course difficulty, − scholarship);
 *    late work is submitted after the due date and loses points.
 *  - `grade` follows ability − difficulty + noise; recent submissions are often not graded yet
 *    (null → no token, spec §3.5).
 */
import type { RawTable } from '../../types';
import {
  DATA_END,
  DAY_MS,
  HOUR_MS,
  categorical,
  clamp,
  col,
  detExp,
  gamma,
  makeTable,
  monthIndexOf,
  normal,
  randomTime,
  round,
  sigmoid,
  sortByTime,
  subRng,
  utc,
  wholeSeconds,
} from '../util';
import type { GeneratedDb } from './types';

export const SCHOOL_SEED = 20240108;

/** Course popularity (log scale): spread across courses, AR(1) persistence per term, per-term drift. */
const POP_SD = 1.0;
const POP_PHI = 0.8;
const POP_DRIFT = 0.15;
/** Students prefer courses of their own department by this factor. */
const OWN_DEPT = 3;
/** Share of students per intake (January 2024, August 2024, January 2025). */
const INTAKE_W0 = 0.7;
const INTAKE_W1 = 0.2;
const INTAKE_W2 = 0.1;
/** Probability that a student takes a summer course. */
const SUMMER_P = 0.45;

const PROGRAMS: { program: string; dept: string }[] = [
  { program: 'Computer Science', dept: 'Computer Science' },
  { program: 'Mathematics', dept: 'Mathematics' },
  { program: 'Biology', dept: 'Biology' },
  { program: 'Economics', dept: 'Economics' },
  { program: 'History', dept: 'History' },
  { program: 'Mechanical Engineering', dept: 'Engineering' },
];
const PROGRAM_WEIGHTS = [0.24, 0.12, 0.17, 0.18, 0.11, 0.18];

interface CourseSpec {
  title: string;
  department: string;
  credits: number;
  difficulty: number;
  /** Also offered in the summer session. */
  summer?: boolean;
}

/** 24 courses, four per department. Every course runs in every regular term; `summer` = also in summer. */
const COURSES: CourseSpec[] = [
  { title: 'Introduction to Programming', department: 'Computer Science', credits: 4, difficulty: 2.2, summer: true },
  { title: 'Data Structures and Algorithms', department: 'Computer Science', credits: 4, difficulty: 3.6 },
  { title: 'Database Systems', department: 'Computer Science', credits: 4, difficulty: 3.0 },
  { title: 'Machine Learning', department: 'Computer Science', credits: 5, difficulty: 4.2 },
  { title: 'Calculus I', department: 'Mathematics', credits: 4, difficulty: 3.1, summer: true },
  { title: 'Linear Algebra', department: 'Mathematics', credits: 4, difficulty: 3.3 },
  { title: 'Probability and Statistics', department: 'Mathematics', credits: 4, difficulty: 3.4 },
  { title: 'Real Analysis', department: 'Mathematics', credits: 5, difficulty: 4.6 },
  { title: 'Cell Biology', department: 'Biology', credits: 4, difficulty: 2.9 },
  { title: 'Genetics', department: 'Biology', credits: 4, difficulty: 3.5 },
  { title: 'Ecology and Evolution', department: 'Biology', credits: 3, difficulty: 2.5, summer: true },
  { title: 'Biochemistry', department: 'Biology', credits: 5, difficulty: 4.0 },
  { title: 'Principles of Microeconomics', department: 'Economics', credits: 3, difficulty: 2.3, summer: true },
  { title: 'Macroeconomics', department: 'Economics', credits: 3, difficulty: 2.7 },
  { title: 'Econometrics', department: 'Economics', credits: 4, difficulty: 3.9 },
  { title: 'Behavioral Economics', department: 'Economics', credits: 3, difficulty: 2.8 },
  { title: 'World History since 1500', department: 'History', credits: 3, difficulty: 2.0, summer: true },
  { title: 'History of Science', department: 'History', credits: 3, difficulty: 2.4 },
  { title: 'Modern European History', department: 'History', credits: 3, difficulty: 2.6 },
  { title: 'Ancient Mediterranean Civilizations', department: 'History', credits: 3, difficulty: 2.2 },
  { title: 'Statics', department: 'Engineering', credits: 4, difficulty: 3.2, summer: true },
  { title: 'Thermodynamics', department: 'Engineering', credits: 4, difficulty: 3.8 },
  { title: 'Fluid Mechanics', department: 'Engineering', credits: 5, difficulty: 4.1 },
  { title: 'Control Systems', department: 'Engineering', credits: 5, difficulty: 4.3 },
];

interface Term {
  kind: 'spring' | 'summer' | 'fall';
  start: number;
  end: number;
}

/** Academic calendar. The summer 2025 term starts but the data stops on 2025-06-30. */
const TERMS: Term[] = [
  { kind: 'spring', start: utc(2024, 1, 15), end: utc(2024, 6, 7) },
  { kind: 'summer', start: utc(2024, 6, 17), end: utc(2024, 8, 9) },
  { kind: 'fall', start: utc(2024, 8, 26), end: utc(2024, 12, 13) },
  { kind: 'spring', start: utc(2025, 1, 13), end: utc(2025, 6, 6) },
  { kind: 'summer', start: utc(2025, 6, 16), end: utc(2025, 8, 8) },
];

/** Intakes: [first term index, enrolment window start, end, weight]. */
const INTAKES: [number, number, number, number][] = [
  [0, utc(2024, 1, 2), utc(2024, 1, 13), INTAKE_W0],
  [2, utc(2024, 8, 5), utc(2024, 8, 24), INTAKE_W1],
  [3, utc(2025, 1, 2), utc(2025, 1, 11), INTAKE_W2],
];

function offeredIn(c: CourseSpec, term: Term): boolean {
  return term.kind === 'summer' ? c.summer === true : true;
}

/** Assignments of a course in a term: two per credit in a regular term, one per credit in summer. */
function assignmentsOf(c: CourseSpec, term: Term): number {
  return term.kind === 'summer' ? c.credits : 2 * c.credits;
}

export function generateSchool(seed = SCHOOL_SEED): GeneratedDb {
  const N_STUDENTS = 110;

  // ── students ──────────────────────────────────────────────────────────────
  const rs = subRng(seed, 1);
  interface Student {
    program: number;
    age: number;
    scholarship: boolean;
    enrolledOn: number;
    firstTerm: number;
    ability: number;
    procrastination: number;
    diligence: number;
  }
  const students: Student[] = [];
  for (let i = 0; i < N_STUDENTS; i++) {
    const program = categorical(rs, PROGRAM_WEIGHTS);
    const intake = INTAKES[categorical(rs, INTAKES.map((x) => x[3]))];
    const ability = normal(rs);
    // scholarships go somewhat more often to stronger students
    const scholarship = rs.bool(sigmoid(-1.25 + 0.8 * ability));
    students.push({
      program,
      age: round(18 + gamma(rs, 1.6, 2.4), 1),
      scholarship,
      enrolledOn: randomTime(rs, intake[1], intake[2]),
      firstTerm: intake[0],
      ability,
      procrastination: normal(rs),
      diligence: clamp(0.82 + normal(rs, 0, 0.1) + (scholarship ? 0.05 : 0), 0.45, 0.99),
    });
  }

  // ── engagement regime per student and month (sticky disengagement) ───────
  const re = subRng(seed, 2);
  const engaged: boolean[][] = students.map((s) => {
    const m0 = monthIndexOf(s.enrolledOn);
    const seq: boolean[] = [];
    let state = true;
    for (let m = 0; m < 19; m++) {
      if (m < m0) {
        seq.push(false);
        continue;
      }
      if (m > m0) state = state ? re.bool(0.9) : re.bool(0.12);
      seq.push(state);
    }
    return seq;
  });

  // ── course popularity: a persistent level per course that drifts slowly from term to term ──
  const rc = subRng(seed, 4);
  const basePop = COURSES.map(() => normal(rc, 0, POP_SD));
  const logPop: number[][] = COURSES.map((_, c) => {
    const series: number[] = [];
    let x = basePop[c];
    for (let ti = 0; ti < TERMS.length; ti++) {
      x = basePop[c] + POP_PHI * (x - basePop[c]) + normal(rc, 0, POP_DRIFT);
      series.push(x);
    }
    return series;
  });
  // each course has its own fixed offset of its due dates within the assignment cycle
  const phase = COURSES.map(() => 0.55 + 0.4 * rc.next());

  // ── enrolments and submissions ────────────────────────────────────────────
  const rx = subRng(seed, 3);
  interface SubRec {
    s: number;
    c: number;
    t: number;
    grade: number | null;
    late: boolean;
  }
  const subs: SubRec[] = [];
  for (let s = 0; s < N_STUDENTS; s++) {
    const st = students[s];
    const dept = PROGRAMS[st.program].dept;
    for (let ti = st.firstTerm; ti < TERMS.length; ti++) {
      const term = TERMS[ti];
      const offered = COURSES.map((c, i) => (offeredIn(c, term) ? i : -1)).filter((i) => i >= 0);
      let k: number;
      if (term.kind === 'summer') k = rx.bool(SUMMER_P) ? 1 : 0;
      else k = rx.bool(0.5) ? 3 : 4;
      // a disengaged student often skips the whole term
      if (!engaged[s][monthIndexOf(term.start)] && rx.bool(0.6)) k = 0;
      const chosen: number[] = [];
      while (chosen.length < Math.min(k, offered.length)) {
        const w = offered.map((ci) =>
          chosen.includes(ci) ? 0 : detExp(logPop[ci][ti]) * (COURSES[ci].department === dept ? OWN_DEPT : 1),
        );
        chosen.push(offered[categorical(rx, w)]);
      }
      const len = term.end - term.start;
      for (const ci of chosen) {
        const course = COURSES[ci];
        const nA = assignmentsOf(course, term);
        for (let a = 0; a < nA; a++) {
          // evenly spread through the term, so a course's monthly workload is steady
          const due = term.start + ((a + phase[ci]) * len) / nA;
          const m = monthIndexOf(due);
          if (!engaged[s][m] || !rx.bool(st.diligence)) continue;
          const pLate = sigmoid(-1.7 + 1.1 * st.procrastination + 0.3 * (course.difficulty - 3) - (st.scholarship ? 0.5 : 0));
          const late = rx.bool(pLate);
          const t = late
            ? wholeSeconds(due + HOUR_MS + rx.next() * 5 * DAY_MS)
            : wholeSeconds(due - rx.next() * 3.5 * DAY_MS);
          if (t > DATA_END) continue;
          const g = 100 * sigmoid(1.0 + 0.9 * st.ability - 0.35 * (course.difficulty - 3) + normal(rx, 0, 0.5)) - (late ? 7 : 0);
          // recent work is often not graded yet → missing grade (no token)
          const ungraded = t > DATA_END - 21 * DAY_MS ? rx.bool(0.6) : rx.bool(0.04);
          subs.push({ s, c: ci, t, grade: ungraded ? null : round(clamp(g, 0, 100), 1), late });
        }
      }
    }
  }
  sortByTime(subs, (x) => x.t, (x) => x.s * 100 + x.c);

  const tables: RawTable[] = [
    makeTable(
      {
        name: 'students',
        pkey: 'student_id',
        fkeys: {},
        timeCol: null,
        description: 'One row per student. A dimension table: no foreign keys and no time column.',
        columns: [
          col('student_id', 'int', 'Primary key (= row position).'),
          col('program', 'str', 'Degree programme the student is enrolled in.'),
          col('age', 'float', 'Age in years at enrolment.'),
          col('scholarship', 'bool', 'Whether the student holds a scholarship.'),
          col('enrolled_on', 'datetime', 'Date of enrolment (a feature, not the time column).'),
        ],
      },
      students.map((st, i) => ({
        student_id: i,
        program: PROGRAMS[st.program].program,
        age: st.age,
        scholarship: st.scholarship,
        enrolled_on: st.enrolledOn,
      })),
    ),
    makeTable(
      {
        name: 'courses',
        pkey: 'course_id',
        fkeys: {},
        timeCol: null,
        description: 'The course catalogue. A dimension table: no foreign keys and no time column.',
        columns: [
          col('course_id', 'int', 'Primary key (= row position).'),
          col('title', 'str', 'Course title.'),
          col('department', 'str', 'Department that teaches the course.'),
          col('credits', 'int', 'Credit points (3–5).'),
          col('difficulty', 'float', 'Average difficulty rating from course evaluations (1–5).'),
        ],
      },
      COURSES.map((c, i) => ({
        course_id: i,
        title: c.title,
        department: c.department,
        credits: c.credits,
        difficulty: c.difficulty,
      })),
    ),
    makeTable(
      {
        name: 'submissions',
        pkey: 'submission_id',
        fkeys: { student_id: 'students', course_id: 'courses' },
        timeCol: 'submitted_at',
        description: 'One row per submitted assignment: who submitted it, for which course, when, the grade and whether it was late.',
        columns: [
          col('submission_id', 'int', 'Primary key (= row position; submissions are sorted by time).'),
          col('student_id', 'int', 'Foreign key → students: who submitted.'),
          col('course_id', 'int', 'Foreign key → courses: the course of the assignment.'),
          col('submitted_at', 'datetime', 'When the assignment was handed in (the time column of this table).'),
          col('grade', 'float', 'Grade from 0 to 100; missing while not yet graded.'),
          col('late', 'bool', 'Whether the submission came after the due date.'),
        ],
      },
      subs.map((x, i) => ({
        submission_id: i,
        student_id: x.s,
        course_id: x.c,
        submitted_at: x.t,
        grade: x.grade,
        late: x.late,
      })),
    ),
  ];

  return { name: 'school', tables };
}
