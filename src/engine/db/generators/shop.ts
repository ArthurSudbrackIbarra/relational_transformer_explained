/**
 * ShopDB (`shop`): a small e-commerce database. TARGET DB, held out from pretraining
 * (BUILD_PLAN §3.1). It is the running example of chapters 2–13 and 15.
 *
 *   customers(customer_id; name, country, age, is_premium, signup_date)
 *   products (product_id;  name, category, price)
 *   orders   (order_id;    customer_id→customers, product_id→products; order_time [time], quantity, channel)
 *   reviews  (review_id;   customer_id→customers, product_id→products; review_time [time], rating, review_text, verified)
 *
 * Generative process (persistent, transferable signal; the paper finds self-label
 * autocorrelation is the main driver of zero-shot transfer):
 *  - Each customer follows a monthly Markov regime active/inactive (stay-active 0.85,
 *    re-activate 0.15) starting active in the signup month. While active, the customer orders
 *    Poisson(λ_c) times a month, λ_c ~ Gamma, higher for premium customers.
 *  - Products have Zipf-like base popularity that drifts slowly (log-AR(1) random walk) plus
 *    category seasonality; a few products launch mid-span. Customers favour one category.
 *  - About a fifth of orders get a review a few days later (verified purchase); about a quarter
 *    of all reviews are unverified (no matching order) and skew extreme. Ratings follow a latent
 *    product quality plus customer leniency plus noise; review_text is one of 25 templates
 *    correlated with the rating.
 *
 * PK = row position (spec §1.2): orders and reviews are sorted chronologically, then numbered.
 */
import type { RawTable } from '../../types';
import {
  DATA_END,
  DAY_MS,
  NUM_MONTHS,
  calendarMonth,
  categorical,
  clamp,
  col,
  detExp,
  detLog,
  gamma,
  logNormal,
  makeTable,
  monthStart,
  monthIndexOf,
  normal,
  poisson,
  randomTime,
  sortByTime,
  subRng,
  utc,
  wholeSeconds,
} from '../util';
import type { GeneratedDb } from './types';

export const SHOP_SEED = 20240101;

const COUNTRIES = ['BR', 'US', 'DE', 'JP', 'IN', 'FR'] as const;
type Country = (typeof COUNTRIES)[number];
const COUNTRY_WEIGHTS = [0.18, 0.25, 0.14, 0.13, 0.17, 0.13];

/** Country-consistent first/last names, so the `name` text agrees with `country`. */
const NAMES: Record<Country, { first: string[]; last: string[] }> = {
  BR: {
    first: ['Ana', 'João', 'Mariana', 'Pedro', 'Beatriz', 'Lucas', 'Camila', 'Rafael', 'Juliana', 'Gabriel', 'Larissa', 'Thiago'],
    last: ['Souza', 'Oliveira', 'Santos', 'Pereira', 'Costa', 'Almeida', 'Ribeiro', 'Carvalho', 'Lima', 'Araújo'],
  },
  US: {
    first: ['Emily', 'Michael', 'Olivia', 'James', 'Sophia', 'Daniel', 'Ava', 'Ethan', 'Madison', 'Ryan', 'Grace', 'Tyler'],
    last: ['Carter', 'Johnson', 'Miller', 'Davis', 'Brooks', 'Turner', 'Parker', 'Hayes', 'Bennett', 'Collins'],
  },
  DE: {
    first: ['Lukas', 'Anna', 'Felix', 'Lena', 'Jonas', 'Marie', 'Paul', 'Sophie', 'Leon', 'Hannah', 'Moritz', 'Clara'],
    last: ['Müller', 'Schmidt', 'Wagner', 'Becker', 'Hoffmann', 'Schulz', 'Koch', 'Richter', 'Neumann', 'Braun'],
  },
  JP: {
    first: ['Haruto', 'Yui', 'Sota', 'Hina', 'Ren', 'Aoi', 'Yuto', 'Sakura', 'Kaito', 'Mio', 'Riku', 'Yuna'],
    last: ['Sato', 'Suzuki', 'Takahashi', 'Tanaka', 'Watanabe', 'Ito', 'Yamamoto', 'Nakamura', 'Kobayashi', 'Kato'],
  },
  IN: {
    first: ['Priya', 'Arjun', 'Ananya', 'Rohan', 'Diya', 'Vikram', 'Isha', 'Aditya', 'Kavya', 'Rahul', 'Meera', 'Karan'],
    last: ['Sharma', 'Patel', 'Iyer', 'Reddy', 'Gupta', 'Nair', 'Mehta', 'Singh', 'Rao', 'Joshi'],
  },
  FR: {
    first: ['Camille', 'Louis', 'Chloé', 'Hugo', 'Manon', 'Jules', 'Léa', 'Arthur', 'Inès', 'Théo', 'Zoé', 'Nathan'],
    last: ['Dubois', 'Martin', 'Bernard', 'Laurent', 'Moreau', 'Lefebvre', 'Girard', 'Fontaine', 'Roux', 'Blanc'],
  },
};

/** Channel preference tilt per country: [web, app, store]. */
const CHANNELS = ['web', 'app', 'store'] as const;
const CHANNEL_TILT: Record<Country, number[]> = {
  BR: [0.5, 0.35, 0.15],
  US: [0.55, 0.3, 0.15],
  DE: [0.4, 0.2, 0.4],
  JP: [0.25, 0.6, 0.15],
  IN: [0.25, 0.65, 0.1],
  FR: [0.4, 0.2, 0.4],
};

const CATEGORIES = ['electronics', 'books', 'home', 'toys', 'sports'] as const;
type Category = (typeof CATEGORIES)[number];

interface ProductSpec {
  name: string;
  category: Category;
  price: number;
  /** Month index (0 = Jan 2024) from which the product can be ordered. */
  launch: number;
}

/** 30 products, six per category. Three launch mid-span (new products appear in the data). */
const PRODUCTS: ProductSpec[] = [
  { name: 'Wireless Mouse', category: 'electronics', price: 24.99, launch: 0 },
  { name: 'Mechanical Keyboard', category: 'electronics', price: 89.0, launch: 0 },
  { name: 'USB-C Hub', category: 'electronics', price: 39.5, launch: 0 },
  { name: 'Noise-Cancelling Headphones', category: 'electronics', price: 179.0, launch: 0 },
  { name: '27-inch Monitor', category: 'electronics', price: 249.0, launch: 0 },
  { name: 'Smart Watch', category: 'electronics', price: 199.0, launch: 8 },
  { name: 'Python Programming Handbook', category: 'books', price: 42.0, launch: 0 },
  { name: 'Mystery Novel Box Set', category: 'books', price: 34.9, launch: 0 },
  { name: 'Weeknight Dinners Cookbook', category: 'books', price: 27.5, launch: 0 },
  { name: 'Illustrated World Atlas', category: 'books', price: 31.0, launch: 0 },
  { name: 'Fantasy Saga Vol. 1', category: 'books', price: 16.99, launch: 0 },
  { name: 'Mindful Habits Workbook', category: 'books', price: 14.5, launch: 12 },
  { name: 'Ceramic Coffee Mug', category: 'home', price: 12.9, launch: 0 },
  { name: 'Cotton Bath Towel Set', category: 'home', price: 38.0, launch: 0 },
  { name: 'LED Desk Lamp', category: 'home', price: 29.9, launch: 0 },
  { name: 'Non-Stick Frying Pan', category: 'home', price: 44.0, launch: 0 },
  { name: 'Scented Candle Trio', category: 'home', price: 19.5, launch: 0 },
  { name: 'Linen Throw Pillow', category: 'home', price: 22.0, launch: 0 },
  { name: 'Wooden Train Set', category: 'toys', price: 54.0, launch: 0 },
  { name: '1000-Piece Jigsaw Puzzle', category: 'toys', price: 18.99, launch: 0 },
  { name: 'Plush Teddy Bear', category: 'toys', price: 21.5, launch: 0 },
  { name: 'Building Blocks Kit', category: 'toys', price: 64.9, launch: 0 },
  { name: 'Remote Control Car', category: 'toys', price: 49.0, launch: 0 },
  { name: 'Strategy Board Game', category: 'toys', price: 39.9, launch: 10 },
  { name: 'Yoga Mat', category: 'sports', price: 29.0, launch: 0 },
  { name: 'Adjustable Dumbbells', category: 'sports', price: 129.0, launch: 0 },
  { name: 'Trail Running Shoes', category: 'sports', price: 119.0, launch: 0 },
  { name: 'Insulated Water Bottle', category: 'sports', price: 24.5, launch: 0 },
  { name: 'Tennis Racket', category: 'sports', price: 89.9, launch: 0 },
  { name: 'Cycling Helmet', category: 'sports', price: 69.0, launch: 0 },
];

/** Units per order (1..5) by category. */
const QTY_WEIGHTS: Record<Category, number[]> = {
  electronics: [0.8, 0.14, 0.04, 0.015, 0.005],
  books: [0.68, 0.2, 0.08, 0.03, 0.01],
  home: [0.45, 0.3, 0.14, 0.07, 0.04],
  toys: [0.6, 0.24, 0.1, 0.04, 0.02],
  sports: [0.58, 0.28, 0.09, 0.03, 0.02],
};

/** Demand seasonality by calendar month (1..12). */
function seasonality(cat: Category, calMonth: number): number {
  if (cat === 'toys' && (calMonth === 11 || calMonth === 12)) return 1.9;
  if (cat === 'electronics' && calMonth === 11) return 1.5;
  if (cat === 'books' && calMonth === 12) return 1.3;
  if (cat === 'sports' && calMonth >= 4 && calMonth <= 6) return 1.35;
  if (cat === 'home' && calMonth === 1) return 1.25;
  return 1;
}

/** 25 review templates, five per star level (index 0 = 1 star). */
const REVIEW_TEMPLATES: string[][] = [
  [
    'Terrible quality, broke within days.',
    'Complete waste of money, do not buy.',
    'Arrived damaged and support never answered.',
    'Nothing like the description, very disappointed.',
    'Worst purchase I have made this year.',
  ],
  [
    'Disappointing, stopped working after a few weeks.',
    'Cheaply made, expected much better for the price.',
    'Not great, several issues out of the box.',
    'Below average, I would not buy it again.',
    'It works, barely, and feels flimsy.',
  ],
  [
    "It's okay, nothing special.",
    'Average product, does the job.',
    'Decent but a bit overpriced.',
    'Mixed feelings: some good parts, some bad.',
    'Fine for occasional use.',
  ],
  [
    'Good quality, works as described.',
    'Solid purchase, happy with it overall.',
    'Very nice, only minor complaints.',
    'Good value for the money.',
    'Does exactly what I needed, would recommend.',
  ],
  [
    'Absolutely love it, exceeded my expectations!',
    'Excellent quality, five stars.',
    'Perfect, I already ordered a second one.',
    'Best purchase this year, highly recommend!',
    'Fantastic, arrived fast and works flawlessly.',
  ],
];

interface Customer {
  country: Country;
  name: string;
  age: number | null;
  isPremium: boolean;
  signup: number;
  lambda: number;
  favCat: number;
  channelW: number[];
  leniency: number;
  reviewProp: number;
}

export function generateShop(seed = SHOP_SEED): GeneratedDb {
  const N_CUSTOMERS = 80;

  // ── customers ─────────────────────────────────────────────────────────────
  const rc = subRng(seed, 1);
  const usedNames = new Set<string>();
  const customers: Customer[] = [];
  for (let i = 0; i < N_CUSTOMERS; i++) {
    const country = COUNTRIES[categorical(rc, COUNTRY_WEIGHTS)];
    let name = '';
    do {
      name = `${rc.choice(NAMES[country].first)} ${rc.choice(NAMES[country].last)}`;
    } while (usedNames.has(name));
    usedNames.add(name);
    const age = rc.bool(0.1) ? null : Math.min(80, Math.round(18 + gamma(rc, 3, 6.5)));
    const isPremium = rc.bool(0.3);
    // most customers sign up early; some join later (new customers appear in the task tables)
    const u = rc.next();
    const signup =
      u < 0.55
        ? randomTime(rc, utc(2024, 1, 1), utc(2024, 3, 31))
        : u < 0.9
          ? randomTime(rc, utc(2024, 4, 1), utc(2024, 12, 31))
          : randomTime(rc, utc(2025, 1, 1), utc(2025, 2, 28));
    const lambda = gamma(rc, 2.2, 0.74) * (isPremium ? 1.5 : 1);
    const favCat = rc.int(CATEGORIES.length);
    const tilt = CHANNEL_TILT[country];
    const channelW = tilt.map((t) => t * (0.4 + rc.next()));
    const leniency = normal(rc);
    const reviewProp = clamp(0.16 * logNormal(rc, 0, 0.6), 0.02, 0.5);
    customers.push({ country, name, age, isPremium, signup, lambda, favCat, channelW, leniency, reviewProp });
  }

  // ── products: latent quality + Zipf-like popularity with slow drift ───────
  const rp = subRng(seed, 2);
  const nP = PRODUCTS.length;
  const quality = PRODUCTS.map(() => normal(rp, 0, 0.8));
  const rank = rp.shuffle(Array.from({ length: nP }, (_, i) => i));
  const baseLogPop = rank.map((r) => -0.9 * detLog(r + 1) + normal(rp, 0, 0.15));
  // logPop[p][m]: AR(1) drift around the base (slow: persistent sales signal)
  const logPop: number[][] = PRODUCTS.map((_, p) => {
    const series: number[] = [];
    let x = baseLogPop[p];
    for (let m = 0; m < NUM_MONTHS; m++) {
      x = baseLogPop[p] + 0.88 * (x - baseLogPop[p]) + normal(rp, 0, 0.17);
      series.push(x);
    }
    return series;
  });

  // ── orders: monthly Markov regime per customer ────────────────────────────
  const ro = subRng(seed, 3);
  interface OrderRec {
    c: number;
    p: number;
    t: number;
    qty: number;
    channel: string;
  }
  const orders: OrderRec[] = [];
  for (let c = 0; c < N_CUSTOMERS; c++) {
    const cu = customers[c];
    const m0 = monthIndexOf(cu.signup);
    let active = true; // a customer signs up because they want to buy
    for (let m = m0; m < NUM_MONTHS; m++) {
      if (m > m0) active = active ? ro.bool(0.85) : ro.bool(0.15);
      if (!active) continue;
      const start = Math.max(monthStart(m), cu.signup + 1000);
      const end = Math.min(monthStart(m + 1), DATA_END);
      const frac = (end - start) / (monthStart(m + 1) - monthStart(m));
      const holiday = calendarMonth(m) === 11 || calendarMonth(m) === 12 ? 1.25 : 1;
      const n = poisson(ro, cu.lambda * frac * holiday);
      for (let k = 0; k < n; k++) {
        const t = randomTime(ro, start, end);
        const w = PRODUCTS.map((ps, p) =>
          m < ps.launch
            ? 0
            : detExp(logPop[p][m]) * seasonality(ps.category, calendarMonth(m)) * (CATEGORIES[cu.favCat] === ps.category ? 3 : 1),
        );
        const p = categorical(ro, w);
        const qty = 1 + categorical(ro, QTY_WEIGHTS[PRODUCTS[p].category]) + (cu.isPremium && ro.bool(0.1) ? 1 : 0);
        const channel = CHANNELS[categorical(ro, cu.channelW)];
        orders.push({ c, p, t, qty: Math.min(qty, 5), channel });
      }
    }
  }
  sortByTime(orders, (o) => o.t, (o) => o.c * 1000 + o.p);

  // ── reviews: verified (from an order) and unverified (no matching order) ──
  const rr = subRng(seed, 4);
  interface ReviewRec {
    c: number;
    p: number;
    t: number;
    rating: number;
    text: string | null;
    verified: boolean;
  }
  const reviews: ReviewRec[] = [];
  const ratingOf = (c: number, p: number, extreme: boolean): number => {
    if (extreme && rr.bool(0.65)) return quality[p] + 0.5 * customers[c].leniency + normal(rr, 0, 0.8) > 0 ? 5 : 1;
    const z = 3.55 + 0.95 * quality[p] + 0.4 * customers[c].leniency + normal(rr, 0, 0.7);
    return clamp(Math.round(z), 1, 5);
  };
  const textOf = (rating: number): string | null => {
    if (rr.bool(0.08)) return null; // star-only review: a missing value → no token (spec §3.5)
    // text correlates with rating: usually the same star level, sometimes a neighboring one
    let level = rating;
    const u = rr.next();
    if (u < 0.1) level = Math.max(1, rating - 1);
    else if (u < 0.2) level = Math.min(5, rating + 1);
    return rr.choice(REVIEW_TEMPLATES[level - 1]);
  };
  for (const o of orders) {
    if (!rr.bool(customers[o.c].reviewProp)) continue;
    const t = o.t + wholeSeconds(2 * DAY_MS + rr.next() * 19 * DAY_MS); // posted 2–21 days later
    if (t > DATA_END) continue;
    const rating = ratingOf(o.c, o.p, false);
    reviews.push({ c: o.c, p: o.p, t, rating, text: textOf(rating), verified: rr.bool(0.96) });
  }
  const nUnverified = Math.round(reviews.length / 3); // → ≈75 % verified overall
  for (let k = 0; k < nUnverified; k++) {
    const c = rr.int(N_CUSTOMERS);
    const t = randomTime(rr, customers[c].signup + DAY_MS, DATA_END);
    const m = monthIndexOf(t);
    const w = PRODUCTS.map((ps, p) => (m < ps.launch ? 0 : detExp(logPop[p][m])));
    const p = categorical(rr, w);
    const rating = ratingOf(c, p, true);
    reviews.push({ c, p, t, rating, text: textOf(rating), verified: rr.bool(0.05) });
  }
  sortByTime(reviews, (r) => r.t, (r) => r.c * 1000 + r.p);

  // ── assemble RawTables (column order = DataFrame order) ───────────────────
  const tables: RawTable[] = [
    makeTable(
      {
        name: 'customers',
        pkey: 'customer_id',
        fkeys: {},
        timeCol: null,
        description: 'One row per registered customer. A dimension table: no foreign keys and no time column.',
        columns: [
          col('customer_id', 'int', 'Primary key (= row position).'),
          col('name', 'str', 'Full name of the customer.'),
          col('country', 'str', 'Country code: BR, US, DE, JP, IN or FR.'),
          col('age', 'float', 'Age in years; missing for about 10% of customers.'),
          col('is_premium', 'bool', 'Whether the customer has a premium membership.'),
          col('signup_date', 'datetime', 'When the customer created the account (a feature, not the time column).'),
        ],
      },
      customers.map((cu, i) => ({
        customer_id: i,
        name: cu.name,
        country: cu.country,
        age: cu.age,
        is_premium: cu.isPremium,
        signup_date: cu.signup,
      })),
    ),
    makeTable(
      {
        name: 'products',
        pkey: 'product_id',
        fkeys: {},
        timeCol: null,
        description: 'The product catalogue. A dimension table: no foreign keys and no time column.',
        columns: [
          col('product_id', 'int', 'Primary key (= row position).'),
          col('name', 'str', 'Product name.'),
          col('category', 'str', 'Product category: electronics, books, home, toys or sports.'),
          col('price', 'float', 'List price in US dollars.'),
        ],
      },
      PRODUCTS.map((ps, i) => ({ product_id: i, name: ps.name, category: ps.category, price: ps.price })),
    ),
    makeTable(
      {
        name: 'orders',
        pkey: 'order_id',
        fkeys: { customer_id: 'customers', product_id: 'products' },
        timeCol: 'order_time',
        description: 'One row per order line: which customer bought which product, when, how many and through which channel.',
        columns: [
          col('order_id', 'int', 'Primary key (= row position; orders are sorted by time).'),
          col('customer_id', 'int', 'Foreign key → customers: who placed the order.'),
          col('product_id', 'int', 'Foreign key → products: what was bought.'),
          col('order_time', 'datetime', 'When the order was placed (the time column of this table).'),
          col('quantity', 'int', 'Units bought (1–5).'),
          col('channel', 'str', 'Sales channel: web, app or store.'),
        ],
      },
      orders.map((o, i) => ({
        order_id: i,
        customer_id: o.c,
        product_id: o.p,
        order_time: o.t,
        quantity: o.qty,
        channel: o.channel,
      })),
    ),
    makeTable(
      {
        name: 'reviews',
        pkey: 'review_id',
        fkeys: { customer_id: 'customers', product_id: 'products' },
        timeCol: 'review_time',
        description: 'Product reviews written by customers. Verified reviews come from an actual order.',
        columns: [
          col('review_id', 'int', 'Primary key (= row position; reviews are sorted by time).'),
          col('customer_id', 'int', 'Foreign key → customers: the reviewer.'),
          col('product_id', 'int', 'Foreign key → products: the reviewed product.'),
          col('review_time', 'datetime', 'When the review was posted (the time column of this table).'),
          col('rating', 'int', 'Star rating, 1 (worst) to 5 (best).'),
          col('review_text', 'str', 'Short review text; missing for star-only reviews.'),
          col('verified', 'bool', 'Whether the review is a verified purchase.'),
        ],
      },
      reviews.map((r, i) => ({
        review_id: i,
        customer_id: r.c,
        product_id: r.p,
        review_time: r.t,
        rating: r.rating,
        review_text: r.text,
        verified: r.verified,
      })),
    ),
  ];

  return { name: 'shop', tables };
}
