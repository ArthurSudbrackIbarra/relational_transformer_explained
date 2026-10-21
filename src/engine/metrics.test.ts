/**
 * Tests for metrics.ts. Reference values computed with scikit-learn 1.8.0
 * (`roc_auc_score`, `r2_score`) in python/.venv.
 */
import { describe, expect, it } from 'vitest';
import { auroc, binaryLabel, denormalize, normalize, r2, sigmoid } from './metrics';

describe('auroc (sklearn roc_auc_score)', () => {
  it('matches sklearn, including tied scores', () => {
    expect(auroc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])).toBeCloseTo(0.75, 14);
    expect(auroc([0, 1, 0, 1, 1, 0, 1], [0.5, 0.5, 0.2, 0.9, 0.5, 0.5, -1.0])).toBeCloseTo(0.5833333333333333, 14);
    expect(auroc([1, 0, 1, 0, 0, 1, 1, 0, 1, 0], [2, 2, 2, 1, 1, 3, 0, 0, 2, 2])).toBeCloseTo(0.7, 14);
    expect(auroc([0, 1], [1.0, 1.0])).toBe(0.5); // a tie counts ½
    expect(auroc([1, 1, 0], [-3.0, 0.5, 0.5])).toBeCloseTo(0.25, 14);
    expect(auroc([true, false, true], [3, 1, 2])).toBe(1);
  });

  it('matches sklearn on 200 samples with many ties', () => {
    const y = '0,1,1,1,0,0,0,0,0,0,0,1,0,1,0,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,1,0,1,0,0,1,1,0,1,1,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,1,0,0,0,1,0,1,0,0,0,0,1,1,0,1,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,1,0,1,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,1,0,0,1,0,0,1,0,0,0'
      .split(',')
      .map(Number);
    const s = '-0.6,0.5,1.6,0.3,-0.6,-1.6,0.7,0.8,-0.5,0.2,-1.3,0.5,1.4,1.1,2.3,0.2,0.6,-0.2,1.6,-0.0,0.4,0.1,3.1,-0.1,-2.0,-0.6,0.7,-0.5,1.4,1.0,-0.2,-0.5,-0.0,-0.7,-1.5,1.2,1.6,-1.3,-1.2,-1.8,-1.0,-3.1,-1.1,1.3,-0.3,0.9,0.5,1.8,1.2,-0.4,2.6,0.7,-1.2,1.2,-0.0,2.1,-0.9,0.8,1.9,0.3,0.2,0.2,3.3,-0.7,0.5,-1.1,0.7,-0.0,0.8,0.4,-0.2,-1.4,-0.8,2.8,1.0,-0.8,-1.3,-1.0,-0.0,0.0,-0.7,-1.3,1.4,0.5,-0.4,-0.2,-0.5,-2.9,1.1,-1.1,-1.0,-0.6,1.7,-1.2,-1.4,0.6,1.8,-1.0,0.6,-0.3,0.3,-0.3,0.8,1.2,1.6,0.6,-3.8,0.3,1.0,-0.1,-0.6,1.1,0.4,0.7,-0.5,1.2,-1.1,2.0,0.2,0.2,-0.3,-0.9,0.7,0.3,0.4,-0.1,0.3,2.0,0.9,1.4,-0.4,0.1,-0.3,0.3,-0.5,0.6,-0.2,0.4,-0.3,0.3,-1.1,1.2,-1.7,-0.0,0.2,1.5,1.3,-0.2,-1.4,-0.2,1.0,1.7,1.6,-1.5,3.0,-0.4,-0.9,2.5,-0.0,0.6,0.2,0.8,1.0,-1.4,2.0,0.9,0.6,-0.8,-1.0,0.1,-0.6,-0.8,0.8,1.4,-0.4,0.7,1.4,-1.1,-0.6,0.9,0.7,1.2,2.2,-1.1,-1.5,0.1,0.1,-0.8,-0.5,-1.0,0.4,-1.0,0.4,1.8,-0.5,-0.2,0.4,0.5,0.1,1.6'
      .split(',')
      .map(Number);
    expect(auroc(y, s)).toBeCloseTo(0.794634835066102, 12);
  });

  it('AUROC on logits equals AUROC on sigmoid probabilities (§8.5)', () => {
    const logits = [-2.5, 0.3, 1.7, -0.1, 4.0, -3.3, 0.3];
    const y = [0, 1, 1, 0, 1, 0, 0];
    expect(auroc(y, logits.map(sigmoid))).toBe(auroc(y, logits));
  });

  it('throws like sklearn when only one class is present or a score is NaN', () => {
    expect(() => auroc([1, 1], [0.2, 0.3])).toThrow(/Only one class/);
    expect(() => auroc([0, 1], [NaN, 0.3])).toThrow(/NaN/);
    expect(() => auroc([0, 2], [0.1, 0.3])).toThrow(/0\/1/);
  });

  it('throws like sklearn on ±inf scores and on empty input (check_array, sklearn 1.8.0)', () => {
    // sklearn: ValueError "Input contains infinity or a value too large for dtype('float64')."
    expect(() => auroc([0, 1, 1], [0.1, Infinity, 0.3])).toThrow(/infinity/);
    expect(() => auroc([0, 1, 1], [0.1, -Infinity, 0.3])).toThrow(/infinity/);
    // sklearn: ValueError "Found array with 0 sample(s) ..."
    expect(() => auroc([], [])).toThrow(/0 sample/);
  });

  it('feed it LOGITS (main.py:284, 337): float64 σ saturates and creates ties the logits do not have', () => {
    // sklearn 1.8.0: roc_auc_score([0,1,0,1], [40,50,-3,60]) = 1.0, on expit(...) = 0.75
    const y = [0, 1, 0, 1];
    const logits = [40, 50, -3, 60];
    expect(auroc(y, logits)).toBe(1);
    expect(auroc(y, logits.map(sigmoid))).toBe(0.75);
  });
});

describe('r2 (sklearn r2_score)', () => {
  it('matches sklearn', () => {
    expect(r2([3, -0.5, 2, 7], [2.5, 0.0, 2, 8])).toBeCloseTo(0.9486081370449679, 14);
    expect(r2([0.3, -1.2, 2.2, 0.0, 5.5], [0.1, -0.9, 1.7, 0.4, 4.0])).toBeCloseTo(0.8980710214818062, 14);
    expect(r2([1, 2, 3], [3, 2, 1])).toBeCloseTo(-3, 14);
    expect(r2([1, 2, 3], [1, 2, 3])).toBe(1);
  });

  it('sklearn edge cases: constant y (force_finite), fewer than two samples', () => {
    expect(r2([1, 1, 1], [1, 1, 1])).toBe(1);
    expect(r2([1, 1, 1], [1, 2, 1])).toBe(0);
    expect(r2([1], [2])).toBeNaN();
  });

  it('throws like sklearn on NaN / ±inf in either argument and on empty input (sklearn 1.8.0)', () => {
    // sklearn: ValueError "Input contains NaN." / "Input contains infinity ..." for y_true AND y_pred
    expect(() => r2([1, NaN, 3], [1, 2, 3])).toThrow(/NaN/);
    expect(() => r2([1, 2, 3], [1, NaN, 3])).toThrow(/NaN/);
    expect(() => r2([Infinity, 2, 3], [1, 2, 3])).toThrow(/infinity/);
    expect(() => r2([1, 2, 3], [1, -Infinity, 3])).toThrow(/infinity/);
    expect(() => r2([], [])).toThrow(/0 sample/);
  });

  it('R² in z-space equals R² in raw units (§8.5)', () => {
    const stats = { mean: 37.5, std: 4.2 };
    const z = [0.3, -1.2, 2.2, 0.0, 1.5];
    const zh = [0.1, -0.9, 1.7, 0.4, 1.0];
    const raw = (a: number[]) => a.map((v) => denormalize(v, stats));
    expect(r2(raw(z), raw(zh))).toBeCloseTo(r2(z, zh), 12);
  });
});

describe('sigmoid, denormalize, binaryLabel', () => {
  it('sigmoid is stable and correct', () => {
    expect(sigmoid(0)).toBe(0.5);
    expect(sigmoid(2)).toBeCloseTo(0.8807970779778823, 15);
    expect(sigmoid(-2)).toBeCloseTo(0.11920292202211755, 15);
    expect(sigmoid(1000)).toBe(1);
    expect(sigmoid(-1000)).toBe(0);
    expect(Number.isFinite(sigmoid(-800))).toBe(true);
  });

  it('denormalize inverts the pre.rs z-score with Train stats (§8.4)', () => {
    const stats = { mean: 42.5, std: 12.020815280171307 };
    expect(denormalize(0, stats)).toBe(42.5);
    expect(denormalize(normalize(34, stats), stats)).toBeCloseTo(34, 12);
    expect(normalize(51, stats)).toBeCloseTo(0.7071067811865476, 12);
  });

  it('binaryLabel is int(z > 0) (main.py:336)', () => {
    expect([0.866, -0.866, 0, 1e-9].map(binaryLabel)).toEqual([1, 0, 0, 1]);
  });
});
