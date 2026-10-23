/** Store behavior (BUILD_PLAN §7): defaults, invariants between knobs, storage safety. */
import { beforeEach, describe, expect, it } from 'vitest';
import { isNonDefault, safeStorage, selectSimParams, SIM_DEFAULTS, useSim } from './store';

describe('useSim store', () => {
  beforeEach(() => {
    useSim.getState().resetSimulation();
    useSim.getState().clearCompleted();
  });

  it('starts at the mini defaults (L = 128, w = 16, eval rngSeed = 0, shop/customer-churn, Val)', () => {
    const p = selectSimParams(useSim.getState());
    expect(p).toEqual({ ...SIM_DEFAULTS });
    expect(isNonDefault(p)).toBe(false);
  });

  it('switching task switches the DB and resets the seed and the selected token', () => {
    const s = useSim.getState();
    s.setSeedNode(42);
    s.setSelectedPos(3);
    s.setTask('racing/driver-dnf');
    const t = useSim.getState();
    expect(t.dbName).toBe('racing');
    expect(t.taskId).toBe('racing/driver-dnf');
    expect(t.seedNode).toBeNull();
    expect(t.selectedPos).toBeNull();
  });

  it('clamps L, w and the RNG seed to integer bounds', () => {
    const s = useSim.getState();
    s.setL(100000);
    s.setW(0);
    s.setRngSeed(-5.7);
    const t = useSim.getState();
    expect(t.L).toBe(512);
    expect(t.w).toBe(1);
    expect(t.rngSeed).toBe(0);
    s.setL(Number.NaN);
    expect(useSim.getState().L).toBe(8);
  });

  it('toggles inference-time ablations and does not persist them', () => {
    const s = useSim.getState();
    s.toggleLayer('nbr');
    s.toggleLayer('col');
    expect(useSim.getState().disabledLayers).toEqual(['nbr', 'col']);
    s.toggleLayer('nbr');
    expect(useSim.getState().disabledLayers).toEqual(['col']);
    const persisted = JSON.parse(safeStorage.getItem('rt-sbs:sim') as string);
    expect(persisted.state.disabledLayers).toBeUndefined();
    expect(persisted.state.L).toBe(128);
  });

  it('toggles completed chapters by hand', () => {
    useSim.getState().toggleCompleted('sampling');
    useSim.getState().toggleCompleted('masks');
    expect(useSim.getState().completed).toEqual({ sampling: true, masks: true });
    useSim.getState().toggleCompleted('sampling');
    expect(useSim.getState().completed).toEqual({ masks: true });
  });

  it('safeStorage never throws without window.localStorage (Node)', () => {
    expect(() => safeStorage.setItem('x', '1')).not.toThrow();
    expect(safeStorage.getItem('x')).toBe('1');
    expect(() => safeStorage.removeItem('x')).not.toThrow();
    expect(safeStorage.getItem('x')).toBeNull();
  });

  it('rejects malformed persisted state on rehydrate', async () => {
    safeStorage.setItem(
      'rt-sbs:sim',
      JSON.stringify({ state: { dbName: 'nope', taskId: 'school/student-dropout', L: 'big', split: 'Holdout', theme: 'blue' }, version: 1 }),
    );
    await useSim.persist.rehydrate();
    const s = useSim.getState();
    expect(s.taskId).toBe('school/student-dropout');
    expect(s.dbName).toBe('school'); // kept consistent with the task id
    expect(s.L).toBe(128);
    expect(s.split).toBe('Val');
    expect(s.theme).toBeNull();
  });
});
