/**
 * Public entry point of the simulation engine (pure TypeScript, browser + Node).
 * Coordination rule: modules only ADD export lines here.
 */
export * from './types';
export * from './rng';
export * from './preprocess';
export * from './sampler';
export * from './masks';
export * from './metrics';
export * from './embeddings';
export * from './model';
export * from './pipeline';
